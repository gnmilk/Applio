"""獨立 Realtime 量測入口；預設僅稽核，不載入模型或開啟音訊裝置。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.realtime_optimization.metrics import audio_quality, percentiles, summarize

RUNS = Path(__file__).parent / "runs"
EXAMPLE = Path(__file__).parent / "preset.example.json"
SAMPLE_RATE = 48000


def new_run_dir(parent=RUNS):
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])
    path.mkdir(exist_ok=False)
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_config(config):
    if set(config) != {"voice_changer", "inference"}:
        raise ValueError("Config 必須只有 voice_changer 和 inference")
    v, p = config["voice_changer"], config["inference"]
    if not isinstance(v, dict) or not isinstance(p, dict):
        raise ValueError("設定區段必須為 JSON object")
    allowed_v = {"block_frame", "cross_fade_overlap_size", "extra_convert_size", "model_path",
                 "index_path", "embedder_model", "embedder_model_custom", "f0_method", "sid",
                 "silent_threshold", "vad_enabled", "vad_sensitivity", "vad_frame_ms",
                 "clean_audio", "clean_strength", "post_process"}
    allowed_p = {"f0_up_key", "index_rate", "protect", "volume_envelope", "f0_autotune",
                 "f0_autotune_strength", "proposed_pitch", "proposed_pitch_threshold", "use_phase_vocoder"}
    if set(v) - allowed_v or set(p) - allowed_p:
        raise ValueError("拒絕未支援設定、錄音、裝置或寫入路徑選項")
    required = {"block_frame", "cross_fade_overlap_size", "extra_convert_size", "model_path",
                "index_path", "embedder_model", "embedder_model_custom"}
    if required - set(v):
        raise ValueError("缺少必要 voice_changer 設定: " + str(sorted(required - set(v))))
    for key, value in {**v, **p}.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"{key} 必須為有限數值")
    n = v["block_frame"]
    if type(n) is not int or n < 480:
        raise ValueError("block_frame 必須為至少 480 的整數 (10 ms)")
    cf, extra = v["cross_fade_overlap_size"], v["extra_convert_size"]
    if type(cf) not in (int, float) or not 0 < cf <= n / SAMPLE_RATE:
        raise ValueError("crossfade 必須大於 0 且不可超過 chunk duration")
    if type(extra) not in (int, float) or not 0 <= extra <= 5:
        raise ValueError("extra context 必須介於 0 和 5 秒")
    if v["embedder_model"] != "custom" or v.get("f0_method", "rmvpe") != "rmvpe":
        raise ValueError("第一階段固定 Custom Embedder + RMVPE，避免下載或換架構")
    if v.get("post_process", False):
        raise ValueError("第一階段不支援 Pedalboard 設定；請關閉 post_process")
    for key in ("vad_enabled", "clean_audio", "post_process", "f0_autotune", "proposed_pitch", "use_phase_vocoder"):
        section = v if key in allowed_v else p
        if key in section and type(section[key]) is not bool:
            raise ValueError(f"{key} 必須是 boolean")
    for key in ("index_rate", "protect", "volume_envelope", "f0_autotune_strength", "clean_strength"):
        section = v if key in allowed_v else p
        if key in section and (type(section[key]) not in (int, float) or not 0 <= section[key] <= 1):
            raise ValueError(f"{key} 必須介於 0 和 1")
    if v.get("vad_frame_ms", 30) not in (10, 20, 30) or v.get("vad_sensitivity", 3) not in (0, 1, 2, 3):
        raise ValueError("VAD frame / sensitivity 不合法")
    for key in ("model_path", "index_path", "embedder_model_custom"):
        if not isinstance(v[key], str) or not v[key].strip():
            raise ValueError(f"{key} 必須為非空白路徑")
    # Match create_pipeline's existing path transformation before validating or hashing.
    # Apply it to a copy so the caller's source configuration stays intact.
    config = json.loads(json.dumps(config))
    config["voice_changer"]["index_path"] = v["index_path"].strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
    return config


def prerequisites(config):
    v = config["voice_changer"]
    embedder = ROOT / v["embedder_model_custom"]
    files = [ROOT / v["model_path"], ROOT / v["index_path"], embedder / "config.json",
             embedder / "pytorch_model.bin", ROOT / "rvc/models/predictors/rmvpe.pt"]
    return [{"path": str(p.resolve()), "exists": p.is_file(),
             "bytes": p.stat().st_size if p.is_file() else None} for p in files]


def audit(config):
    versions = {}
    for name in ("torch", "torchaudio", "numpy", "soundfile", "psutil", "gradio", "sounddevice"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()
    sources = ["rvc/realtime/core.py", "rvc/realtime/pipeline.py", "rvc/realtime/audio.py",
               "rvc/realtime/worker.py", "rvc/realtime/callbacks.py", "rvc/realtime/client.py",
               "rvc/realtime/utils/torch.py", "rvc/realtime/utils/vad.py", "rvc/infer/pipeline.py",
               "rvc/lib/algorithm/synthesizers.py", "rvc/lib/predictors/f0.py",
               "rvc/lib/predictors/RMVPE.py", "rvc/lib/utils.py", "rvc/configs/config.py"]
    sources += sorted(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob("*.py"))
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
            "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "versions": versions, "cuda_build": versions["torch"].split("+cu")[-1]
            if versions["torch"] and "+cu" in versions["torch"] else None,
            "cuda_runtime": None, "gpu_device": None, "driver_version": None,
            "hardware_reason": "稽核不呼叫 CUDA/PortAudio；實際 GPU 結果另見 runtime，實體 Stream 不由此工具執行",
            "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain=v1", "--untracked-files=all"),
            "source_sha256": {name: sha256(ROOT / name) for name in sources},
            "config_sha256": {name: sha256(ROOT / name) for name in ("assets/config.json", "ui_settings.json")
                              if (ROOT / name).is_file()}, "prerequisites": prerequisites(config)}


def make_chunks(audio, block_frame):
    if block_frame <= 0:
        raise ValueError("block_frame 必須大於零")
    return [(np.pad(audio[i:i + block_frame], (0, max(0, block_frame - len(audio[i:i + block_frame])))),
             min(block_frame, len(audio) - i)) for i in range(0, len(audio), block_frame)]


def load_wav(path, *, allow_nonfinite=False):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(x) or (not allow_nonfinite and not np.isfinite(x).all()):
        raise ValueError("輸入 WAV 為空或含非有限數值")
    original = {"path": str(Path(path).resolve()), "sha256": sha256(path),
                "sample_rate": sr, "channels": x.shape[1], "samples": len(x),
                "nonfinite_samples_before_preprocessing": int((~np.isfinite(x)).sum())}
    x = x.mean(axis=1, dtype=np.float32)
    if sr != SAMPLE_RATE:
        import soxr
        x = soxr.resample(x, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    original["preprocessing"] = "聲道平均為 mono；非 48k 以 soxr HQ 離線重採樣；不調整音量"
    return x, original


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(folder, result):
    (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = result.get("performance")
    lines = ["# Realtime 量測報告", "", f"狀態：**{result['status']}**", "",
             result["description"], "", "## 正常 Pipeline 量測", "",
             "此項不包含音訊裝置、worker IPC、Discord、OBS、瀏覽器或實際端到端延遲。", ""]
    if summary:
        lines += ["| 指標 | 值 |", "|---|---|",
                  *[f"| {key} | {summary.get(key)} |" for key in
                    ("chunks", "warmup_chunks", "latency_ms", "deadline_ms", "deadline_misses",
                     "deadline_miss_rate", "realtime_factor", "cpu_core_equivalents")], ""]
        lines += ["completed_passes 另記錄含整輪 startup 與最後 CUDA fence 的完成時間／RTF；"
                  "不混入上表的 steady-state worker 呼叫時間。", ""]
    else:
        lines += ["NOT TESTED：沒有實際模型效能數據。", ""]
    lines += ["## 診斷量測", "", "diagnostic_stage_summary 的 host 欄位包含非同步排程；"
              "CUDA 欄位是該 stream 的時間跨度，可含 CPU 等待空洞，不能當作純 kernel 時間。"
              "階段 inclusive 互相重疊，不能直接相加。所有診斷同步與 trace 開銷均排除於正常效能結果。", "",
              "## 音質", "", "JSON 包含 NaN、clipping、chunk RMS、靜音及邊界跳變旗標。"
              "客觀旗標不能代表中文正確度、聲調、台灣口音或自然年輕女聲。"
              "ASR/F0 未提供時標記 NOT TESTED；沒有安裝模型。", "",
              "## 可重現性", "", "report.json 記錄參數、來源 hash、輸入 hash、環境與限制；"
              "chunks.csv 為正常量測，stages.csv（若有）為獨立診斷。"
              "RVC 生成器含隨機項；使用固定 seed 與相同輸入順序，跨裝置仍不保證 bitwise 一致。", ""]
    if result.get("error"):
        lines += ["## 未執行／失敗原因", "", result["error"], ""]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _reset(vc):
    vc.vc_model.realloc(vc.block_frame, vc.extra_frame, vc.crossfade_frame, vc.sola_search_frame)
    vc.generate_strength()
    if vc.vc_model.vad is not None:
        from rvc.realtime.utils.vad import VADProcessor
        vad = vc.vc_model.vad
        vc.vc_model.vad = VADProcessor(sensitivity_mode=vc._benchmark_vad_sensitivity,
            sample_rate=vad.sample_rate, frame_duration_ms=vc._benchmark_vad_frame_ms)


def _pass(vc, chunks, params, repeat, *, profiler=None, completion_fence=None):
    rows, output, stages = [], [], []
    import psutil
    process = psutil.Process()
    rss = process.memory_info().rss
    pass_start = time.perf_counter_ns()
    for i, (chunk, valid) in enumerate(chunks):
        warmup = vc.vc_model.warmup_blocks > 0
        cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
        y, _ = vc.process_audio(chunk, **params)
        wall_ms = (time.perf_counter_ns() - start) / 1e6
        cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
        if y.shape != chunk.shape:
            raise RuntimeError(f"輸出長度不一致：{y.shape} / {chunk.shape}")
        rows.append({"repeat": repeat, "chunk": i, "wall_ms": wall_ms, "cpu_ms": cpu_ms,
                     "warmup": warmup, "valid_samples": valid})
        output.append(y[:valid].copy())
        rss = max(rss, process.memory_info().rss)
        if profiler:
            for row in profiler.finish_chunk(i):
                row.update(repeat=repeat, warmup=warmup)
                stages.append(row)
    if completion_fence is not None:
        completion_fence()
    completed_ms = (time.perf_counter_ns() - pass_start) / 1e6
    valid_ms = sum(valid for _, valid in chunks) * 1000 / SAMPLE_RATE
    completed = {"repeat": repeat, "wall_ms": completed_ms,
                 "realtime_factor_including_tail": completed_ms / valid_ms if valid_ms else None,
                 "note": "整輪完成時間，包含 startup warmup、輸出保留／RSS 取樣及最後 CUDA fence；"
                         "不包含模型初始化，不與 steady-state chunk service time 混用"}
    return rows, np.concatenate(output), stages, rss, completed


def offline_reference(vc, audio, config, seed):
    """官方 Offline Pipeline，重用相同權重；保留其濾波、padding、F0 與正規化。"""
    import soxr
    import torch
    from rvc.infer.pipeline import Pipeline
    v, p = config["voice_changer"], config["inference"]
    rt = vc.vc_model.pipeline
    offline = Pipeline(rt.tgt_sr, rt.vc.config)
    x = soxr.resample(audio, SAMPLE_RATE, 16000, quality="HQ").astype(np.float32)
    maximum = np.abs(x).max() / .95
    if maximum > 1:
        x /= maximum
    torch.manual_seed(seed)
    y = offline.pipeline(model=rt.hubert_model, net_g=rt.vc.net_g, sid=v.get("sid", 0), audio=x,
                         pitch=p.get("f0_up_key", 0), f0_method="rmvpe", file_index=v["index_path"],
                         index_rate=p.get("index_rate", .5), pitch_guidance=rt.use_f0,
                         volume_envelope=p.get("volume_envelope", 1), version=rt.version,
                         protect=p.get("protect", .5), f0_autotune=p.get("f0_autotune", False),
                         f0_autotune_strength=p.get("f0_autotune_strength", 1),
                         proposed_pitch=p.get("proposed_pitch", False),
                         proposed_pitch_threshold=p.get("proposed_pitch_threshold", 155))
    return soxr.resample(y, rt.tgt_sr, SAMPLE_RATE, quality="HQ").astype(np.float32)


def run_model(options, config, folder, result):
    # Imports happen only after the explicit gate and prerequisite check.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import soundfile as sf
    import torch
    from rvc.realtime.core import VoiceChanger
    from tools.realtime_optimization.instrumentation import instrument_voice_changer

    audio, metadata = load_wav(options.input)
    result["input"] = metadata
    v, p = config["voice_changer"], config["inference"]
    chunks = make_chunks(audio, v["block_frame"])
    torch.manual_seed(options.seed)
    vc = VoiceChanger(**v, record_audio=False)
    vc._benchmark_vad_sensitivity = v.get("vad_sensitivity", 3)
    vc._benchmark_vad_frame_ms = v.get("vad_frame_ms", 30)
    if vc.vc_model.pipeline.index.index is None:
        raise RuntimeError("Index 載入失敗，拒絕以無 Index 結果充當 baseline")
    if vc.vc_model.pipeline.version != "v2":
        raise RuntimeError("第一階段只支援目前 RVC v2 情境")
    device = str(vc.device)
    cuda = device.startswith("cuda")
    result["runtime"] = {"device": device, "dtype": str(vc.vc_model.dtype),
                         "model_sr": vc.vc_model.pipeline.tgt_sr, "cuda_build": torch.version.cuda,
                         "index_vectors": vc.vc_model.pipeline.index.index.ntotal,
                         "threads": torch.get_num_threads(), "seed": options.seed,
                         "gpu_name": torch.cuda.get_device_name(vc.device) if cuda else None}
    # Initialization/cache warmup is not measured. Then reset streaming state for each actual pass.
    with torch.no_grad():
        for i in range(vc.vc_model.warmup_blocks + options.warmup_chunks):
            vc.process_audio(chunks[i % len(chunks)][0], **p)
        if cuda:
            torch.cuda.synchronize(vc.device)
        result["warmup_policy"] = "初始化 warmup 後，每輪重設 buffer；各輪開頭實際 Realtime warmup "
        result["warmup_policy"] += "仍保留於 WAV，從 steady-state latency 統計排除"
        all_rows, rss, completed_passes = [], 0, []
        peaks = []
        first_audio = None
        for repeat in range(options.repeats):
            _reset(vc)
            torch.manual_seed(options.seed + repeat)
            if cuda:
                torch.cuda.synchronize(vc.device)
                torch.cuda.reset_peak_memory_stats(vc.device)
            fence = (lambda: torch.cuda.synchronize(vc.device)) if cuda else None
            rows, converted, _, peak_rss, completed = _pass(vc, chunks, p, repeat, completion_fence=fence)
            completed_passes.append(completed)
            all_rows.extend(rows)
            rss = max(rss, peak_rss)
            if cuda:
                peaks.append({"allocated_bytes": torch.cuda.max_memory_allocated(vc.device),
                              "reserved_bytes": torch.cuda.max_memory_reserved(vc.device)})
            if repeat == 0:
                first_audio = converted
                sf.write(folder / "realtime.wav", converted, SAMPLE_RATE, subtype="FLOAT")
                result["audio_quality"] = audio_quality(converted, SAMPLE_RATE, vc.block_frame,
                    source=audio, expected_silent=[r["warmup"] for r in rows])
        result["performance"] = summarize(all_rows, SAMPLE_RATE, vc.block_frame)
        result["completed_passes"] = completed_passes
        result["memory"] = {"rss_observed_peak_bytes": rss, "gpu_peak_per_repeat": peaks or None,
                            "note": "RSS 每 chunk 完成後取樣，包含輸出保留，並非精確 allocator peak；"
                                    "GPU peak 包含模型／Index 常駐記憶體，僅正常 pass"}
        result["index_backend_after_normal_run"] = "faiss_cpu" if vc.vc_model.pipeline.index.faiss_cpu else "torch_bruteforce"
        write_csv(folder / "chunks.csv", all_rows)
        if options.profile:
            _reset(vc)
            torch.manual_seed(options.seed)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            import contextlib
            trace_ctx = torch.profiler.profile(activities=activities, record_shapes=True,
                        profile_memory=True) if options.trace else contextlib.nullcontext()
            with trace_ctx as trace, instrument_voice_changer(vc, trace=options.trace) as profiler:
                _, profiled_audio, stages, _, _ = _pass(vc, chunks, p, 0, profiler=profiler)
            write_csv(folder / "stages.csv", stages)
            result["diagnostic_stage_summary"] = stage_summary(stages)
            result["profile_output_comparison"] = {
                "max_abs_difference": float(np.max(np.abs(profiled_audio - first_audio)))
                if np.isfinite(profiled_audio).all() and np.isfinite(first_audio).all() else None,
                "bitwise_equal": bool(np.array_equal(profiled_audio, first_audio)),
                "note": "相同 seed/buffer；同步可能影響排程，不影響演算法。差異須進一步調查。"}
            if trace is not None:
                trace.export_chrome_trace(str(folder / "trace.json"))
        if options.offline_reference:
            if v.get("clean_audio"):
                raise ValueError("Offline A/B 需要關閉 clean_audio，確保相同後處理條件")
            offline = offline_reference(vc, audio, config, options.seed)
            sf.write(folder / "offline.wav", offline, SAMPLE_RATE, subtype="FLOAT")
            result["offline_quality"] = audio_quality(offline, SAMPLE_RATE, vc.block_frame)
            result["offline_reference"] = "相同模型／Embedder／Index／參數；官方 Offline Pipeline，"
            result["offline_reference"] += "其濾波、F0、FAISS retrieval、RMS 與輸出正規化有別於 Realtime；不是效能對照組"
    result["status"] = "PASS" if result["performance"]["chunks"] else "NOT TESTED"
    if not result["performance"]["chunks"]:
        result["error"] = "輸入全在 Realtime startup warmup 內；請使用更長 WAV"
    result["asr"] = {"status": "NOT TESTED", "reason": "未指定既有中文 ASR；不下載模型"}
    result["f0_quality"] = {"status": "NOT TESTED", "reason": "使用 quality.py 載入獨立、時間對齊 F0 CSV"}
    result["audio_quality"]["alignment_assumption"] = "來源與輸出暫以 chunk index 對照；SOLA/context 的內容位移尚未校正，靜音旗標需複查"
    result["performance"]["cuda_completion_note"] = "正常量測對照 worker 的 process_audio 呼叫時間；speech 輸出 D2H 會等待。"
    result["performance"]["cuda_completion_note"] += "silent/warmup 輸出 CPU zeros 可在 GPU 工作完成前返回；其成本可能延至下一 chunk，不能視為零 GPU 成本。"
    if result["audio_quality"]["nonfinite_samples"]:
        result["status"] = "FAIL"
        result["error"] = "Realtime 輸出含非有限值，已保留原始 FLOAT WAV 供診斷"


def stage_summary(rows):
    result = {}
    for stage in sorted({r["stage"] for r in rows}):
        selected = [r for r in rows if r["stage"] == stage and not r["warmup"]]
        # Sum multiple calls in the same chunk, then compute per-chunk percentiles.
        entry = {"observed_chunks": len({r["chunk"] for r in selected})}
        for field in ("host_inclusive_ms", "host_exclusive_ms", "cuda_stream_span_ms"):
            values = {}
            for row in selected:
                if row[field] is not None:
                    values[row["chunk"]] = values.get(row["chunk"], 0) + row[field]
            entry[field] = percentiles(list(values.values()))
        result[stage] = entry
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "self-test", "run"), default="audit")
    parser.add_argument("--config", type=Path, default=EXAMPLE)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--allow-inference", action="store_true", help="明確允許 WDF_VIOLATION 後的模型／GPU 測試；不開 Stream")
    parser.add_argument("--profile", action="store_true", help="另跑階段診斷，不計入正常 latency")
    parser.add_argument("--trace", action="store_true", help="診斷 pass 輸出 PyTorch trace；需要 --profile")
    parser.add_argument("--offline-reference", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-chunks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261010)
    options = parser.parse_args(argv)
    if options.mode == "run" and not options.allow_inference:
        parser.error("GPU／模型量測尚未恢復；需明確使用 --allow-inference")
    if options.mode == "run" and not options.input:
        parser.error("run 需要 --input 固定 WAV")
    if options.trace and not options.profile:
        parser.error("--trace 需要 --profile")
    if options.repeats < 1 or options.warmup_chunks < 1:
        parser.error("repeats / warmup-chunks 必須至少為 1")
    os.chdir(ROOT)
    config = validate_config(json.loads(options.config.read_text(encoding="utf-8-sig")))
    folder = new_run_dir()
    result = {"schema_version": 1, "status": "NOT TESTED", "description": "固定 WAV 的純 Pipeline 模擬；無實體音效裝置。",
              "config": config, "config_source": str(options.config), "environment": audit(config),
              "performance": None, "end_to_end_latency_ms": None, "prototype": None,
              "run_options": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(options).items()}}
    try:
        if options.mode == "audit":
            result["error"] = "只執行安全稽核；沒有載入模型、CUDA 或 PortAudio"
        elif options.mode == "self-test":
            import soundfile as sf
            t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
            audio = (.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
            chunks = make_chunks(audio, 12000)
            converted = np.concatenate([x for x, _ in chunks])[:len(audio)]
            np.testing.assert_array_equal(converted, audio)
            sf.write(folder / "fixture.wav", audio, SAMPLE_RATE, subtype="FLOAT")
            result["status"] = "PASS"
            result["description"] = "CPU 合成音的分塊／串接／輸出／報告測試；不是 RVC benchmark 或女声音質測試。"
            result["audio_quality"] = audio_quality(converted, SAMPLE_RATE, 12000, source=audio)
            write_csv(folder / "chunks.csv", [{"chunk": i, "valid_samples": valid} for i, (_, valid) in enumerate(chunks)])
        else:
            missing = [r["path"] for r in result["environment"]["prerequisites"] if not r["exists"]]
            if missing:
                raise FileNotFoundError("缺少本機資產，拒絕下載：" + "; ".join(missing))
            # Fingerprint local weights outside timed sections. Never deserialize during audit.
            result["asset_sha256"] = {r["path"]: sha256(r["path"]) for r in result["environment"]["prerequisites"]}
            result["status"] = "RUNNING"
            write_report(folder, result)
            print(f"模型量測目錄：{folder}", flush=True)
            run_model(options, config, folder, result)
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        after = {name: sha256(ROOT / name) for name in result["environment"]["config_sha256"]}
        result["existing_configs_preserved"] = after == result["environment"]["config_sha256"]
        if not result["existing_configs_preserved"]:
            result["status"] = "FAIL"
            result["error"] = "既有設定檔 hash 改變；請調查並行 UI 操作或非預期寫入"
        write_report(folder, result)
    print(json.dumps({"status": result["status"], "report": str(folder / "report.md"),
                      "reason": result.get("error")}, ensure_ascii=False))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
