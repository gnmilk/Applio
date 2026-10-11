"""以既有 RMVPE 獨立估計三份 WAV 的 F0；不開 Stream、不下載權重。"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.realtime_optimization.benchmark import load_wav, new_run_dir, sha256, write_csv
from tools.realtime_optimization.quality import compare_timed_contours

HOP_SECONDS = .01


def load_predictor(weights, device):
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor
    return RMVPE0Predictor(str(weights), device=device)


def evaluation_start(rows, block_frame, start_seconds=None):
    warmup = sum(str(row["repeat"]) == "0" and str(row["warmup"]).lower() == "true" for row in rows)
    warmup_seconds = warmup * block_frame / 48000
    if start_seconds is None:
        return warmup_seconds + block_frame / 48000
    if not math.isfinite(start_seconds) or start_seconds < warmup_seconds:
        raise ValueError("F0 比較起點不能早於實際 Realtime warmup 結束")
    return start_seconds


def fixed_pitch_shift(inference):
    if inference.get("f0_autotune", False) or inference.get("proposed_pitch", False):
        raise ValueError("固定升調的 F0 比較不支援 autotune／proposed_pitch；請另做動態 contour 評估")
    return inference.get("f0_up_key", 0)


def align_contours(signals, *, start_seconds, delays_ms):
    """正延遲表示輸出較來源晚；只切片，不內插或補 V/UV。"""
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds 必須非負且有限")
    offsets = {"source": 0}
    for label in signals:
        delay = delays_ms.get(label, 0)
        if not math.isfinite(delay) or delay < 0 or delay % 10:
            raise ValueError("delay 必須為非負 10 ms 整數倍；不內插 F0")
        offsets[label] = int(delay / 10)
    start = int(math.ceil(start_seconds / HOP_SECONDS))
    end = min(len(values) - offsets[label] for label, values in signals.items())
    if end <= start:
        raise ValueError("沒有共同有效 F0 區段")
    times = np.arange(start, end) * HOP_SECONDS
    return {label: np.column_stack((times, values[start + offsets[label]:end + offsets[label]]))
            for label, values in signals.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--allow-inference", action="store_true")
    parser.add_argument("--start-seconds", type=float, help="預設為實際 startup 結束後一個 chunk")
    parser.add_argument("--realtime-delay-ms", type=int, default=60)
    parser.add_argument("--offline-delay-ms", type=int, default=0)
    options = parser.parse_args(argv)
    if not options.allow_inference:
        parser.error("RMVPE GPU 評估需要明確 --allow-inference；不開 Stream")
    if ((options.start_seconds is not None and (not math.isfinite(options.start_seconds) or options.start_seconds < 0))
            or any(v < 0 or v % 10 for v in (options.realtime_delay_ms, options.offline_delay_ms))):
        parser.error("start 必須非負；delay 必須是非負 10 ms 整數倍")
    weights = ROOT / "rvc/models/predictors/rmvpe.pt"
    expected = None
    folder = new_run_dir()
    result = {"status": "RUNNING", "benchmark_report": str(options.benchmark_report),
              "tool_sha256": sha256(Path(__file__)),
              "method": "既有 RMVPE FP32，全段獨立波形估計，16k／160 hop／threshold=0.03；不是生成器 pitch cache",
              "weights_sha256": None, "requested_device": "cuda:0", "sample_grid_seconds": HOP_SECONDS,
              "declared_delays_ms": {"offline": options.offline_delay_ms, "realtime": options.realtime_delay_ms},
              "start_seconds": options.start_seconds,
              "limitations": "固定延遲只是近似；SOLA offset、VAD 跳過 context 更新與 F0 估計誤差均影響比較。"
                              "未人工校正每音節，不可將數據當中文聲調正確率、自然度或口音分數。"}
    report_path = folder / "f0_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"F0 評估目錄：{folder}", flush=True)
    try:
        report = json.loads(options.benchmark_report.read_text(encoding="utf-8"))
        source_path = Path(report["input"]["path"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if sha256(source_path) != report["input"]["sha256"]:
            raise ValueError("來源 WAV hash 與 Benchmark 不符")
        paths = {"source": source_path, "offline": options.benchmark_report.parent / "offline.wav",
                 "realtime": options.benchmark_report.parent / "realtime.wav"}
        expected = next((value for key, value in report["asset_sha256"].items()
                         if Path(key).name == "rmvpe.pt"), None)
        if not expected or sha256(weights) != expected:
            raise ValueError("RMVPE 權重 hash 與 Benchmark 不符")
        result["weights_sha256"] = expected
        pitch = fixed_pitch_shift(report["config"]["inference"])
        with (options.benchmark_report.parent / "chunks.csv").open(encoding="utf-8-sig", newline="") as stream:
            start = evaluation_start(list(csv.DictReader(stream)), report["config"]["voice_changer"]["block_frame"], options.start_seconds)
        result["start_seconds"] = start
        # Only these imports may initialize CUDA, after opt-in, hashes and compatibility checks.
        import soxr
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；不以 CPU 假裝 GPU 評估")
        result["device"] = "cuda:0"
        model = load_predictor(weights, "cuda:0")
        signals = {}
        result["files"] = {}
        for label, path in paths.items():
            audio, metadata = load_wav(path)
            result["files"][label] = metadata
            with torch.no_grad():
                signals[label] = model.infer_from_audio(soxr.resample(audio, 48000, 16000, quality="HQ"))
            np.savetxt(folder / (label + "_raw.csv"),
                       np.column_stack((np.arange(len(signals[label])) * HOP_SECONDS, signals[label])),
                       delimiter=",", header="time_s,hz", comments="")
        delays = result["declared_delays_ms"]
        aligned = align_contours(signals, start_seconds=start, delays_ms=delays)
        result["comparison"] = {}
        for label, contour in aligned.items():
            np.savetxt(folder / (label + "_aligned.csv"), contour, delimiter=",", header="time_s,hz", comments="")
            if label != "source":
                result["comparison"][label] = compare_timed_contours(aligned["source"], contour, pitch)
        result["delay_sensitivity"] = {}
        for delay in sorted({max(0, options.realtime_delay_ms + offset) for offset in (-20, -10, 0, 10, 20)}):
            candidate = align_contours(signals, start_seconds=start,
                                       delays_ms={**delays, "realtime": delay})
            result["delay_sensitivity"][str(delay)] = compare_timed_contours(candidate["source"], candidate["realtime"], pitch)
        write_csv(folder / "comparison.csv", [{"signal": label, **value} for label, value in result["comparison"].items()])
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["weights_preserved"] = sha256(weights) == expected if expected and weights.is_file() else None
        if result["weights_preserved"] is False:
            result["status"] = "FAIL"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        lines = ["# 獨立 F0 波形評估", "", f"狀態：**{result['status']}**", "", result["method"], "", result["limitations"], "",
                 "| 輸出 | 共同 voiced frames | V/UV agreement | 中位絕對 cents 誤差 | 趨勢 agreement |",
                 "|---|---|---|---|---|"]
        for label, value in result.get("comparison", {}).items():
            lines.append(f"| {label} | {value['jointly_voiced_frames']} | {value['voicing_agreement']} | "
                         f"{value['median_abs_cents_error']} | {value['slope_direction_agreement']} |")
        lines += ["", "固定延遲與 ±20 ms 敏感度見 JSON；不以最佳數據選延遲。沒有 ASR 或主觀試聽。", ""]
        (folder / "f0_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(report_path), "reason": result.get("error")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
