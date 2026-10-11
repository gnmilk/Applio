"""CPU 音訊品質旗標與外部 ASR/F0 介面；不載入模型，不修改輸入 WAV。"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.realtime_optimization.benchmark import load_wav, new_run_dir, write_csv
from tools.realtime_optimization.metrics import audio_quality, compare_contours


def delayed_source(source, delay_samples, output_length):
    if delay_samples < 0 or output_length < 0:
        raise ValueError("delay / output_length 不得為負數")
    result = np.zeros(output_length, dtype=np.asarray(source).dtype)
    length = min(len(source), max(0, output_length - delay_samples))
    result[delay_samples:delay_samples + length] = source[:length]
    return result


def compare_transcripts(reference, hypothesis):
    def normalize(text):
        text = unicodedata.normalize("NFKC", text)
        return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))
    a, b = normalize(reference), normalize(hypothesis)
    if not a:
        raise ValueError("原音逐字稿不可為空")
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return {"reference_characters": len(a), "edit_distance": previous[-1], "cer": previous[-1] / len(a),
            "normalization": "NFKC、移除空白及 Unicode 標點；不做繁簡轉換",
            "limitations": "ASR 本身可能誤辨；CER 不能代表發音、聲調、台灣口音或自然度。"}


def compare_timed_contours(source, converted, pitch_shift):
    a, b = np.asarray(source, float), np.asarray(converted, float)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape or not len(a):
        raise ValueError("F0 CSV 必須是相同長度的 time_s,hz 兩欄")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a[:, 0], b[:, 0], atol=1e-6, rtol=0):
        raise ValueError("F0 CSV 時間網格不同或含非有限值；先對齊，不自動內插 V/UV")
    if (np.diff(a[:, 0]) <= 0).any() or (a[:, 0] < 0).any():
        raise ValueError("F0 時間必須非負且嚴格遞增")
    return compare_contours(a[:, 1], b[:, 1], pitch_shift)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--realtime", type=Path)
    parser.add_argument("--block-frame", type=int, default=12000)
    parser.add_argument("--realtime-delay-samples", type=int)
    parser.add_argument("--offline-delay-samples", type=int)
    parser.add_argument("--benchmark-report", type=Path, help="同一輸入的 report.json，用於排除 Realtime warmup")
    parser.add_argument("--f0-source", type=Path)
    parser.add_argument("--f0-offline", type=Path)
    parser.add_argument("--f0-realtime", type=Path)
    parser.add_argument("--pitch-shift", type=float, default=9)
    parser.add_argument("--transcripts", type=Path, help="外部逐字稿 JSON: source/offline/realtime/provenance")
    options = parser.parse_args(argv)
    if options.block_frame <= 0 or any(v is not None and v < 0 for v in
                                     (options.realtime_delay_samples, options.offline_delay_samples)):
        parser.error("block-frame 必須大於零，delay 必須非負")
    source, info = load_wav(options.input, allow_nonfinite=True)
    result = {"status": "PASS", "kind": "CPU_WAV_DIAGNOSTICS", "source": info,
              "signals": {"source": audio_quality(source, 48000, options.block_frame)},
              "asr": {"status": "NOT TESTED", "reason": "未提供外部逐字稿"},
              "f0": {"status": "NOT TESTED", "reason": "未提供時間對齊 F0 CSV"},
              "provenance": "既有輸出的模型與設定未經確認；單靠檔名不能視為相同條件 baseline"}
    warmup_chunks = 0
    if options.benchmark_report:
        report = json.loads(options.benchmark_report.read_text(encoding="utf-8"))
        if report.get("input", {}).get("sha256") != info["sha256"]:
            raise ValueError("Benchmark report 的輸入 hash 不符")
        if report["config"]["voice_changer"]["block_frame"] != options.block_frame:
            raise ValueError("Benchmark report 的 chunk size 不符")
        # The startup count is derived from the same first-repeat raw CSV, not average across repeats.
        with (options.benchmark_report.parent / "chunks.csv").open(encoding="utf-8-sig", newline="") as f:
            import csv
            warmup_chunks = sum(r["repeat"] == "0" and r["warmup"] == "True" for r in csv.DictReader(f))
        result["provenance"] = {"benchmark_report": str(options.benchmark_report), "config": report["config"],
                                "note": "仍需確認選取的輸出 WAV 即為該次產物"}
    for label in ("offline", "realtime"):
        path = getattr(options, label)
        if path is None:
            result["signals"][label] = {"status": "NOT TESTED", "reason": "未提供 WAV"}
            continue
        audio, metadata = load_wav(path, allow_nonfinite=True)
        delay = getattr(options, label + "_delay_samples")
        expected = [i < warmup_chunks for i in range((len(audio) + options.block_frame - 1) // options.block_frame)] if label == "realtime" else None
        result["signals"][label] = {"file": metadata, "declared_source_delay_samples": delay,
            "metrics": audio_quality(audio, 48000, options.block_frame,
                source=delayed_source(source, delay, len(audio)) if delay is not None else None,
                expected_silent=expected),
            "note": "未指定 delay 時，不判定相對輸入的非預期靜音；邊界仍以原始 chunk 網格計算"}
    if options.transcripts:
        transcripts = json.loads(options.transcripts.read_text(encoding="utf-8-sig"))
        result["asr"] = {"status": "SUPPLIED_EXTERNAL", "provenance": transcripts.get("provenance", "未提供")}
        for label in ("offline", "realtime"):
            if label in transcripts:
                result["asr"][label] = compare_transcripts(transcripts["source"], transcripts[label])
    if options.f0_source:
        source_f0 = np.loadtxt(options.f0_source, delimiter=",", skiprows=1, ndmin=2)
        result["f0"] = {"status": "SUPPLIED_EXTERNAL", "note": "外部估計必須使用相同方法、hop 與時間對齊；CSV 不自動補值"}
        for label in ("offline", "realtime"):
            path = getattr(options, "f0_" + label)
            if path:
                result["f0"][label] = compare_timed_contours(source_f0,
                    np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2), options.pitch_shift)
    folder = new_run_dir()
    (folder / "quality.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    lines = ["# 音訊品質旗標報告", "", "CPU WAV 分析已完成；沒有模型推論或實體音訊測試。", "",
             "## 可比較性", "", "既有輸出的模型、Index、Embedder 與參數不可由檔名推定；"
             "同條件 Offline／Realtime baseline 應由 benchmark.py 另行產生。", "",
             "| 訊號 | 非有限值 | Clipping samples | RMS | 最大邊界跳變 |", "|---|---|---|---|---|"]
    for label, signal in result["signals"].items():
        metrics = signal.get("metrics", signal)
        if "rms" in metrics:
            lines.append(f"| {label} | {metrics['nonfinite_samples']} | {metrics['clipped_samples']} | {metrics['rms']} | {metrics['boundary_jump_max']} |")
            rows.extend({"signal": label, "chunk": i, "rms_db": db} for i, db in enumerate(metrics["chunk_rms_db"]))
    lines += ["", "## 評估限制", "", "邊界斜率可能只是正常波形；RMS 靜音可能是 VAD／warmup。"
              "這些客觀指標不等於自然女聲品質。", "",
              f"中文內容介面：{result['asr']['status']}。F0 聲調趨勢介面：{result['f0']['status']}。", "",
              "請人工比較咬字、聲調、台灣口音、長句穩定性、V/UV 與高低音轉換；詳見操作說明。", ""]
    (folder / "quality.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(folder / "quality.csv", rows)
    print(json.dumps({"status": "PASS", "report": str(folder / "quality.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
