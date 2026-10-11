"""CPU-only report math. Signal flags are diagnostics, not perceptual scores."""
import numpy as np


def percentiles(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"p50": None, "p95": None, "p99": None}
    return dict(zip(("p50", "p95", "p99"), map(float, np.percentile(values, [50, 95, 99]))))


def summarize(rows, sample_rate, block_frame):
    steady = [r for r in rows if not r["warmup"]]
    elapsed = sum(r["wall_ms"] for r in steady)
    audio_ms = sum(r["valid_samples"] for r in steady) * 1000 / sample_rate
    deadline = block_frame * 1000 / sample_rate
    return {
        "chunks": len(steady), "warmup_chunks": len(rows) - len(steady),
        "latency_ms": percentiles([r["wall_ms"] for r in steady]),
        "warmup_latency_ms": percentiles([r["wall_ms"] for r in rows if r["warmup"]]),
        "deadline_ms": deadline,
        "deadline_misses": sum(r["wall_ms"] > deadline for r in steady),
        "deadline_miss_rate": (sum(r["wall_ms"] > deadline for r in steady) / len(steady)
                               if steady else None),
        "realtime_factor": elapsed / audio_ms if audio_ms else None,
        "cpu_ms": percentiles([r["cpu_ms"] for r in steady]),
        "cpu_core_equivalents": sum(r["cpu_ms"] for r in steady) / elapsed if elapsed else None,
        "note": "只計服務時間 deadline，不含裝置、IPC、scheduler 或網路。"
                "百分位使用線性內插；少量樣本不能建立可靠的尾端分布。",
    }


def audio_quality(audio, sample_rate, block_frame, *, source=None, expected_silent=None,
                  silence_db=-60, active_db=-40, jump_threshold=0.1):
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    clean = np.where(finite, x, 0.0)  # diagnostics only; never modify returned/exported audio
    n = len(x)
    chunks = [clean[i:i + block_frame] for i in range(0, n, block_frame)]
    rms = [float(np.sqrt(np.mean(c * c))) for c in chunks]
    db = [float(20 * np.log10(max(v, 1e-12))) for v in rms]
    silent = [i for i, value in enumerate(db) if value < silence_db]
    unexpected = []
    if source is not None:
        source = np.asarray(source, dtype=np.float64).reshape(-1)
        for i in silent:
            s = source[i * block_frame:(i + 1) * block_frame]
            expected = bool(expected_silent and i < len(expected_silent) and expected_silent[i])
            if len(s) and np.isfinite(s).all() and not expected:
                if 20 * np.log10(max(float(np.sqrt(np.mean(s * s))), 1e-12)) > active_db:
                    unexpected.append(i)
    boundaries = np.arange(block_frame, n, block_frame, dtype=int)
    jumps = np.abs(clean[boundaries] - clean[boundaries - 1])
    delta = np.abs(np.diff(clean))
    ratio_base = float(np.percentile(delta, 95)) if len(delta) else 0.0
    return {
        "samples": n, "duration_s": n / sample_rate,
        "nonfinite_samples": int((~finite).sum()),
        "clipped_samples": int((np.abs(x[finite]) >= 1).sum()),
        "finite_peak": float(np.abs(x[finite]).max()) if finite.any() else None,
        "rms": float(np.sqrt(np.mean(clean * clean))) if n else None,
        "chunk_rms_db": db, "silent_chunks": silent,
        "unexpected_silent_chunks": unexpected if source is not None else None,
        "boundary_jump_max": float(jumps.max()) if len(jumps) else None,
        "boundary_jump": percentiles(jumps),
        "boundary_flags": [int(boundaries[i]) for i, v in enumerate(jumps) if v > jump_threshold],
        "boundary_to_derivative_p95_ratio": (float(jumps.max()) / ratio_base
                                             if len(jumps) and ratio_base > 0 else None),
        "thresholds": {"silence_db": silence_db, "active_input_db": active_db,
                       "boundary_jump": jump_threshold},
        "limitations": "邊界跳變可能是正常波形斜率，靜音可能來自 VAD。"
                        "有限值 peak 不包含 NaN；只有診斷 RMS 將非有限值視為零，原 WAV 不改寫。"
                        "這些旗標不能量測中文正確度、口音或自然度。",
    }


def compare_contours(source, converted, pitch_shift=0):
    a, b = np.asarray(source, float), np.asarray(converted, float)
    if a.ndim != 1 or b.shape != a.shape or not len(a):
        raise ValueError("F0 contours must be nonempty, aligned arrays of equal length")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("F0 must be finite and nonnegative; zero means unvoiced")
    va, vb = a > 0, b > 0
    voiced = va & vb
    error = 1200 * np.log2(b[voiced] / a[voiced]) - pitch_shift * 100
    pairs = voiced[:-1] & voiced[1:]
    # Compare log-F0 slopes only inside jointly voiced regions, allowing a 1-cent plateau.
    ia = np.flatnonzero(pairs)
    da = 1200 * np.log2(a[ia + 1] / a[ia])
    db = 1200 * np.log2(b[ia + 1] / b[ia])
    sa = np.where(np.abs(da) <= 1, 0, np.sign(da))
    sb = np.where(np.abs(db) <= 1, 0, np.sign(db))
    return {
        "frames": len(a), "jointly_voiced_frames": int(voiced.sum()),
        "voicing_agreement": float((va == vb).mean()),
        "median_abs_cents_error": float(np.median(np.abs(error))) if len(error) else None,
        "slope_direction_agreement": float((sa == sb).mean()) if len(ia) else None,
        "pitch_shift_semitones": pitch_shift,
        "limitations": "必須使用獨立且時間對齊的 F0 估計；這是 contour 診斷，"
                        "不能作為中文聲調分類器或主觀音質分數。",
    }
