"""Offline local-weight smoke test; no downloads, streams, recordings or config writes."""
import os
import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from rvc.realtime.core import VoiceChanger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-inference", action="store_true")
    options = parser.parse_args()
    if not options.allow_inference:
        parser.error("Inference smoke test disabled after the interrupted hardware test. Review the WDF_VIOLATION incident before explicitly enabling it.")
    model_dir = ROOT / "logs/TaiwanFemale3_CNHubert_v3"
    embedder = ROOT / "rvc/models/embedders/embedders_custom/chinese_hubert_base_custom"
    model = model_dir / "TaiwanFemale3_CNHubert_v3_90e_28890s.pth"
    index = model_dir / "TaiwanFemale3_CNHubert_v3.index"
    for path in (model, index, embedder / "config.json", embedder / "pytorch_model.bin",
                 ROOT / "rvc/models/predictors/rmvpe.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"Offline prerequisite missing: {path}")
    checkpoint = torch.load(model, map_location="cpu", weights_only=True)
    metadata = {key: checkpoint.get(key) for key in ("version", "sr", "f0", "vocoder", "embedder_model")}
    print("Checkpoint metadata:", json.dumps(metadata, ensure_ascii=False), flush=True)
    assert Path(metadata["embedder_model"]).name == embedder.name, "Use the original trained embedder"
    del checkpoint
    vc = VoiceChanger(block_frame=12000, cross_fade_overlap_size=0.05, extra_convert_size=2.5,
                      model_path=str(model), index_path=str(index), f0_method="rmvpe",
                      embedder_model="custom", embedder_model_custom=str(embedder),
                      silent_threshold=-60, vad_enabled=True)
    print("Loaded local Chinese HuBERT + RVC + Index + RMVPE", flush=True)
    assert vc.vc_model.pipeline.version == "v2"
    assert vc.vc_model.pipeline.tgt_sr == 40000
    assert vc.vc_model.embedder_model_custom == str(embedder)
    assert vc.vc_model.pipeline.hubert_model.config._name_or_path == str(embedder)
    assert vc.vc_model.pipeline.index is not None
    params = dict(f0_up_key=9, index_rate=0.75, protect=0.5, volume_envelope=0.63, use_phase_vocoder=True)
    zeros = np.zeros(12000, dtype=np.float32)
    warmup = vc.vc_model.warmup_blocks
    with torch.no_grad():
        for _ in range(warmup + 2):
            out, _ = vc.process_audio(zeros, **params)
            assert np.all(out == 0), "VAD/silence/warmup must output exact zeros"
        print("Warmup/VAD/silence PASS:", warmup, "warmup blocks", flush=True)
        # Bypass speech classification ONLY on this in-memory test object to
        # exercise conversion; a synthetic tone is not a speech-quality test.
        vad = vc.vc_model.vad
        vc.vc_model.vad = None
        t = np.arange(12000, dtype=np.float32) / 48000
        tone = (0.04 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        peaks = []
        for _ in range(12):
            out, _ = vc.process_audio(tone, **params)
            assert out.shape == (12000,)
            assert np.isfinite(out).all()
            peaks.append(float(np.max(np.abs(out))))
        vc.vc_model.vad = vad
        # Flush the RMS history before testing silence-to-speech residue.
        for _ in range(4):
            out, _ = vc.process_audio(zeros, **params)
        assert np.all(out == 0)
        assert torch.count_nonzero(vc.sola_buffer).item() == 0
    result = {"status": "PASS", "kind": "offline synthetic inference, NOT real microphone realtime",
              "model": str(model), "embedder": str(embedder), "metadata": metadata,
              "device": str(vc.device), "warmup_blocks": warmup,
              "synthetic_peak": max(peaks), "finite": True, "post_speech_silence_zero": True,
              "preset": {"chunk_ms": 250, "crossfade_s": 0.05, "extra_s": 2.5, **params}}
    (Path(__file__).parent / "evidence/offline-smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
