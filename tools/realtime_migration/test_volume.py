"""Realtime migration regression tests; no real audio streams or GPU inference."""
import sys
import types
import threading
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from rvc.realtime.core import Realtime, VoiceChanger


class VolumeTests(unittest.TestCase):
    def make_realtime(self, *, vad=None, threshold=0, warmup=0):
        rt = Realtime.__new__(Realtime)
        rt.device, rt.dtype = "cpu", torch.float32
        rt.resample_in = rt.resample_out = lambda x: x
        rt.audio_buffer = torch.zeros(16)
        rt.convert_buffer = torch.zeros(16)
        rt.pitch_buffer = rt.pitchf_buffer = torch.zeros(2)
        rt.convert_feature_size_16k = rt.silence_front = rt.skip_head = 0
        rt.return_length = rt.block_frame_16k = 16
        rt.warmup_blocks, rt.vad, rt.input_sensitivity = warmup, vad, threshold
        rt.board = rt.reduced_noise = None
        rt.pipeline = types.SimpleNamespace(voice_conversion=lambda *a, **k: torch.full((16,), 0.5))
        return rt

    def test_normal_speech_keeps_model_amplitude(self):
        rt = self.make_realtime()
        out, vol, silent = rt.inference(np.full(16, 0.02, dtype=np.float32), volume_envelope=0.63)
        self.assertFalse(silent)
        self.assertAlmostEqual(vol, 0.02)
        torch.testing.assert_close(out, torch.full((16,), 0.5))

    def test_vad_threshold_and_warmup_keep_silence(self):
        for options in ({"threshold": 0.1}, {"vad": types.SimpleNamespace(is_speech=lambda x: False)}, {"warmup": 2}):
            with self.subTest(options=options):
                out, _, silent = self.make_realtime(**options).inference(np.full(16, 0.02, dtype=np.float32))
                self.assertTrue(silent)
                self.assertEqual(torch.count_nonzero(out).item(), 0)

    def test_silent_block_clears_crossfade_residue(self):
        vc = VoiceChanger.__new__(VoiceChanger)
        vc.vc_model = types.SimpleNamespace(inference=lambda *a: (torch.ones(16), 0.0, True))
        vc.sola_buffer = torch.ones(4)
        vc.record_audio = False
        out, _ = vc.process_audio(np.zeros(16, dtype=np.float32))
        self.assertTrue(np.all(out == 0))
        self.assertEqual(torch.count_nonzero(vc.sola_buffer).item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
