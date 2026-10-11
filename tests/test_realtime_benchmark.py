"""Safe benchmark regressions: CPU fixtures only; never load trained weights."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class BenchmarkMetricsTests(unittest.TestCase):
    def api(self):
        try:
            spec = importlib.util.find_spec("tools.realtime_optimization.metrics")
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, "Benchmark metrics implementation is missing")
        from tools.realtime_optimization import metrics
        return metrics

    def test_latency_excludes_warmup_and_counts_strict_deadlines(self):
        m = self.api()
        rows = [{"wall_ms": x, "cpu_ms": 1, "warmup": i == 0,
                 "valid_samples": 12000} for i, x in enumerate([900, 100, 250, 300])]
        result = m.summarize(rows, 48000, 12000)
        self.assertEqual(result["chunks"], 3)
        self.assertEqual(result["deadline_misses"], 1)
        self.assertEqual(result["latency_ms"]["p50"], 250)
        self.assertAlmostEqual(result["realtime_factor"], 650 / 750)

    def test_no_observations_are_not_zero_latency(self):
        self.assertIsNone(self.api().summarize([], 48000, 12000)["latency_ms"]["p99"])

    def test_quality_counts_nonfinite_and_clipping_without_sanitizing(self):
        x = np.array([0, np.nan, np.inf, -1.2, 1.0, 0], np.float32)
        q = self.api().audio_quality(x, 48000, 3)
        self.assertEqual(q["nonfinite_samples"], 2)
        self.assertEqual(q["clipped_samples"], 2)

    def test_silence_warmup_is_expected_and_boundary_jump_is_reported(self):
        m = self.api()
        x = np.array([0, 0, 0, 0, 0.7, 0.7, 0.7, 0.7])
        q = m.audio_quality(x, 8, 4, source=np.ones(8), expected_silent=[True, False])
        self.assertEqual(q["unexpected_silent_chunks"], [])
        self.assertAlmostEqual(q["boundary_jump_max"], 0.7)
        q = m.audio_quality(np.zeros(8), 8, 4, source=np.ones(8), expected_silent=[True, False])
        self.assertEqual(q["unexpected_silent_chunks"], [1])

    def test_contour_shift_and_unvoiced_are_not_tone_errors(self):
        m = self.api()
        q = m.compare_contours(np.array([100, 120, 0, 110, 90]),
                               np.array([200, 240, 0, 220, 180]), pitch_shift=12)
        self.assertEqual(q["voicing_agreement"], 1)
        self.assertAlmostEqual(q["median_abs_cents_error"], 0)
        self.assertEqual(q["slope_direction_agreement"], 1)


class BenchmarkSafetyTests(unittest.TestCase):
    def api(self):
        try:
            spec = importlib.util.find_spec("tools.realtime_optimization.benchmark")
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, "Benchmark CLI implementation is missing")
        from tools.realtime_optimization import benchmark
        return benchmark

    def test_config_rejects_recording_and_invalid_frame_sizes(self):
        b = self.api()
        good = {"voice_changer": {"block_frame": 12000, "cross_fade_overlap_size": .05,
                "extra_convert_size": 2.5, "model_path": "a.pth", "index_path": "a.index",
                "embedder_model": "custom", "embedder_model_custom": "hubert"}, "inference": {}}
        for key, value in [("record_audio", True), ("block_frame", 0),
                           ("cross_fade_overlap_size", float("nan"))]:
            bad = json.loads(json.dumps(good))
            bad["voice_changer"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_config(bad)

    def test_partial_chunk_is_padded_and_valid_length_preserved(self):
        b = self.api()
        chunks = b.make_chunks(np.arange(5, dtype=np.float32), 4)
        self.assertEqual([n for _, n in chunks], [4, 1])
        np.testing.assert_array_equal(chunks[1][0], [4, 0, 0, 0])

    def test_effective_index_path_matches_upstream_trained_replacement(self):
        b = self.api()
        good = json.loads(b.EXAMPLE.read_text(encoding="utf-8"))
        good["voice_changer"]["index_path"] = "logs/trained_voice/trained.index"
        normalized = b.validate_config(good)
        self.assertEqual(normalized["voice_changer"]["index_path"], "logs/added_voice/added.index")

    def test_quality_loader_preserves_nonfinite_but_inference_loader_refuses(self):
        from unittest.mock import patch
        import soundfile as sf
        b = self.api()
        data = np.array([[0], [np.nan], [np.inf]], dtype=np.float32)
        with patch.object(sf, "read", return_value=(data, 48000)):
            with self.assertRaises(ValueError):
                b.load_wav(b.EXAMPLE)
            result, _ = b.load_wav(b.EXAMPLE, allow_nonfinite=True)
        self.assertTrue(np.isnan(result[1]) and np.isinf(result[2]))

    def test_completed_pass_rtf_counts_tail_fence_without_changing_chunk_service(self):
        from unittest.mock import patch
        import types
        b = self.api()
        vc = types.SimpleNamespace(vc_model=types.SimpleNamespace(warmup_blocks=0),
                                   process_audio=lambda x, **params: (x, 0))
        ticks = iter([0, 0, 100_000_000, 300_000_000])
        with patch("time.perf_counter_ns", side_effect=lambda: next(ticks)):
            rows, audio, stages, rss, completed = b._pass(vc,
                [(np.zeros(12000, dtype=np.float32), 12000)], {}, 0, completion_fence=lambda: None)
        self.assertEqual(rows[0]["wall_ms"], 100)
        self.assertEqual(completed["wall_ms"], 300)
        self.assertAlmostEqual(completed["realtime_factor_including_tail"], 1.2)

    def test_inference_refused_before_importing_rvc(self):
        self.api()
        script = ROOT / "tools/realtime_optimization/benchmark.py"
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(script), "--mode", "run"],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--allow-inference", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_running_checkpoint_precedes_model_load_and_records_cli_options(self):
        from unittest.mock import patch
        b = self.api()
        snapshots = []
        def writer(folder, result):
            snapshots.append(json.loads(json.dumps(result)))
        def load(*args):
            self.assertEqual(snapshots[0]["status"], "RUNNING")
            self.assertEqual(snapshots[0]["run_options"]["warmup_chunks"], 2)
            raise ValueError("隔離的載入失敗")
        with patch.object(b, "audit", return_value={"prerequisites": [], "config_sha256": {}}), \
             patch.object(b, "new_run_dir", return_value=Path("unused")), \
             patch.object(b, "write_report", side_effect=writer), \
             patch.object(b, "run_model", side_effect=load):
            code = b.main(["--mode", "run", "--allow-inference", "--input", "fixture.wav", "--warmup-chunks", "2"])
        self.assertEqual(code, 1)
        self.assertEqual(snapshots[-1]["status"], "FAIL")
        self.assertIn("隔離的載入失敗", snapshots[-1]["error"])

    def test_new_run_directory_never_reuses_an_existing_output(self):
        b = self.api()
        a, c = b.new_run_dir(), b.new_run_dir()
        try:
            self.assertNotEqual(a, c)
            self.assertTrue(a.is_dir() and c.is_dir())
        finally:
            # Only remove the two new empty directories; no recursive cleanup.
            for folder in (a, c):
                self.assertEqual(folder.resolve().parent, b.RUNS.resolve())
                folder.rmdir()


class InstrumentationTests(unittest.TestCase):
    def api(self):
        try:
            spec = importlib.util.find_spec("tools.realtime_optimization.instrumentation")
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, "Stage instrumentation is missing")
        from tools.realtime_optimization.instrumentation import StageProfiler
        return StageProfiler

    def test_nested_scopes_restore_methods_and_preserve_results(self):
        import types
        from unittest.mock import patch
        obj = types.SimpleNamespace(inner=lambda x: x * 2)
        obj.outer = lambda x: obj.inner(x) + 1
        original = obj.inner
        profiler = self.api()("cpu")
        ticks = iter([0, 1_000_000, 3_000_000, 5_000_000])
        with profiler, patch("time.perf_counter_ns", side_effect=lambda: next(ticks)):
            profiler.wrap(obj, "inner", "inner")
            profiler.wrap(obj, "outer", "outer")
            self.assertEqual(obj.outer(4), 9)
        self.assertIs(obj.inner, original)
        rows = profiler.finish_chunk(0)
        self.assertEqual(rows[0]["host_inclusive_ms"], 2)
        self.assertEqual(rows[1]["host_exclusive_ms"], 3)

    def test_exception_restores_class_descriptor(self):
        class Example:
            @staticmethod
            def call():
                raise ValueError("fixture")
        descriptor = Example.__dict__["call"]
        profiler = self.api()("cpu")
        with self.assertRaises(ValueError), profiler:
            profiler.wrap(Example, "call", "fixture")
            Example.call()
        self.assertIs(Example.__dict__["call"], descriptor)

    def test_real_cpu_crossfade_output_unchanged_when_profiled(self):
        import torch
        from rvc.realtime import core
        n = 64
        a, b = torch.linspace(-.2, .2, n), torch.linspace(.3, -.3, n)
        fade_in = torch.linspace(0, 1, n)
        expected = core.phase_vocoder(a, b, 1 - fade_in, fade_in)
        with self.api()("cpu") as profiler:
            profiler.wrap(core, "phase_vocoder", "phase_vocoder")
            actual = core.phase_vocoder(a, b, 1 - fade_in, fade_in)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(profiler.finish_chunk(0)), 1)

    def test_cpu_stream_pipeline_instrumentation_preserves_audio_and_state(self):
        import types
        import torch
        from unittest.mock import patch
        from rvc.realtime import core, pipeline
        from tools.realtime_optimization.instrumentation import instrument_voice_changer

        # Only trained networks are replaced; real resampling, F0 cache, RMS,
        # streaming warmup, SOLA and phase vocoder execute on CPU.
        class Encoder(torch.nn.Module):
            def forward(self, audio):
                feats = torch.nn.functional.avg_pool1d(audio[:, None], 320, 320)
                return {"last_hidden_state": feats.transpose(1, 2).expand(-1, -1, 8)}

        class Generator:
            use_f0, version, tgt_sr, dtype = True, "v2", 48000, torch.float32
            config = types.SimpleNamespace(device="cpu")
            def inference(self, feats, p_len, sid, pitch, pitchf, rate):
                assert pitch.shape == pitchf.shape and feats.shape[1] == int(p_len[0])
                length = round(int(p_len[0]) * float(rate[0])) * 480
                return .15 * torch.sin(torch.arange(length) * .05)

        def factory(*args):
            f0_model = types.SimpleNamespace(get_f0=lambda x, filter_radius: torch.full((len(x) // 160 + 1,), 180.0))
            with patch.object(pipeline.Realtime_Pipeline, "setup_f0", return_value=f0_model):
                return pipeline.Realtime_Pipeline(Generator(), Encoder())

        def make_vc():
            with patch.object(core, "create_pipeline", side_effect=factory):
                return core.VoiceChanger(block_frame=1200, cross_fade_overlap_size=.01,
                    extra_convert_size=.04, silent_threshold=-90)

        signal = (.1 * np.sin(np.arange(1200) * .03)).astype(np.float32)
        baseline, profiled = make_vc(), make_vc()
        with torch.no_grad():
            expected = [baseline.process_audio(signal, volume_envelope=.63)[0] for _ in range(12)]
            with instrument_voice_changer(profiled) as profiler:
                actual = [profiled.process_audio(signal, volume_envelope=.63)[0] for _ in range(12)]
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(baseline.vc_model.warmup_blocks, profiled.vc_model.warmup_blocks)
        torch.testing.assert_close(baseline.sola_buffer, profiled.sola_buffer, rtol=0, atol=0)
        stages = {r["stage"] for r in profiler.finish_chunk(0)}
        self.assertTrue({"f0", "content_encoder", "generator", "sola_correlation", "phase_vocoder",
                         "input_resample", "output_resample", "volume_envelope"} <= stages)


class QualityInterfaceTests(unittest.TestCase):
    def api(self):
        try:
            spec = importlib.util.find_spec("tools.realtime_optimization.quality")
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, "Audio quality comparison implementation is missing")
        from tools.realtime_optimization import quality
        return quality

    def test_transcript_character_errors_count_substitution_and_deletion(self):
        result = self.api().compare_transcripts("今天，天氣很好！", "今天氣很壞")
        self.assertEqual(result["reference_characters"], 6)
        self.assertEqual(result["edit_distance"], 2)
        self.assertAlmostEqual(result["cer"], 2 / 6)

    def test_declared_delay_aligns_source_without_changing_target(self):
        source = np.array([.2, .4, .6])
        result = self.api().delayed_source(source, 2, 5)
        np.testing.assert_array_equal(result, [0, 0, .2, .4, .6])

    def test_contour_mismatched_time_grid_rejected(self):
        q = self.api()
        with self.assertRaises(ValueError):
            q.compare_timed_contours(np.array([[0, 100], [.01, 110]]),
                                     np.array([[0, 200], [.02, 220]]), 12)


class ExistingF0BehaviorTests(unittest.TestCase):
    def test_existing_realtime_and_offline_coarse_f0_differ_at_range_endpoints(self):
        """Pin the discovered baseline difference; changing it requires separate audio validation."""
        import types
        import torch
        from unittest.mock import patch
        from rvc.realtime.pipeline import Realtime_Pipeline
        from rvc.infer import pipeline as offline_module
        realtime = Realtime_Pipeline.__new__(Realtime_Pipeline)
        realtime.f0_method, realtime.f0_min, realtime.f0_max = "rmvpe", 50., 1100.
        realtime.f0_model = types.SimpleNamespace(get_f0=lambda x, filter_radius: torch.tensor([50., 1100.]))
        config = types.SimpleNamespace(device="cpu", x_pad=1, x_query=6, x_center=38, x_max=41)
        offline = offline_module.Pipeline(40000, config)
        class Predictor:
            def __init__(self, **kwargs):
                pass
            def get_f0(self, x, filter_radius):
                return np.array([50., 1100.])
        with patch.object(offline_module, "RMVPE", Predictor):
            coarse_offline, fine_offline = offline.get_f0(np.zeros(320), 2, "rmvpe", 0)
        coarse_rt, fine_rt = realtime.get_f0(torch.zeros(320))
        np.testing.assert_array_equal(coarse_offline, [1, 255])
        np.testing.assert_array_equal(coarse_rt.numpy()[0], [8, 246])
        np.testing.assert_array_equal(fine_rt.numpy()[0], fine_offline)


class IndependentF0Tests(unittest.TestCase):
    def test_start_follows_observed_warmup_and_rejects_early_crop(self):
        from tools.realtime_optimization.f0_evaluation import evaluation_start
        rows = [{"repeat": "0", "warmup": "True"} for _ in range(30)]
        rows += [{"repeat": "1", "warmup": "True"} for _ in range(30)]
        self.assertEqual(evaluation_start(rows, 12000), 7.75)
        self.assertEqual(evaluation_start(rows, 12000, 8), 8)
        with self.assertRaises(ValueError):
            evaluation_start(rows, 12000, 3.5)

    def test_fixed_pitch_shift_defaults_and_rejects_dynamic_modes(self):
        from tools.realtime_optimization.f0_evaluation import fixed_pitch_shift
        self.assertEqual(fixed_pitch_shift({}), 0)
        self.assertEqual(fixed_pitch_shift({"f0_up_key": 9}), 9)
        for key in ("f0_autotune", "proposed_pitch"):
            with self.assertRaises(ValueError):
                fixed_pitch_shift({key: True, "f0_up_key": 9})

    def test_existing_predictor_api_is_used_without_loading_weights(self):
        from tools.realtime_optimization.f0_evaluation import load_predictor
        from unittest.mock import patch
        with patch("rvc.lib.predictors.RMVPE.RMVPE0Predictor") as constructor:
            result = load_predictor(Path("fixture.pt"), "cpu")
        constructor.assert_called_once_with("fixture.pt", device="cpu")
        self.assertIs(result, constructor.return_value)

    def test_alignment_uses_common_frames_and_preserves_unvoiced(self):
        from tools.realtime_optimization.f0_evaluation import align_contours
        signals = {"source": np.array([100., 0, 120, 130, 140]),
                   "offline": np.array([100., 0, 120, 130]),
                   "realtime": np.array([0., 100, 0, 120, 130])}
        result = align_contours(signals, start_seconds=.01, delays_ms={"offline": 0, "realtime": 10})
        np.testing.assert_array_equal(result["source"], [[.01, 0], [.02, 120], [.03, 130]])
        np.testing.assert_array_equal(result["source"], result["realtime"])
        self.assertEqual(len(result["offline"]), 3)

    def test_fractional_frame_delay_and_empty_overlap_are_rejected(self):
        from tools.realtime_optimization.f0_evaluation import align_contours
        for delay, start in [(5, 0), (-10, 0), (0, 2)]:
            with self.assertRaises(ValueError):
                align_contours({"source": np.ones(5), "realtime": np.ones(5)},
                               start_seconds=start, delays_ms={"realtime": delay})

    def test_inference_gate_precedes_torch_import(self):
        from tools.realtime_optimization.f0_evaluation import main
        from unittest.mock import patch
        import builtins
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("rvc.lib.predictors"):
                raise AssertionError("不應載入模型")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaises(SystemExit) as caught:
                main(["--benchmark-report", "missing.json"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
