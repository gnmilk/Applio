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
from rvc.realtime import audio as backend
from tabs.realtime import realtime as ui
from tools.realtime_migration.test_helpers import device


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [device(41, "麥克風 (AM350)", "Windows WASAPI", inputs=2)]
        self.outputs = [device(13, "Voicemeeter Input", "MME", outputs=16),
                        device(73, "Voicemeeter Input", "Windows WASAPI", outputs=2)]

    def test_legacy_label_resolves_current_index_and_host_api(self):
        self.assertTrue(callable(getattr(backend, "resolve_audio_device", None)), "missing stable backend resolver")
        with patch.object(backend, "list_audio_device", return_value=(self.inputs, self.outputs)):
            found = backend.resolve_audio_device("33: Voicemeeter Input (Windows WASAPI)", "output")
            self.assertEqual(found.index, 73)
            self.assertEqual(found.host_api, "Windows WASAPI")

    def test_unavailable_or_ambiguous_endpoint_is_not_guessed(self):
        self.assertTrue(callable(getattr(backend, "resolve_audio_device", None)), "missing stable backend resolver")
        with patch.object(backend, "list_audio_device", return_value=(self.inputs, self.outputs * 2)):
            for label in ("Voicemeeter Input", "Voicemeeter Input (Windows WASAPI)", "99"):
                with self.subTest(label=label), self.assertRaises(RuntimeError):
                    backend.resolve_audio_device(label, "output")

    def test_ui_labels_and_saved_legacy_values_are_stable(self):
        with patch.object(ui, "list_audio_device", return_value=(self.inputs, self.outputs)):
            inputs, outputs = ui.get_audio_devices_formatted()
            self.assertIn("Voicemeeter Input (Windows WASAPI)", outputs)
        self.assertEqual(ui.get_safe_dropdown_value("33: Voicemeeter Input (Windows WASAPI)",
                         ["Voicemeeter Input (MME)", "Voicemeeter Input (Windows WASAPI)"]),
                         "Voicemeeter Input (Windows WASAPI)")

    def test_refresh_refuses_active_realtime(self):
        self.assertTrue(callable(getattr(ui, "reinitialize_audio_backend", None)), "missing guarded refresh")
        with patch.object(ui, "running", True), patch.object(ui, "_reinitialize_audio_backend", return_value=None):
            with self.assertRaises(RuntimeError):
                ui.reinitialize_audio_backend()

    def test_public_refresh_refuses_start_in_progress_under_lock(self):
        with patch.object(ui, "running", False), patch.object(ui, "_starting", True), \
             patch.object(ui, "_reinitialize_audio_backend", return_value=None):
            with self.assertRaises(RuntimeError):
                ui.reinitialize_audio_backend()

    def test_empty_output_queue_does_not_block_callback(self):
        audio = backend.Audio(None)
        out = np.full((16, 2), 99, dtype=np.float32)
        completed = threading.Event()
        def callback():
            audio.audio_queue(out, 1.0)
            completed.set()
        thread = threading.Thread(target=callback, daemon=True)
        thread.start()
        finished = completed.wait(0.25)
        if not finished:
            audio.mon_queue.put(np.zeros(16, dtype=np.float32))
            thread.join(timeout=1)
        self.assertTrue(finished, "output callback waits forever for input; stop/restart can hang")
        self.assertTrue(np.all(out == 0))

    def test_backend_resolves_after_refresh_validates_and_cleans_partial_open(self):
        audio = backend.Audio(None)
        events = []
        streams = []
        generation = [0]
        def query():
            if not generation[0]:
                return [device(36, "AM350", "Windows WASAPI", inputs=2)], [device(33, "VAIO", "Windows WASAPI", outputs=2)]
            return [device(41, "AM350", "Windows WASAPI", inputs=2)], [device(73, "VAIO", "Windows WASAPI", outputs=2)]
        def refresh():
            generation[0] += 1
            events.append("refresh")
        def check(direction, **kw):
            expected = 41 if direction == "input" else 73
            self.assertEqual(kw["device"], expected)
            self.assertEqual(kw["samplerate"], 48000)
            events.append("check-" + direction)
        class Stream:
            def __init__(inner, **kw):
                inner.closed = False
                streams.append(inner)
                self.assertIn(kw["device"], (41, 73))
                self.assertIn("check-input", events)
                self.assertIn("check-output", events)
                events.append("open")
            def start(inner):
                if len(streams) > 1:
                    raise RuntimeError("simulated driver refused start")
            def stop(inner):
                pass
            def close(inner):
                inner.closed = True
        with patch.object(backend, "list_audio_device", side_effect=query), \
             patch.object(backend.sd, "_terminate", return_value=None), \
             patch.object(backend.sd, "_initialize", side_effect=refresh), \
             patch.object(backend.sd, "check_input_settings", side_effect=lambda **k: check("input", **k)), \
             patch.object(backend.sd, "check_output_settings", side_effect=lambda **k: check("output", **k)), \
             patch.object(backend.sd, "InputStream", Stream), patch.object(backend.sd, "OutputStream", Stream):
            with self.assertRaises(RuntimeError):
                audio.start("36: AM350 (Windows WASAPI)", "33: VAIO (Windows WASAPI)", block_frame=12000)
        self.assertTrue(streams, "stream was never opened using freshly resolved indexes")
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertFalse(audio.running)
        self.assertIsNone(audio.input_stream)
        self.assertIsNone(audio.output_stream)

    def test_monitor_constructor_opens_before_input(self):
        audio = backend.Audio(None, monitor=True)
        events = []
        class Stream:
            def __init__(inner, direction, **kwargs):
                events.append((direction, kwargs["device"]))
            def start(inner):
                pass
            def stop(inner):
                pass
            def close(inner):
                pass
        with patch.object(backend.sd, "InputStream", side_effect=lambda **k: Stream("input", **k)), \
             patch.object(backend.sd, "OutputStream", side_effect=lambda **k: Stream("output", **k)):
            audio.run_audio_stream(12000, 1, 2, 3, 2, 2, 2, None, None, None, 48000)
            audio.stop()
        self.assertEqual(events, [("output", 2), ("output", 3), ("input", 1)])

    def test_mme_shortened_alias_must_identify_a_unique_source(self):
        primary = device(1, "Microphone (Very Long USB Device Name A)", "Windows WASAPI", inputs=2)
        sibling = device(2, "Microphone (Very Long USB Device Name B)", "Windows WASAPI", inputs=2)
        shortened = device(3, "Microphone (Very Long USB Device", "MME", inputs=2)
        found = backend._fallback_candidates(primary, "input", [primary, sibling, shortened])
        self.assertEqual(found, [primary], "truncated MME name cannot distinguish source A from B")

    def test_fallback_alias_is_revalidated_after_refresh(self):
        audio = backend.Audio(None)
        mic = device(1, "AM350", "Windows WASAPI", inputs=2)
        primary = device(2, "Speakers (Very Long USB Device Name A)", "Windows WASAPI", outputs=2)
        sibling = device(4, "Speakers (Very Long USB Device Name B)", "Windows WASAPI", outputs=2)
        shortened = device(3, "Speakers (Very Long USB Device", "MME", outputs=2)
        generation = [0]
        attempted = []
        def refresh():
            generation[0] += 1
        def devices():
            return [mic], [primary, shortened] + ([sibling] if generation[0] > 1 else [])
        def open_streams(*args):
            attempted.append(args[2])
            raise RuntimeError("primary unavailable")
        with patch.object(backend, "list_audio_device", side_effect=devices), \
             patch.object(backend.sd, "_terminate"), patch.object(backend.sd, "_initialize", side_effect=refresh), \
             patch.object(backend.sd, "check_input_settings"), patch.object(backend.sd, "check_output_settings"), \
             patch.object(audio, "run_audio_stream", side_effect=open_streams):
            with self.assertRaises(RuntimeError):
                audio.start(mic.label, primary.label, block_frame=12000)
        self.assertNotIn(3, attempted, "new ambiguous source must invalidate previously approved MME fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
