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
from tabs.realtime import realtime as ui
from tools.realtime_migration.test_helpers import device


class LifecycleTests(unittest.TestCase):
    def test_stop_during_stream_open_leaves_consistent_stopped_state(self):
        started_stop = threading.Event()
        stopped = threading.Event()
        thread_holder = []
        def stop():
            started_stop.set()
            ui.stop_realtime()
            stopped.set()
        def open_streams(**kwargs):
            thread = threading.Thread(target=stop, daemon=True)
            thread_holder.append(thread)
            thread.start()
            self.assertTrue(started_stop.wait(1))
            # The UI control lock serializes Stop after stream opening.
            self.assertFalse(stopped.wait(0.05))
        audio = types.SimpleNamespace(start=open_streams, stop=lambda: None)
        worker = types.SimpleNamespace(vc_model=types.SimpleNamespace(warmup_blocks=0),
                    _process=types.SimpleNamespace(is_alive=lambda: True), stop=lambda: None)
        kwargs = {name: p.default if p.default is not inspect.Parameter.empty else 0
                  for name, p in inspect.signature(ui.start_realtime).parameters.items()}
        kwargs.update(input_audio_device="AM350 (Windows WASAPI)", output_audio_device="VAIO (Windows WASAPI)",
                      input_audio_gain=100, output_audio_gain=100, monitor_audio_gain=100,
                      pth_path="test-model.pth", embedder_model="contentvec", chunk_size=250)
        with patch.object(ui, "running", False), patch.object(ui, "callbacks", None), \
             patch.object(ui, "audio_manager", None), patch.object(ui, "reinitialize_audio_backend"), \
             patch.object(ui, "resolve_audio_device", return_value=device(1, "device", "Windows WASAPI", inputs=2, outputs=2)), \
             patch.object(ui, "resolve_sample_rate", return_value=48000), \
             patch.object(ui, "AudioCallbacks", return_value=types.SimpleNamespace(audio=audio, vc=worker)):
            generator = ui.start_realtime(**kwargs)
            next(generator)
            next(generator)
            self.assertTrue(stopped.wait(1))
            list(generator)
            self.assertFalse(ui.running)
            self.assertIsNone(ui.callbacks)
            self.assertIsNone(ui.audio_manager)
        for thread in thread_holder:
            thread.join(timeout=1)

    def test_stop_cancels_a_start_paused_at_first_ui_yield(self):
        events = []
        audio = types.SimpleNamespace(start=lambda **kw: events.append("opened"), stop=lambda: None)
        worker = types.SimpleNamespace(vc_model=types.SimpleNamespace(warmup_blocks=0),
                    _process=types.SimpleNamespace(is_alive=lambda: False), stop=lambda: None)
        kwargs = {name: p.default if p.default is not inspect.Parameter.empty else 0
                  for name, p in inspect.signature(ui.start_realtime).parameters.items()}
        kwargs.update(input_audio_device="AM350 (Windows WASAPI)", output_audio_device="VAIO (Windows WASAPI)",
                      input_audio_gain=100, output_audio_gain=100, monitor_audio_gain=100,
                      pth_path="test-model.pth", embedder_model="contentvec", chunk_size=250)
        with patch.object(ui, "running", False), patch.object(ui, "callbacks", None), \
             patch.object(ui, "audio_manager", None), patch.object(ui, "reinitialize_audio_backend"), \
             patch.object(ui, "resolve_audio_device", return_value=device(1, "device", "Windows WASAPI", inputs=2, outputs=2)), \
             patch.object(ui, "resolve_sample_rate", return_value=48000), \
             patch.object(ui, "AudioCallbacks", return_value=types.SimpleNamespace(audio=audio, vc=worker)), \
             patch.object(ui.time, "sleep", return_value=None):
            generator = ui.start_realtime(**kwargs)
            next(generator)
            ui.stop_realtime()
            list(generator)
            self.assertFalse(ui.running)
        self.assertEqual(events, [], "a canceled start must never open audio streams")

    def test_stop_releases_failed_start_resources_even_when_running_is_false(self):
        events = []
        audio = types.SimpleNamespace(stop=lambda: events.append("audio-stop"))
        worker = types.SimpleNamespace(stop=lambda: events.append("worker-stop"))
        with patch.object(ui, "running", False), patch.object(ui, "audio_manager", audio), \
             patch.object(ui, "callbacks", types.SimpleNamespace(vc=worker)):
            ui.stop_realtime()
            self.assertEqual(events, ["audio-stop", "worker-stop"])
            self.assertIsNone(ui.audio_manager)
            self.assertIsNone(ui.callbacks)

    def test_stream_close_error_does_not_skip_worker_stop(self):
        events = []
        def fail():
            events.append("audio-stop")
            raise RuntimeError("driver refused close")
        audio = types.SimpleNamespace(stop=fail)
        worker = types.SimpleNamespace(stop=lambda: events.append("worker-stop"))
        with patch.object(ui, "running", True), patch.object(ui, "audio_manager", audio), \
             patch.object(ui, "callbacks", types.SimpleNamespace(vc=worker)):
            response = ui.stop_realtime()
            self.assertEqual(events, ["audio-stop", "worker-stop"])
            self.assertIn("driver refused close", response[0])
            self.assertIs(ui.audio_manager, audio, "retain unclosed streams for another explicit Stop")
            self.assertFalse(ui.running)

    def test_dead_model_worker_yields_error_without_busy_waiting_on_zero_warmup(self):
        events = []
        audio = types.SimpleNamespace(start=lambda **kw: None, stop=lambda: events.append("audio-stop"))
        worker = types.SimpleNamespace(vc_model=types.SimpleNamespace(warmup_blocks=0),
                    _process=types.SimpleNamespace(is_alive=lambda: False), stop=lambda: events.append("worker-stop"))
        kwargs = {name: p.default if p.default is not inspect.Parameter.empty else 0
                  for name, p in inspect.signature(ui.start_realtime).parameters.items()}
        kwargs.update(input_audio_device="AM350 (Windows WASAPI)", output_audio_device="VAIO (Windows WASAPI)",
                      input_audio_gain=100, output_audio_gain=100, monitor_audio_gain=100,
                      pth_path="test-model.pth", embedder_model="contentvec", chunk_size=250)
        with patch.object(ui, "running", False), patch.object(ui, "callbacks", None), \
             patch.object(ui, "audio_manager", None), patch.object(ui, "reinitialize_audio_backend"), \
             patch.object(ui, "resolve_audio_device", return_value=device(1, "device", "Windows WASAPI", inputs=2, outputs=2)), \
             patch.object(ui, "resolve_sample_rate", return_value=48000), \
             patch.object(ui, "AudioCallbacks", return_value=types.SimpleNamespace(audio=audio, vc=worker)), \
             patch.object(ui.time, "sleep", return_value=None):
            responses = list(ui.start_realtime(**kwargs))
        self.assertIn("worker stopped", responses[-1][0])
        self.assertEqual(events, ["audio-stop", "worker-stop"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
