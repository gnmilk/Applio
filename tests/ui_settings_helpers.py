"""Construct actual tab UIs; replace only hardware, model and network access."""
import importlib
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
import sounddevice as sd
import torch
from tabs import ui_settings

ROOT = Path(__file__).resolve().parents[1]


class UIPageTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ui_settings.UISettingsStore(Path(self.temp.name) / "ui_settings.json")
        self.native_before = (ROOT / "assets/config.json").read_bytes()
        preferences = ROOT / "ui_settings.json"
        self.preferences_before = preferences.read_bytes() if preferences.exists() else None
        self.model = str(Path(self.temp.name) / "native-model.pth")
        Path(self.model).write_bytes(b"UI fixture only, never load")
        self.index = str(Path(self.temp.name) / "native.index")
        Path(self.index).write_bytes(b"UI fixture only, never load")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module, name, result in [(torch.cuda, "is_available", False), (torch.cuda, "device_count", 0)]:
            self.stack.enter_context(patch.object(module, name, return_value=result))
        for name in ("InputStream", "OutputStream", "query_devices", "query_hostapis"):
            self.stack.enter_context(patch.object(sd, name, side_effect=AssertionError("Physical audio is forbidden")))


    def tearDown(self):
        self.assertEqual((ROOT / "assets/config.json").read_bytes(), self.native_before)
        preferences = ROOT / "ui_settings.json"
        current = preferences.read_bytes() if preferences.exists() else None
        self.assertEqual(current, self.preferences_before, "Test must preserve real user preferences")


    def load_module(self, name):
        # The existing Download module clears Gradio's cache at import. Suppress
        # that unrelated side effect; do not change the author's production code.
        with patch("shutil.rmtree"), patch("os.makedirs"):
            module = importlib.import_module(name)
        if hasattr(module, "get_speakers_id"):
            self.stack.enter_context(patch.object(module, "get_speakers_id", return_value=[0, 3]))
        if hasattr(module, "get_files"):
            self.stack.enter_context(patch.object(module, "get_files", side_effect=lambda kind="model":
                [self.model] if kind == "model" else [self.index]))
        if hasattr(module, "default_weight"):
            self.stack.enter_context(patch.object(module, "default_weight", self.model))
        return module


    async def build(self, name, function):
        module = self.load_module(name)
        bindings = []
        original = ui_settings.remember_page
        def register(*args, **kwargs):
            kwargs["store"] = self.store
            binding = original(*args, **kwargs)
            bindings.append(binding)
            return binding
        with patch.object(module, "remember_page", side_effect=register, create=True):
            with gr.Blocks(analytics_enabled=False) as app:
                getattr(module, function)()
        self.assertEqual(len(bindings), 1, "Actual tab is not connected to UI preference persistence")
        return app, bindings[0]
