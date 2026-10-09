"""UI preference regressions using real Gradio components and temporary files.

No audio streams, model loading or GPU inference are performed here.
"""
import importlib
import asyncio
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr

try:
    settings = importlib.import_module("tabs.ui_settings")
except ModuleNotFoundError:
    settings = None


class UISettingsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(settings, "UI persistence implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ui_settings.json"
        self.store = settings.UISettingsStore(self.path)

    def page(self, name="realtime", **kwargs):
        async def build():
            return self.build_page(name, **kwargs)
        return asyncio.run(build())

    def build_page(self, name, **kwargs):
        with gr.Blocks(analytics_enabled=False) as app:
            components = {
                "pitch": gr.Slider(-24, 24, value=0),
                "chunk_size": gr.Slider(16, 500, value=250),
                "vad_enabled": gr.Checkbox(value=True),
                "embedder_model": gr.Radio(["contentvec", "custom"], value="contentvec"),
                "embedder_custom": gr.Group(visible=False),
                "model_file": gr.Dropdown(["native-model.pth"], value="native-model.pth"),
                "terms_checkbox": gr.Checkbox(value=False),
                "output_info": gr.Textbox(value="", interactive=False),
            }
            binding = settings.remember_page(name, components, store=self.store,
                                             exclude={"model_file"}, **kwargs)
        return app, components, binding

    def test_new_page_restores_numbers_booleans_and_choices(self):
        _, _, page = self.page()
        page.save("pitch", 9)
        page.save("vad_enabled", False)
        page.save("embedder_model", "custom")
        self.store = settings.UISettingsStore(self.path)
        app, restored, _ = self.page()
        self.assertEqual(restored["pitch"].value, 9)
        self.assertFalse(restored["vad_enabled"].value)
        self.assertEqual(restored["embedder_model"].value, "custom")
        self.assertTrue(restored["embedder_custom"].visible)
        props = next(item["props"] for item in app.config["components"]
                     if item["id"] == restored["pitch"]._id)
        self.assertEqual(props["value"], 9)

    def test_reset_is_limited_to_this_pages_registered_fields(self):
        native = Path(self.temp.name) / "config.json"
        original = '{"realtime":{"model_file":"native-model.pth"},"theme":"Original"}'
        native.write_text(original, encoding="utf-8")
        _, components, page = self.page()
        page.save("pitch", 9)
        self.store.set_value("inference", "pitch", 12)
        self.store.set_value("realtime", "future_field", 77)
        updates = page.reset()
        values = dict(zip(page.fields, updates[:len(page.fields)]))
        self.assertEqual(values["pitch"]["value"], 0)
        self.assertEqual(values["chunk_size"]["value"], 250)
        self.assertNotIn("model_file", page.fields)
        self.assertNotIn("terms_checkbox", page.fields)
        self.assertNotIn("output_info", page.fields)
        self.assertEqual(components["model_file"].value, "native-model.pth")
        self.assertEqual(self.store.read_page("inference")["pitch"], 12)
        self.assertEqual(self.store.read_page("realtime")["future_field"], 77)
        self.assertEqual(native.read_text(encoding="utf-8"), original)
        _, reopened, _ = self.page()
        self.assertEqual(reopened["pitch"].value, 0)

    def test_bad_saved_values_use_defaults_without_overwriting_the_file(self):
        for key, value in {"pitch": 99, "vad_enabled": "false",
                           "embedder_model": "removed-model", "chunk_size": float("inf")}.items():
            # Deliberately malformed external file; the normal writer rejects infinity.
            payload = {"version": 1, "pages": {"realtime": {key: value}}}
            self.path.write_text(json.dumps(payload), encoding="utf-8")
            before = self.path.read_bytes()
            _, components, _ = self.page()
            expected = {"pitch": 0, "vad_enabled": True,
                        "embedder_model": "contentvec", "chunk_size": 250}[key]
            self.assertEqual(components[key].value, expected)
            self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_file_does_not_get_destroyed_by_autosave_or_reset(self):
        self.path.write_bytes(b'{"broken":')
        _, components, page = self.page()
        self.assertEqual(components["pitch"].value, 0)
        self.assertIn("失敗", page.save("pitch", 9))
        with self.assertRaises(gr.Error):
            page.reset()
        self.assertEqual(self.path.read_bytes(), b'{"broken":')

    def test_parallel_field_writes_merge_instead_of_losing_other_values(self):
        def save(index):
            settings.UISettingsStore(self.path).set_value("inference", f"field_{index}", index)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(save, range(20)))
        self.assertEqual(self.store.read_page("inference"), {f"field_{i}": i for i in range(20)})

    def test_active_realtime_reset_is_rejected_before_any_write(self):
        _, _, page = self.page(reset_guard=lambda: False)
        page.save("pitch", 9)
        before = self.path.read_bytes()
        with self.assertRaises(gr.Error):
            page.reset()
        self.assertEqual(self.path.read_bytes(), before)

    def test_load_does_not_create_a_settings_file(self):
        self.page()
        self.assertFalse(self.path.exists())

    def test_failed_atomic_replace_preserves_last_saved_state(self):
        from unittest.mock import patch
        self.store.set_value("realtime", "pitch", 9)
        before = self.path.read_bytes()
        with patch.object(settings.os, "replace", side_effect=PermissionError("file locked")):
            with self.assertRaises(settings.UISettingsError):
                self.store.set_value("realtime", "pitch", 12)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_dependent_dropdown_restores_choices_for_saved_model(self):
        self.store.update_page("inference", {"model_file": "B", "sid": 3})
        async def build():
            with gr.Blocks(analytics_enabled=False):
                components = {"model_file": gr.Dropdown(["A", "B"], value="A"),
                              "sid": gr.Dropdown([0], value=0)}
                binding = settings.remember_page("inference", components, store=self.store,
                    context_fields=("model_file",),
                    choice_providers={"sid": lambda values: [0, 3] if values["model_file"] == "B" else [0]})
            return components, binding
        components, page = asyncio.run(build())
        self.assertEqual(components["sid"].value, 3)
        self.assertEqual(components["sid"].choices, [("0", 0), ("3", 3)])
        updates = page.reset("A")
        self.assertEqual(updates[1]["choices"], [("0", 0)])
        self.assertEqual(updates[1]["value"], 0)

    def test_protected_context_can_change_without_being_reset(self):
        async def build():
            with gr.Blocks(analytics_enabled=False):
                components = {"model_file": gr.Dropdown(["A", "B"], value="A"),
                              "sid": gr.Dropdown([0], value=0)}
                binding = settings.remember_page("realtime", components, store=self.store,
                    exclude={"model_file"}, context_fields=("model_file",),
                    choice_providers={"sid": lambda values: [0, 3] if values["model_file"] == "B" else [0]})
            return binding
        page = asyncio.run(build())
        page.save("sid", 3, "B")
        self.assertEqual(self.store.read_page("realtime"), {"sid": 3})
        updates = page.reset("B")
        self.assertEqual(updates[0]["choices"], [("0", 0), ("3", 3)])
        self.assertEqual(updates[0]["value"], 0)
        self.assertNotIn("model_file", self.store.read_page("realtime"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
