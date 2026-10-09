"""Persistence integration for inference, TTS, training and utility pages."""
import asyncio
from functools import partial
from unittest.mock import patch
from tests.ui_settings_helpers import UIPageTestCase


class UIPageIntegrationTests(UIPageTestCase):
    def test_inference_single_and_batch_are_persisted_separately(self):
        self.store.update_page("inference", {"pitch": 9, "pitch_batch": -2,
            "rms_mix_rate": 0.63, "embedder_model": "custom", "autotune": True})
        async def scenario():
            app, page = await self.build("tabs.inference.inference", "inference_tab")
            self.assertEqual(page.fields["pitch"].value, 9)
            self.assertEqual(page.fields["pitch_batch"].value, -2)
            self.assertTrue(page.components["autotune_strength"].visible)
            self.assertTrue(page.components["embedder_custom"].visible)
            self.assertNotIn("preset_dropdown", page.fields)
            self.assertNotIn("formant_preset", page.fields)
            save = next(fn for fn in app.fns.values() if isinstance(fn.fn, partial)
                        and fn.fn.func == page.save and fn.fn.args == ("pitch",))
            result = await app.process_api(save, inputs=[12, self.model])
            self.assertIn("已儲存", result["data"][0])
            self.assertEqual(self.store.read_page("inference")["pitch"], 12)
            self.assertEqual(self.store.read_page("inference")["pitch_batch"], -2)
        asyncio.run(scenario())

    def test_tts_training_and_tool_pages_are_connected(self):
        training = self.load_module("tabs.train.train")
        for name, value in (("get_gpu_info", "UI fixture"), ("get_number_of_gpus", "0"),
                            ("auto_enable_checkpointing", False)):
            self.stack.enter_context(patch.object(training, name, return_value=value))
        self.store.update_page("training", {"total_epoch": 600, "vocoder": "RefineGAN", "sampling_rate": "24000"})
        self.store.update_page("tts", {"tts_rate": 12, "pitch": 9})
        async def scenario():
            _, page = await self.build("tabs.train.train", "train_tab")
            self.assertEqual(page.fields["total_epoch"].value, 600)
            self.assertEqual(page.fields["sampling_rate"].value, "24000")
            self.assertIn(("24000", "24000"), page.fields["sampling_rate"].choices)
            self.assertNotIn("pth_dropdown_export", page.fields)
            _, page = await self.build("tabs.tts.tts", "tts_tab")
            self.assertEqual(page.fields["tts_rate"].value, 12)
            for name, function, field in [
                ("tabs.voice_blender.voice_blender", "voice_blender_tab", "alpha_a"),
                ("tabs.extra.sections.f0_extractor", "f0_extractor_tab", "f0_method"),
                ("tabs.extra.sections.processing", "processing_tab", "model_view_model_path")]:
                _, page = await self.build(name, function)
                self.assertIn(field, page.fields)
        asyncio.run(scenario())
