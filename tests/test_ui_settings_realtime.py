"""Realtime persistence, native-setting preservation and client reset guards."""
import asyncio
import gradio as gr
from unittest.mock import patch
from tests.ui_settings_helpers import UIPageTestCase


class UIRealtimeSettingsTests(UIPageTestCase):
    def test_realtime_roundtrip_and_reset_preserve_native_selection(self):
        module = self.load_module("tabs.realtime.realtime")
        native = module.load_realtime_settings()
        native.update(input_device="Native Input (Windows WASAPI)",
                      output_device="Native Output (Windows WASAPI)",
                      monitor_device="Native Monitor (Windows WASAPI)",
                      model_file=self.model, index_file=self.index)
        self.stack.enter_context(patch.object(module, "load_realtime_settings", return_value=native))
        self.stack.enter_context(patch.object(module, "names", [self.model]))
        self.stack.enter_context(patch.object(module, "get_audio_devices_formatted", return_value=(
            {native["input_device"]: 1}, {native["output_device"]: 2, native["monitor_device"]: 3})))
        self.store.update_page("realtime", {"pitch": 9, "volume_envelope": 0.63,
            "index_rate": 0.75, "chunk_size": 250, "cross_fade_overlap_size": 0.05,
            "extra_convert_size": 2.5, "vad_enabled": True, "embedder_model": "custom"})
        async def scenario():
            app, page = await self.build("tabs.realtime.realtime", "realtime_tab")
            self.assertEqual(page.fields["pitch"].value, 9)
            self.assertEqual(page.fields["volume_envelope"].value, 0.63)
            self.assertEqual(page.fields["chunk_size"].value, 250)
            self.assertTrue(page.components["embedder_custom"].visible)
            protected = {"input_audio_device", "output_audio_device", "monitor_output_device", "model_file", "index_file"}
            self.assertTrue(protected.isdisjoint(page.fields))
            reset = next(fn for fn in app.fns.values() if fn.fn == page.reset)
            native_ids = {page.components[key]._id for key in protected}
            self.assertTrue(native_ids.isdisjoint(component._id for component in reset.outputs))
            # Exercise the same Gradio API processing used by the reset button.
            result = await app.process_api(reset, inputs=[self.model])
            by_name = dict(zip(page.output_names, result["data"]))
            self.assertEqual(by_name["pitch"]["value"], 0)
            self.assertEqual(page.components["model_file"].value, self.model)
            self.assertEqual(page.components["input_audio_device"].value, native["input_device"])
            self.assertEqual(self.store.read_page("realtime")["pitch"], 0)
        asyncio.run(scenario())


    def test_client_realtime_cannot_reset_until_client_has_stopped(self):
        from gradio.state_holder import SessionState
        module = self.load_module("tabs.realtime.realtime")
        self.stack.enter_context(patch.object(module, "client_mode", True))
        self.stack.enter_context(patch.object(module, "names", [self.model]))
        self.store.update_page("realtime-client", {"pitch": 9})
        async def scenario():
            app, page = await self.build("tabs.realtime.realtime", "realtime_tab")
            self.assertIn("client_active", page.components, "Client needs session-local running state")
            active = page.components["client_active"]
            state = SessionState(app)
            start = next(fn for fn in app.fns.values() if active in fn.outputs
                         and (page.components["start_button"]._id, "click") in fn.targets)
            await app.process_api(start, inputs=[], state=state)
            self.assertTrue(state[active._id])
            reset = next(fn for fn in app.fns.values() if fn.fn == page.reset)
            with self.assertRaises(gr.Error):
                await app.process_api(reset, inputs=[self.model, None], state=state)
            self.assertEqual(self.store.read_page("realtime-client")["pitch"], 9)
            stopped = next(fn for fn in app.fns.values() if active in fn.outputs
                           and (page.components["json_button_hidden"]._id, "change") in fn.targets)
            await app.process_api(stopped, inputs=[{"start_button": True, "stop_button": False}], state=state)
            self.assertFalse(state[active._id])
            await app.process_api(reset, inputs=[self.model, None], state=state)
            self.assertEqual(self.store.read_page("realtime-client")["pitch"], 0)
        asyncio.run(scenario())
