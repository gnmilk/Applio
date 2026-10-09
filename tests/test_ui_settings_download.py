"""Download preferences retain the default rate through dependent events."""
import asyncio
from unittest.mock import patch
from tests.ui_settings_helpers import UIPageTestCase


class UIDownloadSettingsTests(UIPageTestCase):
    def test_reset_rate_survives_model_change_cascade(self):
        download = self.load_module("tabs.download.download")
        self.stack.enter_context(patch.object(download, "fetch_pretrained_data", return_value={
            "Titan": {"32k": {}, "40k": {}}, "Other": {"48k": {}}}))
        self.store.update_page("download", {"pretrained_model": "Other", "pretrained_sample_rate": "48k"})
        async def scenario():
            app, page = await self.build("tabs.download.download", "download_tab")
            self.assertEqual(page.fields["pretrained_sample_rate"].value, "48k")
            reset = next(fn for fn in app.fns.values() if fn.fn == page.reset)
            result = await app.process_api(reset, inputs=["Other"])
            defaults = dict(zip(page.output_names, result["data"]))
            self.assertEqual(defaults["pretrained_sample_rate"]["value"], "40k")
            cascade = next(fn for fn in app.fns.values() if fn.fn == download.update_sample_rate_dropdown)
            inputs = [defaults[next(name for name, component in page.fields.items()
                                    if component is source)]["value"] for source in cascade.inputs]
            result = await app.process_api(cascade, inputs=inputs)
            self.assertEqual(result["data"][0]["value"], "40k", "A dependent event must not undo the reset")
        asyncio.run(scenario())
