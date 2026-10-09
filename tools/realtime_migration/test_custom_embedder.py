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
from rvc.lib import utils
from rvc.realtime.worker import _apply_config


class CustomLoaderTests(unittest.TestCase):
    def test_missing_custom_embedder_never_falls_back_in_realtime(self):
        with self.assertRaises(FileNotFoundError):
            utils.load_embedding("custom", str(ROOT / "nonexistent-embedder"), allow_fallback=False)


class CacheTests(unittest.TestCase):
    def test_different_custom_path_reloads_and_same_path_reuses(self):
        old_model = types.SimpleNamespace()
        loaded_paths = []
        new_model = types.SimpleNamespace(to=lambda *a: new_model, float=lambda: new_model, eval=lambda: None)
        state = types.SimpleNamespace(embedder_model="custom", embedder_model_custom="folder-A",
                    device="cpu", pipeline=types.SimpleNamespace(hubert_model=old_model))
        vc = types.SimpleNamespace(vc_model=state)
        def load(kind, path, **kwargs):
            loaded_paths.append((kind, path))
            return new_model
        with patch.object(utils, "load_embedding", side_effect=load):
            _apply_config(vc, {"embedder_model": "custom", "embedder_model_custom": "folder-B"})
            _apply_config(vc, {"embedder_model": "custom", "embedder_model_custom": "folder-B"})
        self.assertEqual(loaded_paths, [("custom", "folder-B")])
        self.assertIs(state.pipeline.hubert_model, new_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
