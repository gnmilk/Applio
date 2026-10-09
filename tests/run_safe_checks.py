"""Run UI preferences and all existing regressions without audio/GPU inference.

Usage: env/python.exe -B -X utf8 tests/run_safe_checks.py
"""
import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def main():
    config_path = ROOT / "assets/config.json"
    config_before = hashlib.sha256(config_path.read_bytes()).hexdigest()
    tracked = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT).decode().splitlines()
    new_sources = ["tabs/ui_settings.py"] + [p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("*.py")]
    sources = sorted(set(tracked + new_sources))
    for rel in sources:
        compile((ROOT / rel).read_text(encoding="utf-8-sig"), rel, "exec")
    print(f"PASS: in-memory Python compilation ({len(sources)} files)", flush=True)

    import torch
    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.cuda, "device_count", return_value=0):
        # Import realtime/librosa before the legacy split-audio isolated stubs.
        from tools.realtime_migration import test_realtime_port
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        suite.addTests(unittest.defaultTestLoader.loadTestsFromNames([
            "tools.realtime_migration.test_realtime_port", "rvc.lib.tools.test_split_audio"
        ]))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_before, "Existing configuration was modified"
    print("PASS: existing assets/config.json preserved", flush=True)
    print("NOT TESTED: physical audio, GPU inference and browser interaction", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
