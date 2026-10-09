"""Compatibility entry point for the split realtime regression suite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.realtime_migration.test_volume import VolumeTests
from tools.realtime_migration.test_audio_devices import DeviceTests
from tools.realtime_migration.test_lifecycle import LifecycleTests
from tools.realtime_migration.test_custom_embedder import CustomLoaderTests, CacheTests

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
