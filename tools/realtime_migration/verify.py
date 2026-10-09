"""Final safe verification: compile in memory and run every local regression.

Never opens a real audio stream or executes GPU inference. Run with -B.
"""
import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-sha256", help="Expected current configuration hash; configuration is never modified")
    args = parser.parse_args()
    config_hash = hashlib.sha256((ROOT / "assets/config.json").read_bytes()).hexdigest()
    if args.config_sha256:
        assert config_hash == args.config_sha256.lower(), "Configuration changed before verification"
    results = {}
    paths = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT).decode().splitlines()
    paths = sorted(set(paths + [str(p.relative_to(ROOT)) for p in HERE.glob("*.py")]))
    for rel in paths:
        source = (ROOT / rel).read_text(encoding="utf-8-sig")
        assert "\0" not in source, f"Corrupt source: {rel}"
        compile(source, rel, "exec")
    results["syntax"] = {"status": "PASS", "files": len(paths), "writes_pycache": False}

    source_audit = json.loads((HERE / "evidence/source-audit.json").read_text(encoding="utf-8"))
    expected_head = source_audit["refs"]["HEAD"]
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    # A subsequent commit must preserve the original baseline in its ancestry;
    # verification should remain usable after committing this migration.
    subprocess.check_call(["git", "merge-base", "--is-ancestor", expected_head, current_head], cwd=ROOT)
    for backup in (HERE / "backup-20261010").rglob("*.py"):
        rel = backup.relative_to(HERE / "backup-20261010").as_posix()
        original = subprocess.check_output(["git", "show", f"{expected_head}:{rel}"], cwd=ROOT).decode()
        assert backup.read_text(encoding="utf-8") == original.replace("\r\n", "\n"), f"Backup mismatch: {rel}"
    # The old installation is a clean, single local baseline commit. Compare
    # normalized source bytes with its Git objects without refreshing its index.
    old_root = Path(source_audit["old_root"])
    for rel in source_audit["old_changed_source"] + ["rvc/lib/utils.py"]:
        original = subprocess.check_output(["git", "-C", str(old_root), "show", f"HEAD:{rel}"]).decode()
        assert (old_root / rel).read_text(encoding="utf-8") == original.replace("\r\n", "\n"), f"Old source changed: {rel}"
    results["preservation"] = {"status": "PASS", "original_baseline": expected_head, "new_head": current_head,
                                "config_sha256_at_verification": config_hash,
                                "config_changed_since_migration_start": config_hash != source_audit["new_config_sha256"],
                                "backups_match_original": True,
                                "old_source_matches_old_baseline": True}

    # Prove the complete core structure is unchanged except this one amplitude
    # expression; VAD, warmup, envelope forwarding and crossfade retain baseline.
    baseline = ast.parse((HERE / "backup-20261010/rvc/realtime/core.py").read_text(encoding="utf-8"))
    for node in ast.walk(baseline):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "resample_out":
            if node.args and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].right, ast.Name) and node.args[0].right.id == "vol_t":
                node.args[0] = node.args[0].left
    current = ast.parse((ROOT / "rvc/realtime/core.py").read_text(encoding="utf-8"))
    assert ast.dump(baseline) == ast.dump(current)
    results["core_preservation"] = {"status": "PASS", "only_behavior_change": "remove final extra RMS multiplication"}

    # Import the realtime regression first so the existing split-audio test
    # cannot replace a not-yet-imported librosa with its isolated test stub.
    from tools.realtime_migration import test_realtime_port
    from rvc.realtime import client, audio, callbacks, pipeline, worker
    from rvc.lib.utils import load_embedding
    assert "block_frame" in inspect.signature(audio.Audio.start).parameters
    assert "read_chunk_size" not in inspect.signature(audio.Audio.start).parameters
    assert "allow_fallback" in inspect.signature(load_embedding).parameters
    results["imports_and_signatures"] = {"status": "PASS", "modules": ["core", "audio", "callbacks", "pipeline", "worker", "client", "UI", "utils"]}

    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "tools.realtime_migration.test_realtime_port", "rvc.lib.tools.test_split_audio"
    ])
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        run = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (HERE / "evidence/regression-tests.log").write_text(log.getvalue(), encoding="utf-8")
    results["regressions"] = {"status": "PASS" if run.wasSuccessful() else "FAIL",
                              "run": run.testsRun, "failures": len(run.failures), "errors": len(run.errors),
                              "real_audio_streams": False, "gpu_inference": False}
    results["hardware_after_bsod"] = {"status": "NOT TESTED", "reason": "WDF_VIOLATION 0x10d parameter1=5; no further real streams/GPU tests"}
    assert hashlib.sha256((ROOT / "assets/config.json").read_bytes()).hexdigest() == config_hash, "Configuration changed during verification"
    results["preservation"]["config_unchanged_during_verification"] = True
    save_json(HERE / "evidence/final-verification.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not run.wasSuccessful():
        print(log.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
