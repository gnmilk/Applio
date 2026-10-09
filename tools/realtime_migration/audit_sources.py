"""Read-only comparison of installed source against local official Git tags."""
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT.parent / "Applio_3.6.4_broken"
OUT = Path(__file__).resolve().parent / "evidence"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def text(data):
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {ref: git("rev-parse", f"{ref}^{{}}" ).decode().strip()
            for ref in ("3.6.4", "3.6.5", "HEAD")}
    paths = git("ls-tree", "-r", "--name-only", "3.6.4").decode().splitlines()
    changes = []
    diff = []
    for rel in paths:
        if not rel.endswith((".py", ".js", ".bat", ".sh")):
            continue
        old_file = OLD / rel
        if not old_file.is_file():
            continue
        official = text(git("show", f"3.6.4:{rel}"))
        local = text(old_file.read_bytes())
        if official == local:
            continue
        changes.append(rel)
        diff.extend(difflib.unified_diff(official.splitlines(True), local.splitlines(True),
                    fromfile=f"official-3.6.4/{rel}", tofile=f"local-3.6.4/{rel}"))
    (OUT / "old-vs-official.patch").write_text("".join(diff), encoding="utf-8")
    (OUT / "upstream-3.6.4-to-3.6.5.patch").write_bytes(git(
        "diff", "3.6.4", "3.6.5", "--", "rvc/realtime", "tabs/realtime", "rvc/lib/utils.py"))
    (OUT / "new-preexisting-vs-tag.patch").write_bytes(git("diff", "3.6.5", "HEAD"))
    manifest = {"refs": refs, "old_root": str(OLD), "old_changed_source": changes,
                "initial_status": git("status", "--porcelain").decode(),
                "new_config_sha256": hashlib.sha256((ROOT / "assets/config.json").read_bytes()).hexdigest()}
    (OUT / "source-audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"refs": refs, "old_changed_source": changes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
