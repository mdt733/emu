"""Keep canonical viewer files identical to the tracked package-side mirror."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_viewer_package_mirror_matches_all_tracked_source_files():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "viewer/", "emu/viewer_src/"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    original = {p[len("viewer/"):] for p in tracked if p.startswith("viewer/")}
    copied = {p[len("emu/viewer_src/"):] for p in tracked
              if p.startswith("emu/viewer_src/")}
    assert original == copied, (
        f"Missing mirror files: {sorted(original - copied)}; "
        f"obsolete mirror files: {sorted(copied - original)}"
    )
    changed = [
        path for path in sorted(original)
        if (ROOT / "viewer" / path).read_bytes()
        != (ROOT / "emu" / "viewer_src" / path).read_bytes()
    ]
    assert not changed, (
        "Tracked viewer package mirror differs from canonical source: "
        + ", ".join(changed)
        + ". Regenerate the mirror using emu viewer-build and commit it."
    )
