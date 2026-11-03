"""No old source-local links dangling after removing legacy docs."""
from pathlib import Path

from scripts.check_current_docs import broken_links


def test_local_missing_paths_are_detected_but_pinned_git_is_external(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "live.md").write_text(
        "[good](../README.md) [gone](history/old.md) "
        "[archived](https://github.com/mdt733/emu/blob/pinned/docs/old.md) "
        "[jump](#heading)\n",encoding="utf-8")
    (tmp_path / "README.md").write_text("# Current",encoding="utf-8")
    assert broken_links(tmp_path)==[("docs/live.md","history/old.md")]


def test_main_checkout_document_links_are_not_broken():
    from scripts.check_current_docs import REPO_ROOT
    assert broken_links(REPO_ROOT)==[]
