"""The dashboard is observation only: no changes to main's PPO or hardware."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from emu.console_dashboard import (
    delta, latest_frozen, read_history, render, sampled_window, target_from_config,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dashboard_read_only_real_columns_and_missing_data(tmp_path, capsys):
    root = tmp_path / "emu-run"
    root.mkdir()
    assert "awaiting training target" in render([], run_name="emu-run", target=None)
    config = root / "training_config.json"
    config.write_text(json.dumps({"steps":120000000}), encoding="utf-8")
    assert target_from_config(root) == 120000000
    sample = [
        {"step":"95000000","reward_mean":"2.0","fall_rate":"0.2",
         "academy_near_attempts_up":"10","academy_near_successes_up":"1",
         "academy_near_attempts_down":"10","academy_near_successes_down":"1",
         "eval_broad_score":"0.80","eval_policy_step":"95000000"},
        {"step":"100000000","reward_mean":"2.5","fall_rate":"0.1",
         "academy_near_attempts_up":"8","academy_near_successes_up":"0",
         "academy_near_attempts_down":"8","academy_near_successes_down":"2",
         "eval_broad_score":"0.83","eval_policy_step":"100000000"},
        # A process boundary can repeat a step. Latest wins.
        {"step":"100000000","reward_mean":"2.6","fall_rate":"0.1",
         "academy_near_attempts_up":"8","academy_near_successes_up":"1",
         "academy_near_attempts_down":"8","academy_near_successes_down":"2",
         "eval_broad_score":"0.83","eval_policy_step":"100000000"},
    ]
    csv_path = root / "metrics.csv"
    _write(csv_path, sample)
    original = csv_path.read_bytes()
    history = read_history(csv_path)
    assert len(history) == 2
    assert delta(history,"reward_mean",precision=1) == "+0.6"
    assert sampled_window(history,"academy_near_attempts_up",
                          "academy_near_successes_up") == "1/8 (12.5%)"
    assert latest_frozen(history).startswith("0.830")
    screen = render(history,run_name="emu-run",target=120000000,width=100,height=27)
    assert "STAIRS" in screen and "sampled" in screen
    assert "NOT all-episode" in screen
    assert "100.0M / 120.0M" in screen
    assert csv_path.read_bytes() == original
    # The public CLI is dependency-free and cannot start PPO.
    from emu.cli import main
    assert main(["dashboard","--run-dir",str(root),"--once","--plain"]) == 0
    assert "EMU" in capsys.readouterr().out
    assert csv_path.read_bytes() == original


def test_dashboard_invalid_or_partial_input_is_not_fake_success(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text("step,reward_mean\nNaN,1\n",encoding="utf-8")
    assert read_history(p) == []
    no_data = render([],run_name="empty",target=100_000_000)
    assert "no sampled attempts" in no_data
    assert "not yet evaluated" in no_data
    assert "--" in no_data
