"""CPU-only regression tests for v5015 posture telemetry and old CSV compatibility."""
import csv
from emu.posture_report import summarize, report

def _write(path, heights, targets=None):
    fields = ['step', 'height'] + (['height_error_m'] if targets is not None else [])
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, h in enumerate(heights):
            row = {'step': (i + 1)*1000, 'height': h}
            if targets is not None:
                row['height_error_m'] = h - targets[i]
            w.writerow(row)

def test_old_csv_detects_drop_without_claiming_why(tmp_path):
    file = tmp_path/'metrics.csv'
    _write(file, [.25]*5 + [.22]*5)
    data = summarize(file)
    assert data['warning'] is True
    assert data['tracking_error_change_m'] is None
    assert abs(data['height_change_m'] + .03) < 1e-8

def test_commanded_crouch_is_not_false_alarm(tmp_path):
    file = tmp_path/'metrics.csv'
    h = [.25]*5 + [.22]*5
    _write(file, h, h)
    data = summarize(file)
    assert data['height_change_m'] < -.015
    assert data['tracking_error_change_m'] == 0
    assert data['warning'] is False

def test_genuine_height_target_failure_is_detected(tmp_path):
    file = tmp_path/'metrics.csv'
    _write(file, [.25]*5 + [.22]*5, [.25]*10)
    data = summarize(file)
    assert abs(data['tracking_error_change_m'] + .03) < 1e-8
    assert data['warning'] is True

def test_no_metrics_is_safe(tmp_path, capsys):
    assert report(tmp_path) == 1
    assert 'No metrics.csv' in capsys.readouterr().out
