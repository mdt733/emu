"""Read-only posture trend checker for existing EMU metrics.csv files.

Uses only the standard library; never starts training, opens checkpoints, or
modifies training directories. v5014 had only absolute height available.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from statistics import median

MIN_DROP_M = 0.015  # diagnostic threshold, not a proven failure boundary

def _number(row: dict, key: str) -> float | None:
    try:
        x = float(row.get(key, ''))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def _mid(rows, field):
    numbers = [_number(r, field) for r in rows]
    valid = [v for v in numbers if v is not None]
    return median(valid) if valid else None

def summarize(path: Path) -> dict | None:
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if _number(r, 'height') is not None]
    if not rows:
        return None
    n = len(rows)
    width = min(5, max(1, n // 3)) if n >= 3 else 1
    old, new = rows[:width], rows[-width:]
    a, b = _mid(old, 'height'), _mid(new, 'height')
    ea, eb = _mid(old, 'height_error_m'), _mid(new, 'height_error_m')
    tracking_delta = eb - ea if ea is not None and eb is not None else None
    return dict(
        path=str(path), rows=n, step=_number(rows[-1], 'step'),
        early_height_m=a, late_height_m=b, height_change_m=b - a,
        early_height_error_m=ea, late_height_error_m=eb,
        tracking_error_change_m=tracking_delta,
        warning=n >= 6 and (tracking_delta < -MIN_DROP_M if tracking_delta is not None
                             else b - a < -MIN_DROP_M),
    )

def report(root: Path) -> int:
    root = root.expanduser()
    files = ([root] if root.is_file() and root.name == 'metrics.csv' else
             sorted(root.rglob('metrics.csv')) if root.is_dir() else [])
    if not files:
        print(f'No metrics.csv under {root}. Try ~/robot_lab/emu_runs or --root /path/to/run.')
        return 1
    found = 0
    for file in files:
        try:
            item = summarize(file)
        except (OSError, UnicodeError, csv.Error) as exc:
            print(f'Could not read {file}: {exc}')
            continue
        if item is None:
            continue
        found += 1
        step = f"{int(item['step']):,}" if item['step'] is not None else '?'
        print(f"\n{file}\n  step {step}; {item['rows']} rows")
        print(f"  torso height: {item['early_height_m']*1000:.1f} → "
              f"{item['late_height_m']*1000:.1f} mm "
              f"(Δ {item['height_change_m']*1000:+.1f} mm)")
        d = item['tracking_error_change_m']
        if d is None:
            print('  target tracking unavailable (v5014); absolute height cannot establish cause')
        else:
            print(f"  measured minus commanded: {item['early_height_error_m']*1000:+.1f} → "
                  f"{item['late_height_error_m']*1000:+.1f} mm (Δ {d*1000:+.1f} mm)")
        if item['warning']:
            print('  CHECK: downward drift >=15mm. Examine fall rate and command distribution.')
    if found == 0:
        print('No finite height readings found.')
        return 1
    print('\nRead-only trend screening; lower height does not prove reward exploitation.')
    return 0

def main(argv=None):
    p = argparse.ArgumentParser(description='Read-only posture-height review for old EMU training runs')
    p.add_argument('--root', type=Path, default=Path.home() / 'robot_lab' / 'emu_runs')
    return report(p.parse_args(argv).root)

if __name__ == '__main__':
    raise SystemExit(main())
