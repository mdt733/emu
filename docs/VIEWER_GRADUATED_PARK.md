# Viewer playground — graduated real-contact obstacle park

**Research viewer update · 31 October 2026.**
Branch: `feature/viewer-graduated-park` (built on park optical-flow and crest-seam repairs).
The published `main` showcase and the source-pinned v5031 PPO training
branches are deliberately unchanged.

## Why we changed the course

The earlier park placed a 5-cm raised box, a 6-cm crest with
unsupported transition seams, and an ~8-cm protruding log along the
first visible obstacle ring. The fixed C100 policy is measurably
strong on 2-cm ascents (8/8) and 2.5-cm ascents (4/8), but still 0/8
on supported 3-cm ascents. An apparently nice walking demonstration
that drops into unlabelled 5–8-cm geometry is misleading.

The viewer now builds **eight deterministic physical support challenges**.
All heights are true MuJoCo collision sizes (not visual decorations),
and the full 4x4 observation and optical-flow simulator continue to
see actual world support.

## The park course

Numbers correspond to the floating *visual-only* signposts in the 3D
park. Navigation intentionally steers around their physical footprints;
use manual controls when you actually want to attempt a challenge.

| # | Feature | Gentle | Balanced (default) | Stretch |
|---|---|---:|---:|---:|
| 1 | Step up / short raised landing | 1.2 cm | **2.0 cm** | 3.0 cm |
| 2 | Changing camber (shallow ridge) | 0.9 cm | **1.5 cm** | 3.0 cm |
| 3 | Joined rise → flat crest → descent | 1.5 cm | **2.5 cm** | 5.0 cm |
| 4 | Diagonal sharp ridge | 0.9 cm | **1.5 cm** | 3.0 cm |
| 5 | Gentle ramp approach → sharp step down | 1.5 cm | **2.5 cm** | 3.5 cm |
| 6 | Two shallow rolling ripples | 0.6 cm | **1.2 cm** | 2.4 cm |
| 7 | Rounded horizontal log (top height) | 1.5 cm | **2.4 cm** | 4.0 cm |
| 8 | Two low banks with a level valley between | 0.7 cm | **1.4 cm** | 2.5 cm |

These are mixed skills, not a single automatic lap in increasing
difficulty order. The flat centre remains generously clear of obstacles.
The radial hills beyond 4.2 m remain a distinct challenge controlled
by **Park max slope**. Three-level course selection changes only browser
physics, not the trained actor, and rebuilds/resets the simulation.

**Don't equate the viewer with a training examination.** Browser MuJoCo
versus MJX, differences in collision support and sensing, route commands
and a single hand-driven demonstration can alter survival. For claims,
use a matched frozen exam with the same policy/commands/conditions.
The `v5031_verified_ascent_physical_comparison` bundle supplies the
100M baseline in the repository-independent physical-exam protocol.

## How to watch via remote GPU / Tailscale

Clone the viewer branch to a *new* checkout; leave the training checkout
and results alone.

```bash
git clone --single-branch --branch feature/viewer-graduated-park \
  https://github.com/mdt733/emu.git "$HOME/emu-viewer-graduated"
cd "$HOME/emu-viewer-graduated"
source "$HOME/emu-graduation/.venv/bin/activate"
python -m emu.cli viewer --live \
  --root "$HOME/robot_lab/emu_runs/v5031_verified_ascent_fork_v2_s747" \
  --port 8768 --no-open
```

Open a second terminal on the remote/desktop client and tunnel:

```powershell
ssh -N -L 8768:127.0.0.1:8768 mdt733@100.85.225.38
```

In the browser open `http://127.0.0.1:8768/`, then **Settings → Challenge difficulty**.
Enable/disable **Park challenge course** as desired.
Use **Automatically load newer policies** to follow trained checkpoint
changes (every 12 seconds), but comparisons are clearer with auto-load
and autonomous navigation OFF while steering at a fixed requested speed.

The viewer does not serve training weights remotely without the SSH
tunnel. The trainer and public published actor are separate.

## Future research (not implemented here)

The two-slab MJX training terrain still cannot produce within-episode
continuous flat→incline→flat, rounded valleys or irregular surfaces.
Read [terrain-transition gap](TERRAIN_TRANSITION_GAP.md). Build a
**separate read-only organic-terrain exam** before adding terrain to a
new PPO campaign. Retain flat, speed, real step and constant-normal
slope controls so improvement doesn't hide regressions. Avoid dense
batched MJX heightfields on the 8 GB GPU until tested for correctness
and memory use.
