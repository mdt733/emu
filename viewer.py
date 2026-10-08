#!/usr/bin/env python3
"""Interactive MuJoCo viewer for trained EMU policies.

Running ``python viewer.py`` is the normal entry point: the viewer scans local
completed runs, scores their held-out capability, and loads the strongest
available checkpoint. ``--run-dir`` and ``--checkpoint`` override selection.

A Tk control panel provides Policy task, Manual sticks, Cruise 70%, Dynamic
demo and Random demo modes. Keyboard and optional pygame joystick controls are
also supported. Commands are clamped to the selected checkpoint's trained
envelope rather than the project's absolute design ceiling.

Manual sticks is continuous teleoperation: there is no arbitrary training
episode timeout while a human is driving. The autonomous showcase modes keep
the normal finite episode horizon, reset cleanly, and refresh a newer
``latest_mjx.pkl`` between episodes when following a live training run.

The policy forward pass mirrors training.py's MLP structure so the viewer runs
the trained network directly rather than guessing at its architecture. See
install.txt or ``python viewer.py --help`` for controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path

try:
    import tkinter as tk
    _TK_AVAILABLE = True
except ImportError:
    tk = None
    _TK_AVAILABLE = False

_VDIR = Path(__file__).resolve().parent
if str(_VDIR) not in sys.path:
    sys.path.insert(0, str(_VDIR))

_VERSION_FILE = _VDIR / 'VERSION'
VERSION = _VERSION_FILE.read_text(encoding='utf-8').strip() if _VERSION_FILE.exists() else 'unknown'
_RAW_RUN_BASE = Path(os.environ.get('EMU_RUN_ROOT', str(_VDIR.parent / 'emu_runs'))).expanduser()
_DEFAULT_VERSION_RUN_ROOT = _RAW_RUN_BASE / VERSION

VX_STEP, VY_STEP, YAW_STEP = 0.05, 0.03, 0.15
GLFW_W, GLFW_S, GLFW_A, GLFW_D = 87, 83, 65, 68
GLFW_UP, GLFW_DOWN, GLFW_R, GLFW_SPACE = 265, 264, 82, 32
GLFW_Q, GLFW_ESC = 81, 256
GLFW_N = 78
GLFW_C = 67  # cruise at --vx-frac of this checkpoint's vx_hi

AUTOPILOT_MODES = frozenset({'Policy task', 'Cruise 70%', 'Dynamic demo', 'Random demo'})


class DragPad:
    """A 2-axis virtual joystick: a circular drag pad on a tk.Canvas.
    Reports normalized (dx, dy) in [-1,1]x[-1,1] via on_change, dy already
    flipped so UP is +1 (matching normal joystick convention, not raw
    screen coordinates where y grows downward). Spring-returns to center
    and fires on_release when the mouse button is let go -- the same
    live/continuous behavior the existing gamepad path already has (see
    EmuViewer._poll_joystick): nothing here is "set and forget" the way a
    keyboard nudge is, letting go always means "stop commanding this
    axis", matching a real analog stick's own physical return spring.
    """

    def __init__(self, parent, size=150, label='', on_change=None, on_release=None):
        self.size = size
        self.radius = size / 2 - 14  # track radius, leaves room for the knob at full deflection
        self.center = size / 2
        self.on_change = on_change or (lambda dx, dy: None)
        self.on_release = on_release or (lambda: None)
        self.dx = self.dy = 0.0

        self.frame = tk.Frame(parent, bg='#1e1e1e')
        if label:
            tk.Label(self.frame, text=label, fg='#cccccc', bg='#1e1e1e', font=('TkDefaultFont', 9)).pack()
        self.canvas = tk.Canvas(self.frame, width=size, height=size, bg='#262626', highlightthickness=0)
        self.canvas.pack()
        c = self.center
        self.canvas.create_oval(c - self.radius, c - self.radius, c + self.radius, c + self.radius,
                                 outline='#555555', width=2)
        self.canvas.create_line(c - self.radius, c, c + self.radius, c, fill='#3a3a3a')
        self.canvas.create_line(c, c - self.radius, c, c + self.radius, fill='#3a3a3a')
        self.knob_r = 12
        self.knob = self.canvas.create_oval(c - self.knob_r, c - self.knob_r, c + self.knob_r, c + self.knob_r,
                                             fill='#5a9fd4', outline='#8fc1e8', width=2)
        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_press)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

    def _on_press(self, event) -> None:
        c = self.center
        ex, ey = event.x - c, event.y - c
        dist = (ex * ex + ey * ey) ** 0.5
        if dist > self.radius:
            ex, ey = ex * self.radius / dist, ey * self.radius / dist
        self.canvas.coords(self.knob, c + ex - self.knob_r, c + ey - self.knob_r,
                            c + ex + self.knob_r, c + ey + self.knob_r)
        self.dx = ex / self.radius
        self.dy = -ey / self.radius  # flip: screen-down is negative joystick-Y
        self.on_change(self.dx, self.dy)

    def _on_release(self, _event) -> None:
        c = self.center
        self.canvas.coords(self.knob, c - self.knob_r, c - self.knob_r, c + self.knob_r, c + self.knob_r)
        self.dx = self.dy = 0.0
        self.on_release()


class DragSlider:
    """A 1-axis virtual joystick (horizontal only), for yaw. Visually a
    rounded track + knob, distinct from DragPad's circle, so it reads as
    "only goes left-right" at a glance rather than looking like a second,
    smaller movement pad."""

    def __init__(self, parent, width=150, height=46, label='', on_change=None, on_release=None):
        self.width, self.height = width, height
        self.travel = width / 2 - 14
        self.cx, self.cy = width / 2, height / 2
        self.on_change = on_change or (lambda dx: None)
        self.on_release = on_release or (lambda: None)
        self.dx = 0.0

        self.frame = tk.Frame(parent, bg='#1e1e1e')
        if label:
            tk.Label(self.frame, text=label, fg='#cccccc', bg='#1e1e1e', font=('TkDefaultFont', 9)).pack()
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg='#262626', highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_line(14, self.cy, width - 14, self.cy, fill='#555555', width=4, capstyle=tk.ROUND)
        self.knob_r = 13
        self.knob = self.canvas.create_oval(self.cx - self.knob_r, self.cy - self.knob_r,
                                             self.cx + self.knob_r, self.cy + self.knob_r,
                                             fill='#5a9fd4', outline='#8fc1e8', width=2)
        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_press)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

    def _on_press(self, event) -> None:
        ex = event.x - self.cx
        ex = max(-self.travel, min(self.travel, ex))
        self.canvas.coords(self.knob, self.cx + ex - self.knob_r, self.cy - self.knob_r,
                            self.cx + ex + self.knob_r, self.cy + self.knob_r)
        self.dx = ex / self.travel
        self.on_change(self.dx)

    def _on_release(self, _event) -> None:
        self.canvas.coords(self.knob, self.cx - self.knob_r, self.cy - self.knob_r,
                            self.cx + self.knob_r, self.cy + self.knob_r)
        self.dx = 0.0
        self.on_release()


class ControlPanel:
    """A separate Tk window with the two drag sticks plus button
    equivalents of every keyboard action, so nothing requires the MuJoCo
    render window to have keyboard focus. Runs cooperatively on the
    caller's own thread (call .pump() once per frame from the main loop)
    -- NOT root.mainloop(), which would block that loop; MuJoCo's own
    render/event handling already lives on launch_passive's own thread,
    so this class only ever touches the main thread, avoiding any
    cross-toolkit threading question entirely.

    Nothing here touches jax/mujoco directly -- it only calls the small
    set of callbacks passed in, so this class (and DragPad/DragSlider
    above) can be constructed and driven with real synthetic Tk events in
    a test with no checkpoint, no environment, and no GPU.
    """

    def __init__(self, on_move, on_release_move, on_reset, on_random, on_cruise, on_release_policy, on_quit, on_mode=None, policy_label=""):
        self.root = tk.Tk()
        self.root.title('EMU control panel')
        self.root.configure(bg='#1e1e1e')
        self.root.protocol('WM_DELETE_WINDOW', on_quit)
        self.closed = False

        sticks = tk.Frame(self.root, bg='#1e1e1e')
        sticks.pack(padx=14, pady=(14, 6))
        self.pad = DragPad(sticks, label='vx / vy  (drag)', on_change=lambda dx, dy: on_move(dx, dy, None),
                            on_release=lambda: on_release_move('vxvy'))
        self.pad.frame.pack(side=tk.LEFT, padx=(0, 18))
        self.slider = DragSlider(sticks, label='yaw  (drag left/right)',
                                  on_change=lambda dx: on_move(None, None, -dx),
                                  on_release=lambda: on_release_move('yaw'))
        self.slider.frame.pack(side=tk.LEFT, anchor=tk.N, pady=(24, 0))

        self.readout = tk.Label(self.root, text='vx=+0.000  vy=+0.000  yaw=+0.000',
                                 fg='#8fc1e8', bg='#1e1e1e', font=('TkFixedFont', 10))
        self.readout.pack(pady=(0, 6))
        if policy_label:
            tk.Label(self.root, text=policy_label, fg='#aaaaaa', bg='#1e1e1e', font=('TkDefaultFont', 8),
                     wraplength=520, justify=tk.LEFT).pack(padx=12, pady=(0, 6), fill=tk.X)

        mode_row = tk.Frame(self.root, bg='#1e1e1e')
        mode_row.pack(pady=(0, 8))
        tk.Label(mode_row, text='Control mode:', fg='#cccccc', bg='#1e1e1e').pack(side=tk.LEFT, padx=(0, 6))
        self.mode_var = tk.StringVar(value='Policy task')
        modes = ('Policy task', 'Manual sticks', 'Cruise 70%', 'Dynamic demo', 'Random demo')
        self.mode_menu = tk.OptionMenu(mode_row, self.mode_var, *modes, command=(lambda value: on_mode(value) if on_mode else None))
        self.mode_menu.config(bg='#333333', fg='#eeeeee', activebackground='#444444', highlightthickness=0)
        self.mode_menu['menu'].config(bg='#333333', fg='#eeeeee')
        self.mode_menu.pack(side=tk.LEFT)

        btns = tk.Frame(self.root, bg='#1e1e1e')
        btns.pack(pady=(0, 14))
        style = dict(bg='#333333', fg='#eeeeee', activebackground='#444444', activeforeground='#ffffff',
                     relief=tk.FLAT, padx=10, pady=4)
        tk.Button(btns, text='Reset (R)', command=on_reset, **style).grid(row=0, column=0, padx=4)
        tk.Button(btns, text='Random (N)', command=on_random, **style).grid(row=0, column=1, padx=4)
        tk.Button(btns, text='Cruise (C)', command=on_cruise, **style).grid(row=0, column=2, padx=4)
        tk.Button(btns, text="Release to policy (Space)", command=on_release_policy, **style).grid(row=0, column=3, padx=4)
        tk.Button(btns, text='Quit (Q)', command=on_quit, **style).grid(row=0, column=4, padx=4)

    def set_readout(self, vx: float, vy: float, yaw: float, manual: bool) -> None:
        tag = 'MANUAL' if manual else 'policy'
        self.readout.config(text=f'vx={vx:+.3f}  vy={vy:+.3f}  yaw={yaw:+.3f}   [{tag}]')

    def set_mode(self, mode: str) -> None:
        """Update the dropdown without invoking its OptionMenu callback."""
        self.mode_var.set(mode)

    def pump(self) -> bool:
        """Call once per frame from the main loop. Returns False once the
        window has been closed, so the caller can stop pumping it."""
        if self.closed:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            return False
        return True

    def destroy(self) -> None:
        if not self.closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self.closed = True


def _checkpoint_for_run(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    ck = run_dir / 'checkpoints'
    for name in ('best_broad_eval.pkl', 'latest_mjx.pkl', 'best_known_good.pkl', 'best_effort.pkl', 'best_seen.pkl'):
        p = ck / name
        if p.exists():
            return p
    return None


def _last_csv_row(path: Path) -> dict:
    try:
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default))
        return x if x == x else default
    except Exception:
        return default


def discover_best_checkpoint(search_root: Path | None = None) -> tuple[Path | None, str]:
    """Find the strongest completed local EMU run without needing CLI flags."""
    roots = []
    if search_root is not None:
        roots.append(Path(search_root))
    roots += [_DEFAULT_VERSION_RUN_ROOT / 'rung_runs', _DEFAULT_VERSION_RUN_ROOT,
              _VDIR / 'library' / 'runs', Path.cwd() / 'library' / 'runs']
    seen = set(); candidates = []
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen or not root.exists():
            continue
        seen.add(root)
        dirs = [root] + [d for d in root.iterdir() if d.is_dir()]
        for d in dirs:
            ck = _checkpoint_for_run(d)
            if ck is None:
                continue
            row = _last_csv_row(d / 'fixed_eval.csv')
            if row:
                core = _f(row, 'eval_core_survival', 0.0)
                stress = _f(row, 'eval_stress30_survival', 0.0)
                dyn = _f(row, 'eval_dynamic50_survival', stress)
                dyn80 = _f(row, 'eval_dynamic80_survival', dyn)
                res = _f(row, 'eval_resilience50_survival', stress)
                res75 = _f(row, 'eval_resilience75_survival', res)
                robust = _f(row, 'eval_robust40_survival', stress)
                broad = _f(row, 'eval_broad_tail3', float('nan'))
                if not (broad == broad):
                    broad = _f(row, 'eval_broad_score', float('nan'))
                legacy_score = 0.05*core + 0.10*stress + 0.15*dyn + 0.20*dyn80 + 0.15*res + 0.20*res75 + 0.15*robust
                score = broad if broad == broad else legacy_score
                why = (f'held-out={score:.3f} core={core:.3f} stress={stress:.3f} '
                       f'dyn50={dyn:.3f} dyn80={dyn80:.3f} res50={res:.3f} res75={res75:.3f} robust={robust:.3f}')
            else:
                score = -1.0
                why = 'no fixed_eval.csv; checkpoint fallback'
            candidates.append((score, ck.stat().st_mtime, ck, d, why))
    if not candidates:
        return None, 'no local trained checkpoint discovered'
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, ck, run_dir, why = candidates[0]
    return ck, f'{run_dir.name}: {why}'


def find_latest_checkpoint(run_dir: Path | None) -> Path | None:
    return _checkpoint_for_run(run_dir)


def mlp_apply(layers, x, jnp):
    """Exact copy of training.py's mlp() closure: tanh hidden layers, plain
    linear output. Verified against a real checkpoint's actual param shapes
    before this file was written."""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer['w'] + layer['b'])
    last = layers[-1]
    return x @ last['w'] + last['b']


class EmuViewer:
    def __init__(self, args: argparse.Namespace):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        self.args = args
        self.speed = max(0.01, float(args.speed))

        from environment import (EmuMJXEnv, TASK_VX, TASK_VY, TASK_YAW, CONTROL_DT,
                                 SEED_VX_LO, SEED_VX_HI, VX_LO_MIN, VX_HI_MAX,
                                 YAW_MAX_RAD_S_LIMIT, VY_MAX_MPS_LIMIT)
        self.TASK_VX, self.TASK_VY, self.TASK_YAW = TASK_VX, TASK_VY, TASK_YAW
        self.seed_vx_lo, self.seed_vx_hi = SEED_VX_LO, SEED_VX_HI
        self.control_dt = CONTROL_DT

        # training.py's PPOState gets pickled as __main__.PPOState, since
        # training.py runs as __main__ — unpickling it from any other script
        # fails with "Can't get attribute 'PPOState' on module '__main__'"
        # unless __main__ has that name too. Confirmed by reproducing this
        # exact failure against a real checkpoint before adding the line below.
        import training as _training_module
        sys.modules['__main__'].PPOState = _training_module.PPOState
        self._training_module = _training_module

        self.policy_layers = None
        self.loaded_checkpoint = ""
        self.curriculum_params: dict = {}
        self.obs_mean = None  # set below; None means "no normalization" (neutral)
        self.obs_std = None
        self.obs_norm_mode = 'all'
        self.obs_norm_mask = None
        self.run_training_args: dict = {}
        # Branch-specific command sampler contract. v5006 trains mixed commands
        # inside a p-norm blob; the old viewer only knew each axis' independent
        # frontier and therefore generated untrained rectangular corners.
        self.combo_mix_frac = 0.0
        self.combo_p = 1.35
        self.combo_probe_frac = 0.08
        self.combo_radius = 1.0
        self.regime_command_frac = 0.5
        self.watch_run_dir: Path | None = None
        self._checkpoint_mtime_ns = 0
        self.demo_episode_steps = int(args.max_episode_steps)
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
        self.discovery_reason = ''
        if args.checkpoint:
            ckpt = Path(args.checkpoint).expanduser().resolve()
        elif run_dir is not None:
            ckpt = find_latest_checkpoint(run_dir)
            self.watch_run_dir = run_dir
        else:
            ckpt, self.discovery_reason = discover_best_checkpoint()
            if ckpt is not None:
                self.watch_run_dir = ckpt.parent.parent
                print(f'[viewer] Auto-selected current best model: {ckpt}')
                print(f'[viewer] Selection basis: {self.discovery_reason}')
        if ckpt is not None and ckpt.exists():
            try:
                with ckpt.open('rb') as f:
                    payload = pickle.load(f)
                self.policy_layers = payload['state'].params['pi']
                self.curriculum_params = payload.get('curriculum_params', {}) or {}
                obs_norm = payload.get('obs_norm')
                if obs_norm is not None:
                    # Checkpoints from before observation normalization was added
                    # don't have this key — get() above handles that gracefully.
                    # A policy trained on normalized obs produces garbage actions
                    # fed raw obs, so this isn't optional once a checkpoint has it.
                    self.obs_mean = jnp.asarray(obs_norm['mean'])
                    self.obs_std = jnp.sqrt(jnp.asarray(obs_norm['var'])) + 1e-6
                    print('[viewer] Loaded obs normalization stats from checkpoint.')
                self.loaded_checkpoint = str(ckpt)
                try:
                    self._checkpoint_mtime_ns = ckpt.stat().st_mtime_ns
                except OSError:
                    self._checkpoint_mtime_ns = 0
                print(f'[viewer] Loaded checkpoint: {ckpt}')

                # The checkpoint itself stores policy/curriculum/normalizer state, while
                # branch-specific environment construction flags live beside it in
                # run_meta.json. Reconstruct those here so an actuator-realism or
                # latency-trained policy is not silently demonstrated in a different
                # physics model. Older/copied checkpoints simply fall back to defaults.
                run_meta_path = ckpt.parent.parent / 'run_meta.json'
                if run_meta_path.exists():
                    try:
                        run_meta = json.loads(run_meta_path.read_text(encoding='utf-8'))
                        self.run_training_args = dict(run_meta.get('args') or {})
                    except Exception as meta_exc:
                        print(f'[viewer] Could not read {run_meta_path.name}: {meta_exc!r}; using default viewer physics.')
            except Exception as exc:
                print(f'[viewer] Checkpoint load failed — running zero action instead.\n{exc}')
        else:
            print('[viewer] No checkpoint found — running zero action (sanity-check mode).')

        ta = self.run_training_args
        self.obs_norm_mode = str(ta.get('obs_norm_mode', 'all') or 'all')
        try:
            self.obs_norm_mask = jnp.asarray(_training_module.build_obs_norm_mask(self.obs_norm_mode), dtype=jnp.float32)
        except Exception:
            self.obs_norm_mode = 'all'
            self.obs_norm_mask = jnp.ones((_training_module.OBS_DIM,), dtype=jnp.float32)
        # The viewer owns its own episode horizon. Keep the environment's internal
        # timeout effectively out of the way so Manual sticks can be genuine
        # continuous teleoperation; autonomous modes still reset every
        # ``demo_episode_steps`` in the viewer loop below.
        viewer_internal_horizon = max(int(args.max_episode_steps), 10_000_000)
        self.env = EmuMJXEnv(
            str(_VDIR / 'robot.xml'), num_envs=1, seed=args.seed, max_episode_steps=viewer_internal_horizon,
            actuator_realism=bool(ta.get('actuator_realism', False)),
            comm_latency_steps=int(ta.get('comm_latency_steps', 0) or 0),
            push_max_force_n=ta.get('push_max_force_n', None),
            stepping_activity_target=ta.get('stepping_activity_target', None),
            speed_tolerance=ta.get('speed_tolerance', None),
            legacy_rng_reuse=bool(ta.get('legacy_rng_reuse', False)),
            vx_hi_max=float(ta.get('vx_hi_max', VX_HI_MAX)),
            vx_lo_min=float(ta.get('vx_lo_min', VX_LO_MIN)),
            vy_max_mps=float(ta.get('vy_max_mps', VY_MAX_MPS_LIMIT)),
            yaw_max_rad_s=float(ta.get('yaw_max_rad_s', YAW_MAX_RAD_S_LIMIT)),
            target_rate_limit_rad_s=float(ta.get('target_rate_limit_rad_s', 1.75)),
            disable_target_slew=bool(ta.get('disable_target_slew', False)),
            phase_speed_gain=float(ta.get('phase_speed_gain', 0.0)),
            phase_rate_max_hz=float(ta.get('phase_rate_max_hz', 2.6)),
            sensor_noise_scale=float(ta.get('sensor_noise_scale', 1.0)),
            open_frontier=bool(ta.get('open_frontier', False)),
            combo_mix_frac=float(ta.get('combo_mix_frac', 0.0) or 0.0),
            combo_p=float(ta.get('combo_p', 1.35) or 1.35),
            combo_probe_frac=float(ta.get('combo_probe_frac', 0.08) or 0.08),
            phase_obs_scale=float(ta.get('phase_obs_scale', 1.0) or 0.0),
        )
        print(f'[viewer] Observation normalization mode: {self.obs_norm_mode}')
        print('[viewer] Environment match: '
              f'actuator_realism={self.env.actuator_realism} '
              f'latency={self.env.comm_latency_steps} '
              f'push_max={self.env.push_max_force_n:.3f}N '
              f'no_slew={self.env.disable_target_slew}')

        self.params_arr = self.env.params_array(self.curriculum_params)
        self._reset_serial = 0
        self.state, self.obs = self._reset()
        self.cmd_vx = self.cmd_vy = self.cmd_yaw = 0.0
        self.manual = False
        self.step_count = 0
        self.control_mode = "Policy task"
        self.demo_ticks = 0

        # The actual current command window this checkpoint was trained
        # on, so manual control (WASD clamps, the random-command key) stays
        # inside what the policy has actually seen. curriculum_params
        # stores steady_difficulty, not the physical vx_lo/vx_hi/yaw_max/
        # vy_max bounds directly -- derived here via the same plain linear
        # interpolation (seed + difficulty*(ceiling-seed)) environment.py's
        # own _lerp uses. Falls back to steady_difficulty=0.0 (the seed
        # defaults) if an older checkpoint has no curriculum_params at all.
        self._command_constants = (SEED_VX_LO, SEED_VX_HI, self.env.vx_lo_min, self.env.vx_hi_max,
                                   self.env.yaw_max_rad_s, self.env.vy_max_mps)
        self._refresh_command_envelope()
        self._refresh_command_profile()

        # --vx-frac auto-commands a straight-line cruise at launch instead
        # of starting at the policy's own command / zero. Fraction of THIS
        # checkpoint's own vx_hi -- see the module docstring for why
        # that's the right denominator, not VX_HI_MAX.
        self.vx_frac = float(getattr(args, 'vx_frac', 0.0) or 0.0)
        if self.vx_frac > 0.0:
            self._apply_cruise_command(announce=False)
            print(f'[viewer] Auto-cruise: vx={self.cmd_vx:+.3f} m/s '
                  f'({100*self.vx_frac:.0f}% of this checkpoint\'s vx_hi={self.vx_hi:.3f} m/s), vy=0, yaw=0')

        # mujoco.viewer.launch_passive only ever delivers key events
        # through a key_callback argument -- there is no polling API for
        # this. key_callback runs on the viewer's own render thread, so
        # pending keys go through a lock rather than being touched
        # directly from run()'s loop.
        self._pending_keys: list[int] = []
        self._keys_lock = threading.Lock()

        # Optional joystick/gamepad input, additive to WASD (both work
        # simultaneously; whichever moved most recently wins since both
        # just set self.cmd_vx/vy/yaw + self.manual the same way).
        # pygame's axis NUMBERING for a given physical stick is not
        # consistent across OS/driver/controller combinations (XInput vs
        # DirectInput vs raw HID all differ) -- rather than guess, axis
        # indices are CLI-configurable and --joystick-debug prints every
        # raw axis value live so the right ones are obvious in seconds
        # instead of guessed at.
        self.joystick = None
        self._pygame = None
        if getattr(args, 'joystick', False):
            self._init_joystick()

        # Control panel: a separate Tk window with mouse-draggable sticks
        # and buttons for everything keyboard already does, so none of it
        # needs the MuJoCo render window to have focus (a real, repeated
        # annoyance with launch_passive's key handling). Optional and
        # gracefully degrades: if tkinter genuinely isn't installed,
        # keyboard/gamepad control still works exactly as before.
        self.panel = None
        self._quit_requested = False
        if not getattr(args, 'no_panel', False):
            if _TK_AVAILABLE:
                try:
                    self.panel = ControlPanel(
                        on_move=self._panel_on_move, on_release_move=self._panel_on_release_move,
                        on_reset=self._panel_on_reset, on_random=self._panel_on_random,
                        on_cruise=self._panel_on_cruise, on_release_policy=self._panel_on_release_policy,
                        on_quit=self._panel_on_quit, on_mode=self._set_control_mode,
                        policy_label=(Path(self.loaded_checkpoint).parent.parent.name if self.loaded_checkpoint else 'zero-action sanity check'))
                except tk.TclError as exc:
                    # tkinter importing fine doesn't mean a display is actually
                    # reachable -- a genuinely headless remote server (no X
                    # server, no DISPLAY) fails here, not at import time.
                    # Keyboard/gamepad control still works via the MuJoCo
                    # window itself, same as --no-panel.
                    print(f'[viewer] --no-panel not set, but the control panel window could not be '
                          f'opened ({exc!r}) -- likely no display available on this machine. '
                          f'Continuing with keyboard/gamepad control only.')
            else:
                print('[viewer] --no-panel not set, but tkinter is not installed -- control panel '
                      'disabled, keyboard/gamepad control still works. Install it with '
                      '`apt install python3-tk` (Linux) or `dnf install python3-tkinter` -- it '
                      'ships with the standard python.org installer on Windows/Mac already.')

    def _panel_on_move(self, dx: float | None, dy: float | None, yaw: float | None) -> None:
        """Wired to both DragPad (dx, dy given, yaw None) and DragSlider
        (yaw given, dx/dy None) -- only touches the axes it was actually
        called for, so dragging one stick never resets the other."""
        self._enter_manual_mode()
        if dx is not None and dy is not None:
            vx_lo, vx_hi = sorted((self.vx_lo, self.vx_hi))
            # Same remap DragPad/the gamepad path both use: dy in [-1,1] onto
            # [vx_lo, vx_hi] directly (not symmetric around 0), since vx_lo
            # may itself be a small negative/positive seed value, not -vx_hi.
            self.cmd_vx = vx_lo + (dy + 1.0) / 2.0 * (vx_hi - vx_lo)
            self.cmd_vy = dx * self.vy_max
            self.manual = True
        if yaw is not None:
            self.cmd_yaw = yaw * self.yaw_max
            self.manual = True

    def _refresh_command_envelope(self) -> None:
        """Recompute joystick limits from the checkpoint's current frontier."""
        (seed_vx_lo, seed_vx_hi, vx_lo_min, vx_hi_max,
         yaw_max_limit, vy_max_limit) = self._command_constants
        cp = self.curriculum_params
        if bool(self.run_training_args.get('open_frontier', False)):
            self.vx_lo = float(cp.get('frontier_vx_lo', -0.20))
            self.vx_hi = float(cp.get('frontier_vx_hi', 0.60))
            self.yaw_max = abs(float(cp.get('frontier_yaw', 1.50)))
            self.vy_max = abs(float(cp.get('frontier_vy', 0.22)))
        else:
            steady_d = max(0.0, min(1.0, float(cp.get('steady_difficulty', 0.0))))
            self.vx_lo = seed_vx_lo + steady_d * (vx_lo_min - seed_vx_lo)
            self.vx_hi = seed_vx_hi + steady_d * (vx_hi_max - seed_vx_hi)
            self.yaw_max = steady_d * yaw_max_limit
            self.vy_max = steady_d * vy_max_limit

    def _refresh_command_profile(self) -> None:
        """Refresh the command-generation contract from the trained branch.

        v5006 does not train a rectangular vx/vy/yaw box. Combo branches train
        mixed commands inside a p-norm blob with its own frontier radius. The
        viewer must therefore use the same geometry for Random/Dynamic/manual
        control, otherwise it can ask a good policy to execute combinations it
        never saw in training.
        """
        ta = self.run_training_args or {}
        self.combo_mix_frac = max(0.0, min(1.0, float(ta.get('combo_mix_frac', 0.0) or 0.0)))
        self.combo_p = max(1.0, float(ta.get('combo_p', 1.35) or 1.35))
        self.combo_probe_frac = max(0.0, float(ta.get('combo_probe_frac', 0.08) or 0.08))
        self.regime_command_frac = max(0.0, min(1.0, float(ta.get('regime_command_frac', 0.5) or 0.5)))
        if bool(ta.get('open_frontier', False)):
            self.combo_radius = max(0.10, float(self.curriculum_params.get('frontier_combo_radius', 1.0)))
        else:
            self.combo_radius = 1.0

    def _combo_norm(self, vx: float, vy: float, yaw: float) -> float:
        """Return the v5006 normalized combined-command p-norm."""
        p = max(1.0, float(getattr(self, 'combo_p', 1.35)))
        vx_scale = max(self.vx_hi if vx >= 0.0 else -self.vx_lo, 1e-6)
        nx = vx / vx_scale
        ny = vy / max(self.vy_max, 1e-6)
        nz = yaw / max(self.yaw_max, 1e-6)
        return (abs(nx) ** p + abs(ny) ** p + abs(nz) ** p) ** (1.0 / p)

    def _project_to_trained_combo(self, vx: float, vy: float, yaw: float) -> tuple[float, float, float]:
        """Project a command onto the trained mixed-command blob.

        Axis-isolated branches deliberately retain their rectangular semantics;
        combo branches constrain simultaneous active axes to the current learned
        combo frontier. This protects manual teleop from accidentally requesting
        an untrained corner while still allowing a --raw/stress path later.
        """
        mix = float(getattr(self, 'combo_mix_frac', 0.0))
        radius = float(getattr(self, 'combo_radius', 1.0))
        if mix <= 0.0 or self.vy_max <= 1e-9 or self.yaw_max <= 1e-9:
            return vx, vy, yaw
        n = self._combo_norm(vx, vy, yaw)
        if n <= radius or n <= 1e-9:
            return vx, vy, yaw
        scale = radius / n
        return vx * scale, vy * scale, yaw * scale

    def _sample_trained_random_command(self) -> tuple[float, float, float, bool]:
        """Sample a command using the branch's actual v5006 task geometry.

        Returns (vx, vy, yaw, mixed). The mixed path mirrors environment.py's
        birth sampler: independent physical draw -> p-norm projection -> radial
        interior/probe sample. The axis path mirrors its axis-isolation slice.
        """
        import random
        lo, hi = sorted((self.vx_lo, self.vx_hi))
        vx = random.uniform(lo, hi)
        vy = random.uniform(-self.vy_max, self.vy_max)
        yaw = random.uniform(-self.yaw_max, self.yaw_max)
        if self.combo_mix_frac <= 0.0 or random.random() >= self.combo_mix_frac:
            axis = random.randrange(3)
            mid = 0.5 * (self.seed_vx_lo + self.seed_vx_hi)
            if axis == 0:
                return vx, 0.0, 0.0, False
            if axis == 1:
                return max(lo, min(hi, mid)), vy, 0.0, False
            return max(lo, min(hi, mid)), 0.0, yaw, False

        vx_scale = max(self.vx_hi if vx >= 0.0 else -self.vx_lo, 1e-6)
        nx = vx / vx_scale
        ny = vy / max(self.vy_max, 1e-6)
        nz = yaw / max(self.yaw_max, 1e-6)
        norm = (abs(nx) ** self.combo_p + abs(ny) ** self.combo_p + abs(nz) ** self.combo_p) ** (1.0 / self.combo_p)
        radial = 0.35 + 0.65 * random.random()
        probe = self.combo_radius * radial
        scale = probe / max(norm, 1e-6)
        return vx * scale, vy * scale, yaw * scale, True

    def _maybe_reload_latest_checkpoint(self) -> bool:
        """Adopt a newer checkpoint from the watched run between demo episodes.

        Explicit ``--checkpoint`` means a frozen snapshot and is never followed.
        ``--run-dir`` and auto-discovery follow that run's ``latest_mjx.pkl``.
        A partially-written checkpoint is harmless: keep the current policy and
        try again at the next autonomous episode boundary.
        """
        if self.watch_run_dir is None:
            return False
        latest = self.watch_run_dir / 'checkpoints' / 'latest_mjx.pkl'
        if not latest.exists():
            return False
        try:
            mtime_ns = latest.stat().st_mtime_ns
        except OSError:
            return False
        if str(latest) == self.loaded_checkpoint and mtime_ns <= self._checkpoint_mtime_ns:
            return False
        try:
            with latest.open('rb') as f:
                payload = pickle.load(f)
            self.policy_layers = payload['state'].params['pi']
            self.curriculum_params = payload.get('curriculum_params', {}) or {}
            obs_norm = payload.get('obs_norm')
            if obs_norm is None:
                self.obs_mean = self.obs_std = None
            else:
                self.obs_mean = self.jnp.asarray(obs_norm['mean'])
                self.obs_std = self.jnp.sqrt(self.jnp.asarray(obs_norm['var'])) + 1e-6

            # A run's construction flags are normally immutable, but re-read the
            # metadata so copied/resumed runs do not silently use stale norm mode.
            run_meta_path = self.watch_run_dir / 'run_meta.json'
            if run_meta_path.exists():
                try:
                    run_meta = json.loads(run_meta_path.read_text(encoding='utf-8'))
                    new_args = dict(run_meta.get('args') or {})
                    self._command_constants = (SEED_VX_LO, SEED_VX_HI,
                                               float(new_args.get('vx_lo_min', VX_LO_MIN)),
                                               float(new_args.get('vx_hi_max', VX_HI_MAX)),
                                               float(new_args.get('yaw_max_rad_s', YAW_MAX_RAD_S_LIMIT)),
                                               float(new_args.get('vy_max_mps', VY_MAX_MPS_LIMIT)))
                    new_mode = str(new_args.get('obs_norm_mode', 'all') or 'all')
                    if new_mode != self.obs_norm_mode:
                        self.obs_norm_mode = new_mode
                        self.obs_norm_mask = self.jnp.asarray(
                            self._training_module.build_obs_norm_mask(self.obs_norm_mode),
                            dtype=self.jnp.float32)
                    self.run_training_args = new_args
                except Exception as meta_exc:
                    print(f'[viewer] Live checkpoint metadata refresh skipped: {meta_exc!r}')

            self.params_arr = self.env.params_array(self.curriculum_params)
            self._refresh_command_envelope()
            self._refresh_command_profile()
            self.loaded_checkpoint = str(latest)
            self._checkpoint_mtime_ns = mtime_ns
            print('[viewer] Adopted newer training checkpoint between showcase episodes: '
                  f'{latest.name}  command window vx∈[{min(self.vx_lo,self.vx_hi):.3f},'
                  f'{max(self.vx_lo,self.vx_hi):.3f}] vy∈±{self.vy_max:.3f} yaw∈±{self.yaw_max:.3f}')
            return True
        except Exception as exc:
            print(f'[viewer] New checkpoint exists but could not be loaded yet ({exc!r}); '
                  'keeping the current policy until the next episode boundary.')
            return False

    def _reset_showcase_episode(self, reason: str) -> None:
        """Boundary for autonomous demo episodes; safe point to hot-refresh."""
        refreshed = self._maybe_reload_latest_checkpoint()
        self.state, self.obs = self._reset()
        self.step_count = 0
        self.demo_ticks = 0
        if self.control_mode == 'Random demo':
            self._sample_random_command()
        elif self.control_mode == 'Dynamic demo':
            self._update_demo_command(force=True)
        elif self.control_mode == 'Cruise 70%':
            self._apply_cruise_command(0.70, announce=False)
        suffix = ' + loaded newer checkpoint' if refreshed else ''
        print(f'[viewer] Showcase episode reset: {reason}{suffix}')

    def _panel_on_release_move(self, axes: str) -> None:
        """Spring-return to the same neutral command the pad maps to.

        vx is not generally symmetric around zero (the seed gait is deliberately
        forward-moving), so the centre of the vx pad is the midpoint of the
        checkpoint's trained vx window, not an untrained hard zero.
        """
        if axes == 'vxvy':
            vx_lo, vx_hi = sorted((self.vx_lo, self.vx_hi))
            self.cmd_vx = 0.5 * (vx_lo + vx_hi)
            self.cmd_vy = 0.0
        elif axes == 'yaw':
            self.cmd_yaw = 0.0
        self._enter_manual_mode()

    def _panel_on_reset(self) -> None:
        self.state, self.obs = self._reset()
        self.step_count = 0
        print('[viewer] Episode reset')

    def _panel_on_random(self) -> None:
        self._set_control_mode('Random demo')

    def _panel_on_cruise(self) -> None:
        self._set_control_mode('Cruise 70%')

    def _panel_on_release_policy(self) -> None:
        self._set_control_mode('Policy task')

    def _panel_on_quit(self) -> None:
        self._quit_requested = True

    def _set_control_mode(self, mode: str) -> None:
        self.control_mode = str(mode)
        self.demo_ticks = 0
        if self.panel is not None:
            self.panel.set_mode(self.control_mode)
        if mode == 'Policy task':
            self._release_to_policy_task(announce=False)
        elif mode == 'Cruise 70%':
            self._apply_cruise_command(0.70, announce=False)
        elif mode == 'Manual sticks':
            # Enter manual control without an artificial command jump. The
            # sticks can move away from the currently sampled task from here.
            task = self.jax.device_get(self.state.task)[0]
            self.cmd_vx = float(task[self.TASK_VX])
            self.cmd_vy = float(task[self.TASK_VY])
            self.cmd_yaw = float(task[self.TASK_YAW])
            self.manual = True
        elif mode == 'Random demo':
            self._sample_random_command()
        elif mode == 'Dynamic demo':
            self.manual = True; self._update_demo_command(force=True)
        print(f'[viewer] Control mode: {self.control_mode}')

    def _enter_manual_mode(self) -> None:
        """Human input owns the command stream until explicitly released.

        This is deliberately separate from ``self.manual``: Cruise/Random/
        Dynamic also inject commands into the policy, but they are autonomous
        showcase modes and retain finite 700-step episodes. A stick/keyboard
        nudge means genuine teleoperation, so it switches the mode itself.
        """
        self.manual = True
        if self.control_mode != 'Manual sticks':
            self.control_mode = 'Manual sticks'
            self.demo_ticks = 0
            if self.panel is not None:
                self.panel.set_mode('Manual sticks')
            print('[viewer] Control mode: Manual sticks (continuous teleoperation)')

    def _update_demo_command(self, force: bool = False) -> None:
        if self.control_mode == 'Random demo':
            period = max(1, int(round(1.8 / self.control_dt)))
            if force or self.demo_ticks % period == 0:
                self._sample_random_command()
            self.demo_ticks += 1; return
        if self.control_mode != 'Dynamic demo':
            return
        period = max(1, int(round(1.0 / self.control_dt)))
        if force or self.demo_ticks % period == 0:
            k = (self.demo_ticks // period) % 6
            hi = max(0.0, self.vx_hi); lo = min(0.0, self.vx_lo)
            raw_seq = [(0.85*hi,0.0,0.0),(0.45*hi,0.65*self.vy_max,0.0),(0.55*hi,0.0,0.65*self.yaw_max),
                       (max(lo,-0.55*abs(lo if lo < 0.0 else hi)),0.0,0.0),(0.45*hi,-0.55*self.vy_max,-0.50*self.yaw_max),(0.35*hi,0.0,-0.65*self.yaw_max)]
            self.cmd_vx, self.cmd_vy, self.cmd_yaw = self._project_to_trained_combo(*raw_seq[k]); self.manual = True
            print(f'[viewer] Dynamic demo -> vx={self.cmd_vx:+.3f} vy={self.cmd_vy:+.3f} yaw={self.cmd_yaw:+.3f}')
        self.demo_ticks += 1

    def _init_joystick(self) -> None:
        try:
            import pygame
        except ImportError:
            print('[viewer] --joystick requested but pygame is not installed. '
                  'Install it with: pip install pygame')
            return
        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        if n == 0:
            print('[viewer] --joystick requested but no controller was detected by pygame. '
                  'Falling back to keyboard only. (On WSL: pygame generally cannot see a USB '
                  'controller at all -- USB devices are not passed through by default. Running '
                  'this script with native Windows Python, outside WSL, is the simplest fix; it '
                  'only needs CPU + pygame, nothing WSL/CUDA-specific.)')
            return
        idx = min(max(0, int(self.args.joystick_index)), n - 1)
        self.joystick = pygame.joystick.Joystick(idx)
        self.joystick.init()
        print(f'[viewer] Joystick connected: {self.joystick.get_name()} '
              f'({self.joystick.get_numaxes()} axes, {self.joystick.get_numbuttons()} buttons)')

    def _poll_joystick(self) -> None:
        if self.joystick is None:
            return
        pygame = self._pygame
        pygame.event.pump()
        n_axes = self.joystick.get_numaxes()
        deadzone = float(self.args.joystick_deadzone)

        def axis(i: int) -> float:
            if i < 0 or i >= n_axes:
                return 0.0
            v = self.joystick.get_axis(i)
            return 0.0 if abs(v) < deadzone else v

        if self.args.joystick_debug:
            raw = [f'{self.joystick.get_axis(i):+.2f}' for i in range(n_axes)]
            print(f'[joystick axes] {raw}', end='\r')

        vx_lo, vx_hi = sorted((self.vx_lo, self.vx_hi))
        vx_s = -axis(self.args.joystick_vx_axis)   # forward on stick = negative Y in pygame's convention
        vy_s = axis(self.args.joystick_vy_axis)
        yaw_s = -axis(self.args.joystick_yaw_axis)
        if vx_s == 0.0 and vy_s == 0.0 and yaw_s == 0.0 and not self.manual:
            return  # sticks centered and nothing else has claimed manual control yet -- leave the policy's own command alone
        self._enter_manual_mode()
        # vx_s in [-1,1] -> linearly onto [vx_lo, vx_hi] (vx_lo may itself be
        # negative -- a small commanded reverse speed -- so 0 stick deflection
        # is NOT necessarily 0 commanded vx; matches how N/C already treat
        # this checkpoint's own trained window as the ground truth range).
        self.cmd_vx = vx_lo + (vx_s + 1.0) / 2.0 * (vx_hi - vx_lo)
        self.cmd_vy = vy_s * self.vy_max
        self.cmd_yaw = yaw_s * self.yaw_max
        self.manual = True

        if self.joystick.get_numbuttons() > 0 and self.joystick.get_button(0):
            self.state, self.obs = self._reset()
            self.step_count = 0

    def _on_key(self, key: int) -> None:
        with self._keys_lock:
            self._pending_keys.append(key)

    def _apply_cruise_command(self, frac: float | None = None, announce: bool = True) -> None:
        """Straight-line cruise at `frac` of this checkpoint's own vx_hi
        (defaults to --vx-frac, or 0.70 if that was never set -- pressing
        C with no --vx-frac still does something reasonable). vy/yaw
        zeroed so this is a clean forward-only demo command, not a random
        point in the trained window."""
        f = self.vx_frac if frac is None else float(frac)
        if f <= 0.0:
            f = 0.70
        vx_lo, vx_hi = sorted((self.vx_lo, self.vx_hi))
        self.cmd_vx = max(vx_lo, min(vx_hi, f * self.vx_hi))
        self.cmd_vy = 0.0
        self.cmd_yaw = 0.0
        self.manual = True
        if announce:
            print(f'[viewer] Cruise: vx={self.cmd_vx:+.3f} m/s ({100*f:.0f}% of vx_hi={self.vx_hi:.3f} m/s), vy=0, yaw=0')

    def _sample_random_command(self) -> None:
        self.cmd_vx, self.cmd_vy, self.cmd_yaw, mixed = self._sample_trained_random_command()
        self.manual = True
        lo, hi = sorted((self.vx_lo, self.vx_hi))
        kind = 'mixed p-norm' if mixed else 'axis slice'
        norm = self._combo_norm(self.cmd_vx, self.cmd_vy, self.cmd_yaw) if mixed else 0.0
        print(f'[viewer] Random trained command ({kind}, combo radius={self.combo_radius:.3f}): '
              f'vx={self.cmd_vx:+.3f} vy={self.cmd_vy:+.3f} yaw={self.cmd_yaw:+.3f} '
              f'norm={norm:.3f} | window vx∈[{lo:.3f},{hi:.3f}] yaw∈±{self.yaw_max:.3f} vy∈±{self.vy_max:.3f}')

    def _reset(self):
        """Viewer reset with fresh task RNG and a zeroed training timeout clock.

        Training intentionally randomises ``step_i`` across 2048 envs so healthy
        episodes do not all time out on the same PPO update. With a one-env viewer
        that same mechanism creates a bizarre short demo (and the same short demo
        every reset if the same seed is reused). The viewer therefore uses a fresh
        folded-in reset key and explicitly starts the timeout clock at zero. The
        environment itself has a very long internal horizon; autonomous showcase
        modes enforce their 700-step boundary in ``run()``, while Manual sticks
        intentionally has no arbitrary timeout.
        """
        key = self.jax.random.fold_in(self.env.key0, int(self._reset_serial))
        self._reset_serial += 1
        state = self.env.reset_jit(key, self.params_arr)
        state = state._replace(step_i=self.jnp.zeros_like(state.step_i))
        return state, self.env.obs(state)

    def _inject_manual_command(self) -> None:
        """Make a viewer command a real policy input, not just UI state.

        The environment carries both the currently visible task and an internal
        transition target/ramp state. Updating only ``state.task`` is immediately
        overwritten by that ramp state, and the cached observation still contains
        the old command. Keep all of them coherent and refresh the newest history
        frame so the very next action sees the joystick command.

        v5006 combo-trained branches do not have a rectangular simultaneous
        vx/vy/yaw envelope. Manual control therefore projects combined commands
        onto the current learned p-norm frontier before injecting them.
        """
        jnp = self.jnp
        self.cmd_vx, self.cmd_vy, self.cmd_yaw = self._project_to_trained_combo(
            self.cmd_vx, self.cmd_vy, self.cmd_yaw)
        task = self.state.task
        task = task.at[0, self.TASK_VX].set(self.cmd_vx)
        task = task.at[0, self.TASK_VY].set(self.cmd_vy)
        task = task.at[0, self.TASK_YAW].set(self.cmd_yaw)
        cmd = jnp.asarray([[self.cmd_vx, self.cmd_vy, self.cmd_yaw]], dtype=jnp.float32)
        state = self.state._replace(
            task=task,
            cmd_target=cmd,
            cmd_ramp_from=cmd,
            cmd_ramp_frac=jnp.ones_like(self.state.cmd_ramp_frac),
            cmd_ramp_speed=jnp.zeros_like(self.state.cmd_ramp_speed),
            cmd_hold_left=jnp.full_like(self.state.cmd_hold_left, 1_000_000_000),
        )
        base = self.env._base_obs(state.data, task, state.prev_action, state.phase)
        state = state._replace(hist=state.hist.at[:, -1, :].set(base))
        self.state = state
        self.obs = self.env.obs(state)

    def _release_to_policy_task(self, announce: bool = True) -> None:
        # Manual mode deliberately overwrites the transition target/ramp state.
        # A fresh reset is the clean way to hand all command generation back to
        # the environment rather than trying to reconstruct the sampled target.
        self.cmd_vx = self.cmd_vy = self.cmd_yaw = 0.0
        self.manual = False
        self.state, self.obs = self._reset()
        self.step_count = 0
        if announce:
            print('[viewer] Back to the policy\'s own learned task (fresh episode)')

    def _get_action(self):
        jnp = self.jnp
        if self.manual:
            self._inject_manual_command()
        if self.policy_layers is None:
            return jnp.zeros((1, 6), dtype=jnp.float32)
        obs = self.obs
        if self.obs_mean is not None:
            obs = self._training_module.normalize_obs(
                obs, self.obs_mean, self.obs_std, self.obs_norm_mask)
        mean = mlp_apply(self.policy_layers, obs, jnp)
        return jnp.clip(mean, -1.0, 1.0)

    def run(self):
        import mujoco
        import mujoco.viewer
        import numpy as np

        mj_model = self.env.mj_model
        data = mujoco.MjData(mj_model)
        step_dt = float(self.control_dt)
        real_dt = step_dt / max(self.speed, 1e-6)
        policy_txt = 'zero-action sanity check' if self.policy_layers is None else (self.loaded_checkpoint or self.args.checkpoint or self.args.run_dir)

        with mujoco.viewer.launch_passive(mj_model, data, key_callback=self._on_key) as v:
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.2, -15, 135
            print(f'[viewer] Policy: {policy_txt}')
            combo_note = (f' combo p={self.combo_p:.2f} radius={self.combo_radius:.3f}'
                          if self.combo_mix_frac > 0.0 else '')
            print(f'[viewer] Current command window: vx∈[{min(self.vx_lo,self.vx_hi):.3f},'
                  f'{max(self.vx_lo,self.vx_hi):.3f}] yaw∈±{self.yaw_max:.3f} vy∈±{self.vy_max:.3f}'
                  f' (m/s, rad/s){combo_note}')
            print("[viewer] Controls: W/S=vx  A/D=yaw  Up/Dn=vy  N=random trained command  "
                  "C=cruise (vx-frac)  "
                  "R=reset  Space=back to policy's own command  Q=quit")
            if self.joystick is not None:
                print(f"[viewer] Joystick active: {self.joystick.get_name()} -- "
                      f"left stick Y=vx (axis {self.args.joystick_vx_axis}), "
                      f"left stick X=vy (axis {self.args.joystick_vy_axis}), "
                      f"right stick X=yaw (axis {self.args.joystick_yaw_axis}), "
                      f"button 0=reset. Wrong stick doing the wrong thing? rerun with "
                      f"--joystick-debug to see live raw axis numbers, then pass the right "
                      f"ones via --joystick-vx-axis/--joystick-vy-axis/--joystick-yaw-axis.")
            if self.panel is not None:
                print('[viewer] Control panel open in a separate window -- drag the sticks or use '
                      'its buttons; none of it needs this MuJoCo window to have focus.')
            while v.is_running() and not self._quit_requested:
                t0 = time.perf_counter()
                with self._keys_lock:
                    kp, self._pending_keys = self._pending_keys, []
                for key in kp:
                    self._handle_key(key, v)
                self._poll_joystick()
                if self.panel is not None:
                    if not self.panel.pump():
                        self.panel = None  # window was closed directly (not via the Quit button) -- stop pumping it, keep running
                    else:
                        self.panel.set_readout(self.cmd_vx, self.cmd_vy, self.cmd_yaw, self.manual)

                self._update_demo_command()
                action = self._get_action()
                out = self.env.step(self.state, action, self.params_arr)
                self.state, self.obs = out.state, out.obs
                reward, done, metrics = out.reward, out.done, out.metrics
                self.step_count += 1
                if bool(np.asarray(done)[0]):
                    fell = bool(np.asarray(metrics.get('episode_fell', [0.0]))[0] > 0.5)
                    timed_out = bool(np.asarray(metrics.get('episode_timeout', [0.0]))[0] > 0.5)
                    reason = 'fall' if fell else ('timeout' if timed_out else 'termination')
                    if self.control_mode in AUTOPILOT_MODES:
                        print(f'[viewer] Autonomous episode ended: {reason} (viewer step {self.step_count}).')
                        self._reset_showcase_episode(reason)
                    else:
                        print(f'[viewer] Teleop ended by real {reason} (viewer step {self.step_count}) — resetting physical state only.')
                        self.state, self.obs = self._reset()
                        self.step_count = 0
                elif self.control_mode in AUTOPILOT_MODES and self.step_count >= self.demo_episode_steps:
                    # Training/demo modes deliberately remain episodic so they
                    # provide a rolling, comparable 700-step showcase and a safe
                    # checkpoint-refresh boundary. Manual sticks has no such loop.
                    self._reset_showcase_episode(f'{self.demo_episode_steps}-step showcase horizon')

                qpos = np.asarray(self.jax.device_get(self.state.data.qpos))[0]
                qvel = np.asarray(self.jax.device_get(self.state.data.qvel))[0]
                with v.lock():
                    data.qpos[:len(qpos)] = qpos
                    data.qvel[:len(qvel)] = qvel
                    mujoco.mj_kinematics(mj_model, data)
                v.sync()

                remaining = real_dt - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        if self.panel is not None:
            self.panel.destroy()
        print('[viewer] Closed.')

    def _handle_key(self, key: int, v) -> None:
        # Clamped to this checkpoint's actual current command window (see
        # __init__), so manual control can't push the robot to commands
        # the policy never trained on.
        vx_lo, vx_hi = sorted((self.vx_lo, self.vx_hi))
        nudged = True
        if key == GLFW_W:
            self._enter_manual_mode(); self.cmd_vx = min(self.cmd_vx + VX_STEP, vx_hi)
        elif key == GLFW_S:
            self._enter_manual_mode(); self.cmd_vx = max(self.cmd_vx - VX_STEP, vx_lo)
        elif key == GLFW_A:
            self._enter_manual_mode(); self.cmd_yaw = min(self.cmd_yaw + YAW_STEP, self.yaw_max)
        elif key == GLFW_D:
            self._enter_manual_mode(); self.cmd_yaw = max(self.cmd_yaw - YAW_STEP, -self.yaw_max)
        elif key == GLFW_UP:
            self._enter_manual_mode(); self.cmd_vy = min(self.cmd_vy + VY_STEP, self.vy_max)
        elif key == GLFW_DOWN:
            self._enter_manual_mode(); self.cmd_vy = max(self.cmd_vy - VY_STEP, -self.vy_max)
        else:
            nudged = False
        if nudged:
            print(f'[viewer] command: vx={self.cmd_vx:+.3f} vy={self.cmd_vy:+.3f} yaw={self.cmd_yaw:+.3f}')

        if key == GLFW_N:
            self._set_control_mode('Random demo')
        elif key == GLFW_C:
            self._set_control_mode('Cruise 70%')
        elif key == GLFW_SPACE:
            self._set_control_mode('Policy task')
        elif key == GLFW_R:
            self.state, self.obs = self._reset()
            self.step_count = 0
            print('[viewer] Episode reset')
        elif key in (GLFW_Q, GLFW_ESC):
            print('[viewer] Quit requested.')
            v.close()


def _find_run_dir_from_env() -> str:
    for var in ('ROBOT_LAB_RUN_DIR', 'EMU_LAB_RUN_DIR'):
        val = os.environ.get(var, '')
        if val and Path(val).exists():
            return val
    runs_dir = os.environ.get('ROBOT_LAB_RUNS_DIR', '')
    if runs_dir and Path(runs_dir).exists():
        candidates = sorted(
            (d for d in Path(runs_dir).iterdir() if d.is_dir() and (d / 'checkpoints' / 'latest_mjx.pkl').exists()),
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return ''


def main(argv=None):
    ap = argparse.ArgumentParser(description='EMU interactive viewer')
    ap.add_argument('--run-dir', default='', help='external raw rung directory; uses its checkpoints/best_broad_eval.pkl or latest_mjx.pkl')
    ap.add_argument('--checkpoint', default='', help='Explicit checkpoint .pkl path (overrides --run-dir)')
    ap.add_argument('--speed', default=0.5, type=float, help='Playback speed multiplier (default: 0.5)')
    ap.add_argument('--seed', default=42, type=int)
    ap.add_argument('--max-episode-steps', default=700, type=int)
    ap.add_argument('--device', default='cpu', help='JAX device for the viewer itself (default cpu)')
    ap.add_argument('--no-panel', action='store_true',
                     help='Disable the mouse-draggable control panel window (tkinter) -- keyboard '
                          '(needs the MuJoCo window focused) and --joystick (if enabled) still work.')
    # Fraction of THIS CHECKPOINT'S own vx_hi to auto-command as a
    # straight-line cruise at launch (0 disables, the default startup
    # behaviour). Also bound to the C key, which re-applies this fraction
    # at any point (e.g. after N randomizes or R resets); the C key falls
    # back to 0.70 if --vx-frac was never passed.
    ap.add_argument('--vx-frac', default=0.0, type=float,
                     help="Auto-command a forward cruise at launch, as a fraction (0-1+) of this "
                          "checkpoint's OWN vx_hi -- e.g. 0.7 for 70%% of its currently-trained top "
                          "speed. NOT a fraction of VX_HI_MAX (the design ceiling): an early/mid-run "
                          "checkpoint has likely never seen commands anywhere near that. 0 (default) "
                          "disables -- same startup behaviour as before this flag existed.")
    ap.add_argument('--joystick', action='store_true',
                     help='Enable gamepad control (needs pygame: pip install pygame). Additive to '
                          'WASD and the control panel, not a replacement -- all three work at once. '
                          'Left stick=vx/vy, right stick X=yaw, button 0=reset. On WSL this generally '
                          'cannot see a USB controller at all; run natively on Windows Python instead '
                          '(CPU-only, no CUDA/WSL needed for the viewer).')
    ap.add_argument('--joystick-index', default=0, type=int, help='Which connected joystick to use, if more than one')
    ap.add_argument('--joystick-vx-axis', default=1, type=int, help='Axis index for vx (default: left stick Y)')
    ap.add_argument('--joystick-vy-axis', default=0, type=int, help='Axis index for vy (default: left stick X)')
    ap.add_argument('--joystick-yaw-axis', default=3, type=int, help='Axis index for yaw (default: right stick X)')
    ap.add_argument('--joystick-deadzone', default=0.15, type=float, help='Ignore axis movement smaller than this')
    ap.add_argument('--joystick-debug', action='store_true',
                     help='Print every raw axis value live -- use this first if the default axis '
                          'numbers control the wrong thing on your controller/OS.')
    args = ap.parse_args(argv)

    if args.device and args.device.lower() not in ('auto', 'default', ''):
        os.environ.setdefault('JAX_PLATFORM_NAME', args.device.lower())

    if not args.run_dir and not args.checkpoint:
        args.run_dir = _find_run_dir_from_env()
        if args.run_dir:
            print(f'[viewer] Auto-discovered run dir from environment: {args.run_dir}')
        else:
            print(f'[viewer] No run specified; scanning {_DEFAULT_VERSION_RUN_ROOT} for the strongest held-out model...')

    EmuViewer(args).run()


if __name__ == '__main__':
    main()
