"""Generate the *lesson* to practise, separately from the physics engine.

`TaskSampler` defines exactly what the curriculum can ask an environment to
attempt: physical velocity/attitude targets, regime selection, transition
ramps and optional capability-field prompts.

It is a stateless mixin of EmuMJXEnv rather than a second configurable
object: both training and evaluation share the exact same per-world JAX RNG,
limits and JIT compilation. No reset/step physics, reward maths, policy
weights or observation construction lives here.

Reading order:
  1. TASK_FIELDS and T: teacher-only task packet schema
  2. command_phase_rate: desired gait cadence, not observed steps
  3. _sample_task: birth target for each independent world
  4. _sample_cmd_target: later velocity command during TRANSITION
  5. _transition_hold_and_jump / _init_transition_state: change timing

A `task` is an instruction. Success/failure comes from independently
measured physics and must not be inferred from the requested command.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NamedTuple
from .contracts import (
    CONTROL_DT, COMBO_P, PHASE_RATE_MAX_HZ, SEED_VX_HI, SEED_VX_LO,
    FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS, FIELD_NODE_STRIDE,
    FIELD_EDGE_STRIDE, FIELD_NODE_START, FIELD_EDGE_START,
    FIELD_EDGE_SCALES, P,
)

# Command-transition dynamics (see _sample_cmd_target / TRANSITION's own
# difficulty path in _transition_hold_and_jump).
TRANSITION_HOLD_MIN_STEPS = 45.0   # shortest command hold, reached when min_hold_s bottoms out
TRANSITION_HOLD_MAX_STEPS = 700.0  # longest hold, the starting value (~never resamples mid-episode)
TRANSITION_RAMP_MIN_STEPS = 4.0    # near-instant target change, the hardest end
TRANSITION_RAMP_MAX_STEPS = 120.0  # gentle ramp, the starting/easy value
CARRYOVER_MAX_PROB = 0.35          # at TRANSITION's hardest point, up to this fraction of resets carry over qvel
# RESILIENCE-regime episodes get the full current push ceiling -- that's
# the regime actually testing it. STEADY/TRANSITION get a reduced fraction
# instead of the same full level, so ordinary walking practice stays
# representative of normal operation rather than being perturbed as hard
# as a dedicated recovery drill.
REGULAR_RESILIENCE_FRAC = 0.25
# Curriculum ceilings are written in physical units. Each regime's 0..1
# difficulty scalar interpolates between a seed value and these limits.
VX_HI_MAX = 0.45             # m/s -- STEADY: vx_hi walks from SEED_VX_HI up to this
VX_LO_MIN = -0.15            # m/s -- STEADY: vx_lo walks from SEED_VX_LO down to this
YAW_MAX_RAD_S_LIMIT = 1.20   # rad/s -- STEADY: yaw_max walks from 0 up to this
VY_MAX_MPS_LIMIT = 0.18     # m/s -- STEADY: vy_max walks from 0 up to this
HEIGHT_RANGE_MAX_M = 0.018   # m -- POSTURE (global): height_range walks from 0 up to
                              # this; pitch/roll/step-height ranges derive from the
                              # same posture_difficulty, since they're a genuinely
                              # coupled "how much does static pose vary" bundle, not
                              # four independently-meaningful physical limits.
TRANSITION_HOLD_MAX_S = TRANSITION_HOLD_MAX_STEPS * CONTROL_DT  # ~14.0s -- TRANSITION:
                                                                  # easiest end of hold-time
TRANSITION_HOLD_MIN_S = TRANSITION_HOLD_MIN_STEPS * CONTROL_DT  # ~0.9s -- hardest end
# Short reward-settle window after reset. Termination remains active; only
# tracking score avoids over-interpreting a few frames of spawn transient.
# STEADY and RESILIENCE regimes never resample their command --
# cmd_hold_left is initialized to this for them, safely larger than any
# real episode length (max_episode_steps is ~700), so the resample branch
# in _step_impl simply never fires. int32-safe (max ~2.1 billion).
INFINITE_HOLD_STEPS = 1_000_000_000
# Per-environment task-vector schema.  The actor only sees the command/sensor
# subset exposed by _base_obs; regime, birth difficulty and rehearsal are
# teacher-only state.  Resilience is a unitless disturbance fraction and is
# converted to Newtons exactly once in _step_impl.
TASK_FIELDS = (
    "regime", "vx", "vy", "yaw", "height", "pitch", "roll", "step_height",
    "resilience", "phase_rate",
    # Immutable-at-birth diagnostics: score an episode against the difficulty
    # that created it, not the possibly newer governor state when it ends.
    "birth_steady_d", "birth_transition_d", "birth_resilience_d", "birth_posture_d",
    "athletic_rehearsal",  # teacher-only full-envelope rehearsal tag
    "sensor_scale",        # observation corruption scale; never an actor channel
    "field_taught",        # teacher-only provenance; never actor/reward input
)
T = SimpleNamespace(**{name: i for i, name in enumerate(TASK_FIELDS)})
TASK_DIM = len(TASK_FIELDS)


def command_phase_rate(vx, vy, vx_hi_max, phase_speed_gain, base_phase_rate):
    """Sampled gait cadence: base 0.85–1.20 Hz plus birth-speed gain, max 2.6 Hz.

    Keep pitch-command limits out of this calculation. This function also gives
    CPU-only contract tests a way to catch invalid JAX argument signatures
    before a multi-hour MJX GPU compile.
    """
    import jax.numpy as jnp
    speed_fraction = jnp.clip(jnp.sqrt(vx * vx + vy * vy) / jnp.maximum(0.05, vx_hi_max), 0.0, 1.5)
    return jnp.minimum(PHASE_RATE_MAX_HZ, base_phase_rate + phase_speed_gain * speed_fraction)



class CommandFrame(NamedTuple):
    """The next command request, its ramp state, and transition diagnostics."""
    task: Any                 # advanced command for the next observation
    stored_task: Any          # next state, including teacher-only provenance
    ramp_frac: Any
    transition_active: Any
    cmd_resampled: Any
    target_next: Any
    ramp_from_next: Any
    ramp_frac_next: Any
    ramp_speed_next: Any
    hold_left_next: Any


class CommandEnvelope(NamedTuple):
    """Velocity/yaw command limits; full and reduced share the same units."""
    vx_hi: Any
    vx_lo: Any
    vy: Any
    yaw: Any


class TaskSampler:
    """Stateless lesson-generation methods consumed by EmuMJXEnv.

    Requires the existing environment's JAX context, num_envs, physical
    command ceilings, attitude profile and optional capability-field knobs.
    It does not store a second copy of that configuration or model state.
    """

    def _lerp(self, seed, ceiling, difficulty):
        """The one function every regime's difficulty scalar goes through
        to become a real physical range -- plain linear interpolation from
        seed to ceiling (see the module doctrine for why one shared scalar
        with no per-component rate table is enough)."""
        jnp = self.jnp
        return seed + jnp.clip(difficulty, 0.0, 1.0) * (ceiling - seed)

    def _field_node_table(self, params):
        return params[FIELD_NODE_START:FIELD_NODE_START + FIELD_NODE_ANCHORS * FIELD_NODE_STRIDE].reshape((FIELD_NODE_ANCHORS, FIELD_NODE_STRIDE))

    def _field_edge_table(self, params):
        return params[FIELD_EDGE_START:FIELD_EDGE_START + FIELD_EDGE_ANCHORS * FIELD_EDGE_STRIDE].reshape((FIELD_EDGE_ANCHORS, FIELD_EDGE_STRIDE))

    def _weighted_anchor_index(self, key, weights):
        jax, jnp = self.jax, self.jnp
        w = jnp.maximum(weights, 0.0)
        total = jnp.sum(w)
        u = jax.random.uniform(key, ()) * jnp.maximum(total, 1e-6)
        idx = jnp.sum(u > jnp.cumsum(w)).astype(jnp.int32)
        return jnp.clip(idx, 0, w.shape[0]-1), total

    def _command_envelopes(self, params) -> tuple[CommandEnvelope, CommandEnvelope]:
        """One source for birth and transition target bounds.

        A physical frontier, when enabled, wins over the 0–1 steady
        difficulty interpolation. The reduced envelope keeps the original
        regime_command_frac lerp back toward the two seed velocity bounds.
        This is deterministic JAX maths: NO extra random key is consumed.
        """
        jnp = self.jnp
        steady_d = params[P.steady_difficulty]
        command_frac = jnp.clip(params[P.regime_command_frac], 0.0, 1.0)
        if self.open_frontier:
            vx_hi = jnp.maximum(SEED_VX_HI, params[P.frontier_vx_hi])
            vx_lo = jnp.minimum(SEED_VX_LO, params[P.frontier_vx_lo])
            yaw_max = jnp.maximum(0.0, params[P.frontier_yaw])
            vy_max = jnp.maximum(0.0, params[P.frontier_vy])
        else:
            vx_hi = self._lerp(SEED_VX_HI, self.vx_hi_max, steady_d)
            vx_lo = self._lerp(SEED_VX_LO, self.vx_lo_min, steady_d)
            yaw_max = self._lerp(0.0, self.yaw_max_rad_s, steady_d)
            vy_max = self._lerp(0.0, self.vy_max_mps, steady_d)
        full = CommandEnvelope(vx_hi, vx_lo, vy_max, yaw_max)
        reduced = CommandEnvelope(
            SEED_VX_HI + command_frac * (vx_hi - SEED_VX_HI),
            SEED_VX_LO + command_frac * (vx_lo - SEED_VX_LO),
            command_frac * vy_max, command_frac * yaw_max,
        )
        return full, reduced

    def _sample_task(self, keys, params):
        """Draws each env's regime (STEADY/TRANSITION/RESILIENCE) and that
        regime's command envelope, built from one difficulty scalar via
        _lerp. TRANSITION/RESILIENCE spawn at a fraction of STEADY's
        current envelope (regime_command_frac) rather than a range of
        their own. Initial-state noise (pose tilt, position, angular
        velocity) for RESILIENCE is NOT sampled here: those are one-time
        reset perturbations, not an ongoing target that needs to persist
        for the whole episode the way vx/vy/yaw/posture do, so they're
        computed directly in _reset_with_keys from resilience_difficulty
        and this task's own regime column instead of bloating TASK_DIM
        with fields that are only ever read once.
        """
        jax, jnp = self.jax, self.jnp
        steady_d = params[P.steady_difficulty]
        posture_d = params[P.posture_difficulty]
        resilience_d = params[P.resilience_difficulty]
        # The SAME physical limits now govern birth and later transitions.
        full, reduced = self._command_envelopes(params)
        vx_hi, vx_lo, vy_max, yaw_max = full
        vx_hi_reduced, vx_lo_reduced, vy_max_reduced, yaw_max_reduced = reduced

        # POSTURE (global, every regime): height/pitch/roll/step-height are
        # a coupled bundle, all driven by the one posture_difficulty
        # fraction directly.
        height_range = self._lerp(0.0, HEIGHT_RANGE_MAX_M, posture_d)
        pitch_range = self.pitch_limit_rad * posture_d
        roll_range = 0.055 * posture_d
        step_height_range = 0.030 * jnp.maximum(0.05, posture_d)

        # RESILIENCE difficulty remains unitless here. Non-resilience
        # regimes still get a small AMBIENT fraction of it
        # (REGULAR_RESILIENCE_FRAC), and _step_impl converts that fraction
        # to Newtons exactly once at the force-application site.

        regime_probs = jnp.asarray([params[P.regime_frac_steady], params[P.regime_frac_transition],
                                    params[P.regime_frac_resilience]], dtype=jnp.float32)
        regime_probs = jnp.maximum(regime_probs, 0.0)
        regime_probs = regime_probs / jnp.maximum(jnp.sum(regime_probs), 1e-6)
        regime_cdf = jnp.cumsum(regime_probs)

        def one(k):
            # Independent draws, one purpose each -- no key is reused
            # for two different quantities. A reused key here isn't just
            # "correlated" output, it can be an exact bit-for-bit collision
            # depending on how two different split-counts happen to overlap,
            # which is why every draw below gets its own key rather than
            # sharing one across quantities.
            # Keep split arity stable so deleting a dead task feature does not
            # reshuffle unrelated stochastic streams under the same seed.
            k_regime, k_vx, k_vy, k_yaw, k_height, k_pitch, k_roll, k_step, k_phase, _reserved, k_athletic, k_combo, k_axis, k_radius, k_field_use, k_field_pick = jax.random.split(k, 16)
            u = jax.random.uniform(k_regime, ())
            regime = jnp.sum(u > regime_cdf).astype(jnp.float32)
            # Full-envelope rehearsal preserves the proven athletic distribution
            # while the adaptive frontier explores beyond it.
            is_athletic = jax.random.uniform(k_athletic, ()) < jnp.clip(params[P.athletic_rehearsal_frac], 0.0, 0.5)
            # Rehearsal is STEADY so it never contaminates transition/push tests.
            regime = jnp.where(is_athletic, 0.0, regime)
            is_steady = regime < 0.5
            is_transition = jnp.logical_and(regime > 0.5, regime < 1.5)
            is_narrow = jnp.logical_not(is_steady)  # TRANSITION or RESILIENCE: both use the reduced envelope

            use_vx_hi = jnp.where(is_narrow, vx_hi_reduced, vx_hi)
            use_vx_lo = jnp.where(is_narrow, vx_lo_reduced, vx_lo)
            use_vy = jnp.where(is_narrow, vy_max_reduced, vy_max)
            use_yaw = jnp.where(is_narrow, yaw_max_reduced, yaw_max)
            vx = jax.random.uniform(k_vx, (), minval=use_vx_lo, maxval=use_vx_hi)
            vy = jax.random.uniform(k_vy, (), minval=-use_vy, maxval=use_vy)
            yaw = jax.random.uniform(k_yaw, (), minval=-use_yaw, maxval=use_yaw)

            # Command-combination curriculum: a configurable fraction of
            # samples are true mixed-axis commands constrained by a p-norm
            # blob; the remainder are axis-isolation samples so each boundary
            # remains measurable. Mixed samples sometimes probe just beyond the
            # current blob radius, which is how local shape expansion gets data.
            if self.combo_mix_frac > 0.0:
                combo = jax.random.uniform(k_combo, ()) < self.combo_mix_frac
                axis = jax.random.randint(k_axis, (), 0, 3)
                vx_scale = jnp.where(vx >= 0.0, jnp.maximum(use_vx_hi, 1e-4), jnp.maximum(-use_vx_lo, 1e-4))
                nx = vx / vx_scale
                ny = vy / jnp.maximum(use_vy, 1e-4)
                nz = yaw / jnp.maximum(use_yaw, 1e-4)
                norm = jnp.power(jnp.abs(nx)**COMBO_P + jnp.abs(ny)**COMBO_P + jnp.abs(nz)**COMBO_P, 1.0/COMBO_P)
                combo_radius = jnp.where(self.open_frontier, jnp.maximum(0.10, params[P.frontier_combo_radius]), 1.0)
                # Mixed samples cover the blob interior and a thin just-outside
                # probe band instead of living only on the mathematical boundary.
                radial = 0.35 + (0.65 + self.combo_probe_frac) * jax.random.uniform(k_radius, ())
                probe = combo_radius * radial
                scale = jnp.where(norm > 1e-6, probe / jnp.maximum(norm, 1e-6), 1.0)
                vx_m, vy_m, yaw_m = vx*scale, vy*scale, yaw*scale
                vx_a = jnp.where(axis == 0, vx, 0.0)
                vy_a = jnp.where(axis == 1, vy, 0.0)
                yaw_a = jnp.where(axis == 2, yaw, 0.0)
                # Preserve the seed-band forward motion when isolating vy/yaw;
                # this unstable robot is not trained to stand statically.
                vx_a = jnp.where(axis == 0, vx_a, jnp.clip(0.5*(SEED_VX_LO+SEED_VX_HI), use_vx_lo, use_vx_hi))
                vx = jnp.where(combo, vx_m, vx_a)
                vy = jnp.where(combo, vy_m, vy_a)
                yaw = jnp.where(combo, yaw_m, yaw_a)
            # A bounded fraction of ordinary births
            # may be drawn from the capability field's current teaching packet.
            # Anchors are clipped to the governor's current envelope; the field
            # chooses *where* to spend practice while the governor still owns
            # global safety/frontier expansion.
            node_table = self._field_node_table(params)
            ni, node_total = self._weighted_anchor_index(k_field_pick, node_table[:, 3])
            node_cmd = node_table[ni, :3]
            node_recovery = jnp.clip(node_table[ni, 4], 0.0, 1.0)
            field_use = jnp.logical_and(
                jax.random.uniform(k_field_use, ()) < jnp.clip(params[P.field_node_frac], 0.0, 0.95),
                node_total > 1e-6)
            field_use = jnp.logical_and(field_use, jnp.logical_not(is_athletic))
            nvx = jnp.clip(node_cmd[0], use_vx_lo, use_vx_hi)
            nvy = jnp.clip(node_cmd[1], -use_vy, use_vy)
            nyaw = jnp.clip(node_cmd[2], -use_yaw, use_yaw)
            vx = jnp.where(field_use, nvx, vx)
            vy = jnp.where(field_use, nvy, vy)
            yaw = jnp.where(field_use, nyaw, yaw)
            height = jax.random.uniform(k_height, (), minval=-height_range, maxval=height_range)
            pitch = jax.random.uniform(k_pitch, (), minval=-pitch_range, maxval=pitch_range)
            roll = jax.random.uniform(k_roll, (), minval=-roll_range, maxval=roll_range)
            step_height = 0.012 + jax.random.uniform(k_step, (), minval=0.0, maxval=step_height_range)
            base_phase_rate = jax.random.uniform(k_phase, (), minval=0.85, maxval=1.20)
            phase_rate = command_phase_rate(vx, vy, self.vx_hi_max,
                                            self.phase_speed_gain, base_phase_rate)
            # Keep disturbance intensity unitless in the task vector. Convert it
            # to Newtons only where the external force is applied.
            # The field may choose a locally useful recovery severity for a
            # resilience birth, but never above the governor's globally-earned
            # resilience difficulty.  This turns the recovery layer into an
            # actual curriculum signal while preserving the governor as the
            # physical safety ceiling.
            if self.field_teach_recovery:
                birth_resilience_d = jnp.where(
                    jnp.logical_and(field_use, regime > 1.5),
                    jnp.minimum(node_recovery, jnp.clip(resilience_d, 0.0, 1.0)),
                    jnp.clip(resilience_d, 0.0, 1.0))
            else:
                birth_resilience_d = jnp.clip(resilience_d, 0.0, 1.0)
            if self.open_frontier:
                resilience_frac = jnp.clip(params[P.frontier_push_n] / jnp.maximum(self.push_max_force_n, 1e-6), 0.0, 1.0)
            else:
                resilience_frac = birth_resilience_d
            resilience_task = jnp.where(regime > 1.5, jnp.maximum(resilience_frac, 0.08 / jnp.maximum(self.push_max_force_n, 1e-6)),
                                        resilience_frac * REGULAR_RESILIENCE_FRAC)
            sensor_scale_task = jnp.maximum(0.0, params[P.frontier_sensor_scale])
            # Full athletic rehearsal samples the proven control
            # envelope directly. The three axes are intentionally independent:
            # this is memory rehearsal of the old rectangular distribution, not
            # evidence for the new p-norm frontier.
            vx_ath = jax.random.uniform(k_vx, (), minval=-0.20, maxval=0.60)
            vy_ath = jax.random.uniform(k_vy, (), minval=-0.22, maxval=0.22)
            yaw_ath = jax.random.uniform(k_yaw, (), minval=-1.50, maxval=1.50)
            vx = jnp.where(is_athletic, vx_ath, vx)
            vy = jnp.where(is_athletic, vy_ath, vy)
            yaw = jnp.where(is_athletic, yaw_ath, yaw)
            height = jnp.where(is_athletic, 0.0, height)
            pitch = jnp.where(is_athletic, 0.0, pitch)
            roll = jnp.where(is_athletic, 0.0, roll)
            step_height = jnp.where(is_athletic, 0.012, step_height)
            resilience_task = jnp.where(is_athletic, 0.0, resilience_task)

            # Capability-atlas command probes use exact physical commands,
            # bypassing random envelope sampling without changing the actor.
            vx = jnp.where(jnp.isfinite(params[P.probe_cmd_vx]), params[P.probe_cmd_vx], vx)
            vy = jnp.where(jnp.isfinite(params[P.probe_cmd_vy]), params[P.probe_cmd_vy], vy)
            yaw = jnp.where(jnp.isfinite(params[P.probe_cmd_yaw]), params[P.probe_cmd_yaw], yaw)
            cols = [regime, vx, vy, yaw, height, pitch, roll, step_height, resilience_task, phase_rate,
                    steady_d, params[P.transition_difficulty], birth_resilience_d, posture_d,
                    is_athletic.astype(jnp.float32), sensor_scale_task, field_use.astype(jnp.float32)]
            assert len(cols) == TASK_DIM, f"task vector has {len(cols)} columns, TASK_DIM says {TASK_DIM}"
            return jnp.asarray(cols, dtype=jnp.float32)
        return jax.vmap(one)(keys)

    def _sample_cmd_target(self, keys, params, current_cmd=None):
        """Draws a fresh (vx, vy, yaw) TRANSITION-regime target.
        Only ever called for envs currently in TRANSITION (see _step_impl's
        resample gate) -- STEADY and RESILIENCE hold their one spawn target
        for the whole episode, so there's nothing to redraw for them. The
        envelope is scaled TWICE: once by regime_command_frac (transition
        never fights steady's own full command difficulty at the same
        time -- see module doctrine), then again by jump_frac, which is 0
        early in transition_difficulty's own range and 1 from its midpoint
        on -- see _transition_hold_and_jump for why jump size and hold time
        are two separately-progressing halves of ONE scalar, not one axis
        moving both at once.
        """
        jax, jnp = self.jax, self.jnp
        _, jump_frac = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            jump_frac = jnp.asarray(1.0, dtype=jnp.float32)
        _, reduced = self._command_envelopes(params)
        vx_hi, vx_lo, vy_max, yaw_max = reduced
        vx_hi_j = SEED_VX_HI + jump_frac * (vx_hi - SEED_VX_HI)
        vx_lo_j = SEED_VX_LO + jump_frac * (vx_lo - SEED_VX_LO)
        yaw_max_j = jump_frac * yaw_max
        vy_max_j = jump_frac * vy_max

        node_table = self._field_node_table(params)
        edge_table = self._field_edge_table(params)
        if current_cmd is None:
            current_cmd = jnp.zeros((self.num_envs, 3), dtype=jnp.float32)

        def one(k, cur):
            k1, k2, k3, k_node_use, k_node_pick, k_edge_use, k_edge_pick = jax.random.split(k, 7)
            vx = jax.random.uniform(k1, (), minval=vx_lo_j, maxval=vx_hi_j)
            vy = jax.random.uniform(k2, (), minval=-vy_max_j, maxval=vy_max_j)
            yaw = jax.random.uniform(k3, (), minval=-yaw_max_j, maxval=yaw_max_j)
            base = jnp.stack([vx, vy, yaw])

            ni, node_total = self._weighted_anchor_index(k_node_pick, node_table[:, 3])
            node_target = node_table[ni, :3]
            use_node = jnp.logical_and(jax.random.uniform(k_node_use, ()) < jnp.clip(params[P.field_node_frac], 0.0, 0.95), node_total > 1e-6)

            # Graph anchors are local source->target maneuvers.  Prefer anchors
            # whose source is near this env's current command, without adding a
            # high-dimensional table to the actor or policy network.
            scale = jnp.asarray(FIELD_EDGE_SCALES, dtype=jnp.float32)
            dist2 = jnp.sum(((edge_table[:, :3] - cur[None, :]) / scale[None, :]) ** 2, axis=1)
            local_w = jnp.maximum(edge_table[:, 6], 0.0) * jnp.exp(-0.5 * dist2 / (0.45**2))
            ei, edge_total = self._weighted_anchor_index(k_edge_pick, local_w)
            edge_target = edge_table[ei, 3:6]
            use_edge = jnp.logical_and(jax.random.uniform(k_edge_use, ()) < jnp.clip(params[P.field_edge_frac], 0.0, 0.95), edge_total > 1e-6)

            target = jnp.where(use_node, node_target, base)
            target = jnp.where(use_edge, edge_target, target)  # graph has priority
            target = target.at[0].set(jnp.clip(target[0], vx_lo_j, vx_hi_j))
            target = target.at[1].set(jnp.clip(target[1], -vy_max_j, vy_max_j))
            target = target.at[2].set(jnp.clip(target[2], -yaw_max_j, yaw_max_j))
            target = target.at[0].set(jnp.where(jnp.isfinite(params[P.probe_cmd_vx]), params[P.probe_cmd_vx], target[0]))
            target = target.at[1].set(jnp.where(jnp.isfinite(params[P.probe_cmd_vy]), params[P.probe_cmd_vy], target[1]))
            target = target.at[2].set(jnp.where(jnp.isfinite(params[P.probe_cmd_yaw]), params[P.probe_cmd_yaw], target[2]))
            field_taught = jnp.logical_or(use_node, use_edge).astype(jnp.float32)
            return target, field_taught
        return jax.vmap(one)(keys, current_cmd)

    def _transition_hold_and_jump(self, transition_difficulty):
        """Map one transition difficulty scalar onto jump size and frequency.

        Difficulty 0..0.5 increases command-jump magnitude while keeping a
        generous hold time. Difficulty 0.5..1.0 keeps the jump range at maximum
        and shortens the hold time. This preserves one curriculum decision while
        avoiding simultaneous movement of both transition dimensions.
        """
        jnp = self.jnp
        d = jnp.clip(transition_difficulty, 0.0, 1.0)
        jump_frac = jnp.clip(d / 0.5, 0.0, 1.0)
        hold_frac = jnp.clip((d - 0.5) / 0.5, 0.0, 1.0)
        min_hold_s = TRANSITION_HOLD_MAX_S - hold_frac * (TRANSITION_HOLD_MAX_S - TRANSITION_HOLD_MIN_S)
        return min_hold_s, jump_frac

    def _init_transition_state(self, task, params, hold_keys=None):
        """(cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed,
        cmd_hold_left) for a freshly (re)sampled task. frac=1.0 means
        "already arrived" -- task's own VX/VY/YAW is exactly the value to
        start at, no ramp-in needed. Only TRANSITION-regime envs ever get a
        finite hold; STEADY and RESILIENCE get a hold effectively longer
        than any episode (INFINITE_HOLD_STEPS), because neither regime
        resamples its command mid-episode -- see the module doctrine for
        why RESILIENCE holds one target throughout rather than also
        transitioning. `hold_keys` is None only for the one-time env-
        creation reset in _reset_impl, where per-env draw diversity doesn't
        matter (every real episode reset goes through _reset_with_keys
        instead, which always passes real keys).
        """
        jax, jnp = self.jax, self.jnp
        # v5015 adds a continuously changing chassis-pitch target to the
        # existing task ramps. The learned field remains 3D (vx,vy,yaw).
        columns = ([T.vx, T.vy, T.yaw, T.pitch] if self.attitude_profile == 'v5015'
                   else [T.vx, T.vy, T.yaw])
        cmd_target = task[:, columns]
        cmd_ramp_from = cmd_target
        cmd_ramp_frac = jnp.ones((self.num_envs,), dtype=jnp.float32)
        cmd_ramp_speed = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        is_transition = jnp.logical_and(task[:, T.regime] > 0.5, task[:, T.regime] < 1.5)
        min_hold_s, _ = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            min_hold_s = jnp.maximum(CONTROL_DT, params[P.frontier_transition_hold_s])
        hold_min = min_hold_s / CONTROL_DT  # steps
        if hold_keys is None:
            transition_hold = jnp.full((self.num_envs,), hold_min * 0.8, dtype=jnp.int32)
        else:
            hold_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(hold_keys)
            transition_hold = jnp.round(hold_min * (0.6 + 0.4 * hold_draw)).astype(jnp.int32)
        cmd_hold_left = jnp.where(is_transition, transition_hold, INFINITE_HOLD_STEPS)
        return cmd_target, cmd_ramp_from, cmd_ramp_frac, cmd_ramp_speed, cmd_hold_left

    def _advance_commands(self, state, params, resample_keys) -> CommandFrame:
        """Advance the lesson and prepare the command for the NEXT actor step.

        The caller must score this frame against state.task, not the result
        returned here; the actor cannot react to information it hasn't seen.
        """
        jax, jnp = self.jax, self.jnp
        # Command-transition dynamics: ramps the *tracked* command
        # (task[:, VX/VY/YAW], which every other function below reads)
        # smoothly toward an occasionally-resampled target instead of
        # Command hold/ramp timing tightens with transition difficulty. The
        # A transition is defined by the PRE-advance ramp state.  This is
        # crucial at the digital limit: when ramp_speed==1 a newly-resampled
        # command advances 0->1 in one control frame, so testing ramp_frac<1
        # only *after* the advance made the sharpest maneuver invisible.
        transition_active = state.cmd_ramp_frac < 1.0
        cmd_resampled = transition_active & (state.cmd_ramp_frac <= 1e-6) & (state.cmd_ramp_speed > 0.0)
        ramp_frac = jnp.clip(state.cmd_ramp_frac + state.cmd_ramp_speed, 0.0, 1.0)
        cmd_now = state.cmd_ramp_from + ramp_frac[:, None] * (state.cmd_target - state.cmd_ramp_from)
        task = state.task.at[:, T.vx].set(cmd_now[:, 0]).at[:, T.vy].set(cmd_now[:, 1]).at[:, T.yaw].set(cmd_now[:, 2])
        if self.attitude_profile == 'v5015':
            task = task.at[:, T.pitch].set(cmd_now[:, 3])

        hold_left = state.cmd_hold_left - 1
        should_resample = hold_left <= 0
        rk_target, rk_hold, rk_ramp = self._split_batch(resample_keys, 3)
        # Only TRANSITION-regime envs ever have a finite cmd_hold_left (see
        # _init_transition_state) -- STEADY/RESILIENCE's INFINITE_HOLD_STEPS
        # means should_resample is always False for them, so new_target/
        # new_hold below are computed for every env (vmap has no branch)
        # but only ever actually USED where should_resample is True, i.e.
        # only ever for TRANSITION envs.
        new_target, new_field_taught = self._sample_cmd_target(rk_target, params, cmd_now[:, :3])
        if self.attitude_profile == 'v5015':
            pitch_keys = jax.vmap(lambda k: jax.random.fold_in(k, 5015))(rk_target)
            pitch_range = self.pitch_limit_rad * jnp.clip(task[:, T.birth_posture_d], 0.0, 1.0)
            pitch_draws = jax.vmap(lambda k: jax.random.uniform(k, (), minval=-1.0, maxval=1.0))(pitch_keys)
            new_target = jnp.concatenate((new_target, (pitch_draws * pitch_range)[:, None]), axis=1)
        hold_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(rk_hold)
        ramp_draw = jax.vmap(lambda k: jax.random.uniform(k, ()))(rk_ramp)
        min_hold_s, _ = self._transition_hold_and_jump(params[P.transition_difficulty])
        if self.open_frontier:
            min_hold_s = jnp.maximum(CONTROL_DT, params[P.frontier_transition_hold_s])
        hold_min = min_hold_s / CONTROL_DT  # steps
        new_hold = jnp.maximum(1.0, jnp.round(hold_min * (0.6 + 0.4 * hold_draw))).astype(jnp.int32)
        if self.open_frontier:
            # Explicit acceleration frontier: ramp duration is a real count of
            # 20ms control frames and may shrink toward one frame, the natural
            # digital limit. No arbitrary locomotion-speed ceiling is involved.
            ramp_min = jnp.maximum(1.0, params[P.frontier_ramp_steps])
            new_ramp_steps = jnp.maximum(1.0, jnp.round(ramp_min * (0.75 + 0.50 * ramp_draw)))
        else:
            trans_frac = (TRANSITION_HOLD_MAX_S - min_hold_s) / (TRANSITION_HOLD_MAX_S - TRANSITION_HOLD_MIN_S)
            ramp_min = TRANSITION_RAMP_MIN_STEPS + (1.0 - trans_frac) * (TRANSITION_RAMP_MAX_STEPS - TRANSITION_RAMP_MIN_STEPS)
            new_ramp_steps = jnp.maximum(1.0, jnp.round(ramp_min * (0.5 + 0.5 * ramp_draw)))
        new_ramp_speed = 1.0 / new_ramp_steps

        cmd_target_next = jnp.where(should_resample[:, None], new_target, state.cmd_target)
        cmd_ramp_from_next = jnp.where(should_resample[:, None], cmd_now, state.cmd_ramp_from)
        cmd_ramp_frac_next = jnp.where(should_resample, 0.0, ramp_frac)
        cmd_ramp_speed_next = jnp.where(should_resample, new_ramp_speed, state.cmd_ramp_speed)
        cmd_hold_left_next = jnp.where(should_resample, new_hold, hold_left).astype(jnp.int32)
        # Provenance follows the newly-selected transition target from the next
        # state onward.  It never enters actor observation or reward.
        task_for_state = task.at[:, T.field_taught].set(
            jnp.where(should_resample, new_field_taught, task[:, T.field_taught]))

        return CommandFrame(
            task=task, stored_task=task_for_state,
            ramp_frac=ramp_frac, transition_active=transition_active,
            cmd_resampled=cmd_resampled, target_next=cmd_target_next,
            ramp_from_next=cmd_ramp_from_next,
            ramp_frac_next=cmd_ramp_frac_next,
            ramp_speed_next=cmd_ramp_speed_next,
            hold_left_next=cmd_hold_left_next,
        )

