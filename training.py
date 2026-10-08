#!/usr/bin/env python3
"""EMU PPO trainer, curriculum governor, checkpointing and evaluation.

This file intentionally keeps the complete learning loop together: a small
from-scratch actor/critic PPO implementation, observation normalisation, the
RegimeGovernor curriculum, checkpoint/recovery semantics, fixed held-out
evaluation, logging and CLI entry points. environment.py owns the task/physics
side and robot.xml owns the MuJoCo model.

EMU is trained as a dynamically unstable biped. The seed behaviour is a moving
gait; standing motionless is not a prerequisite or primary target.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from environment import (ACTION_DIM, OBS_DIM, PROGRAM_VERSION, ROBOT_VERSION, ENV_VERSION, VERSION, EmuMJXEnv, CONTROL_DT,
                         SEED_VX_LO, SEED_VX_HI, VX_HI_MAX, VX_LO_MIN, YAW_MAX_RAD_S_LIMIT, VY_MAX_MPS_LIMIT,
                         HEIGHT_RANGE_MAX_M, PUSH_MAX_FORCE_N, TRANSITION_HOLD_MAX_S, TRANSITION_HOLD_MIN_S, TRANSITION_RAMP_MAX_STEPS,
                         BASE_OBS_DIM, HISTORY_LEN)

LOG_PREFIX = f'[{ROBOT_VERSION}]'

def source_provenance_from_env() -> dict[str, Any]:
    """Source identity supplied by run_ladder.py.

    Direct training.py runs still work; unknown Git fields are recorded as
    empty/unknown rather than guessed. Checkpoints carry this metadata too.
    """
    dirty_raw = os.environ.get('EMU_GIT_DIRTY', '')
    dirty = dirty_raw.lower() in ('1', 'true', 'yes', 'dirty') if dirty_raw else None
    return {
        'version': os.environ.get('EMU_SOURCE_VERSION', VERSION),
        'git_commit': os.environ.get('EMU_GIT_COMMIT', ''),
        'git_branch': os.environ.get('EMU_GIT_BRANCH', ''),
        'git_describe': os.environ.get('EMU_GIT_DESCRIBE', ''),
        'git_dirty': dirty,
    }


# ---------------------------------------------------------------------------
# Observation normalization.
#
# Observations mix quantities with very different physical scales. Running
# mean/variance statistics therefore live beside PPOState and are checkpointed
# explicitly. Keeping them outside PPOState makes the JIT/state plumbing simpler
# and lets recovery restore the representation independently of network weights.

def update_running_stats(mean, var, count, batch):
    """Chan et al. parallel variance algorithm — a batch update of a running
    mean/var estimate. Standard, well-documented, numerically stable."""
    import jax.numpy as jnp
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
    new_var = m2 / tot_count
    return new_mean, new_var, tot_count


def normalize_obs(obs, obs_mean, obs_std, obs_norm_mask=None):
    import jax.numpy as jnp
    z = jnp.clip((obs - obs_mean) / obs_std, -10.0, 10.0)
    if obs_norm_mask is None:
        return z
    # Keep bounded/categorical channels in their natural coordinates when
    # requested. This prevents curriculum-driven frequency changes (notably
    # regime one-hots) from moving the policy's coordinate system.
    return jnp.where(obs_norm_mask > 0.5, z, obs)


def build_obs_norm_mask(mode: str):
    """Build the observation-normalisation mask for all history frames.

    ``all`` normalises every channel. ``continuous`` normalises physical-scale
    quantities while leaving naturally bounded/categorical channels raw.
    """
    import numpy as _np
    if mode == 'all':
        return _np.ones((OBS_DIM,), dtype=_np.float32)
    if mode != 'continuous':
        raise ValueError(f'unknown obs norm mode {mode!r}')
    base = _np.zeros((BASE_OBS_DIM,), dtype=_np.float32)
    # v5007 deployable packet: normalise physical-scale channels, leave
    # naturally bounded gravity/action/confidence/phase/load channels raw.
    base[3:18] = 1.0    # gyro, q, qv estimate
    base[24:44] = 1.0   # actual ctrl, foot FK, vxy estimate, ToF height, commands
    base[46:53] = 1.0   # shin flex / rate (loads at 53:55 remain bounded raw)
    return _np.tile(base, HISTORY_LEN)

# ---------------------------------------------------------------------------
# PPO implementation kept in this file to avoid helper-module sprawl.
# ---------------------------------------------------------------------------

class PPOState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: Any
    obs: Any
    key: Any


def make_trainer(env, *, seed: int, lr: float, rollout: int, epochs: int, minibatches: int,
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 entropy_cost: float = 0.002, value_cost: float = 0.5, max_grad_norm: float = 1.0,
                 hidden_sizes: tuple[int, ...] = (256, 256), logp_on_raw_action: bool = True,
                 obs_norm_mode: str = 'all'):
    import jax
    import jax.numpy as jnp
    import optax

    obs_norm_mask = jnp.asarray(build_obs_norm_mask(obs_norm_mode), dtype=jnp.float32)

    _tmp_state, _tmp_obs = env.reset({})
    obs_dim = int(_tmp_obs.shape[-1])
    action_dim = 6
    rollout = int(rollout)
    epochs = int(epochs)
    minibatches = int(max(1, minibatches))
    batch_size = rollout * env.num_envs
    if batch_size % minibatches != 0:
        # Keep the implementation simple/predictable. The CLI chooses sane defaults.
        raise ValueError(f'rollout*envs={batch_size} must be divisible by minibatches={minibatches}')
    mb_size = batch_size // minibatches

    def init_layer(k, fan_in, fan_out, scale=1.0):
        w = jax.random.normal(k, (fan_in, fan_out), dtype=jnp.float32) * (scale / jnp.sqrt(float(fan_in)))
        b = jnp.zeros((fan_out,), dtype=jnp.float32)
        return {'w': w, 'b': b}

    def init_mlp(k, out_dim, out_scale=1.0):
        sizes = (int(obs_dim),) + tuple(hidden_sizes) + (int(out_dim),)
        keys = jax.random.split(k, len(sizes)-1)
        layers = []
        for i, (ki, fan_in, fan_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
            scale = out_scale if i == len(sizes)-2 else np.sqrt(2.0)
            layers.append(init_layer(ki, fan_in, fan_out, scale=scale))
        return layers

    def init_params(k):
        k1,k2,k3 = jax.random.split(k, 3)
        return {
            'pi': init_mlp(k1, action_dim, out_scale=0.01),
            'vf': init_mlp(k2, 1, out_scale=1.0),
            'log_std': jnp.full((action_dim,), -1.55, dtype=jnp.float32),
        }

    def mlp(layers, x):
        for layer in layers[:-1]:
            x = jnp.tanh(x @ layer['w'] + layer['b'])
        return x @ layers[-1]['w'] + layers[-1]['b']

    def apply(params, obs):
        mean = mlp(params['pi'], obs)
        value = jnp.squeeze(mlp(params['vf'], obs), axis=-1)
        return mean, value

    # Exploration-std floor/ceiling, applied in the forward pass (so the
    # stored parameter can wander but the *behavioural* std can't leave
    # this band). Two reasons, one defensive and one mechanical.
    #   Defensive: `log_std` is a plain trainable parameter with no
    #   stop_gradient, no annealing and only entropy_cost=0.002 pushing back.
    #   A real run drifted from sigma=0.212 at birth to 0.126 by 50M steps
    #   and was still trending down; there is nothing in the code that
    #   would stop it reaching ~0 given enough steps.
    #   Mechanical: KL between two Gaussians with the same std scales as
    #   (delta_mu)^2 / (2*sigma^2). Every factor-of-2 shrink in sigma makes
    #   the SAME parameter step cost 4x the KL, which can turn a KL guard
    #   that fires occasionally early in a run into one that fires on
    #   every update later, even with nothing else about the configuration
    #   changing. A floor keeps the trust region a fixed-ish size in
    #   behaviour space instead of quietly collapsing.
    # 0.10 is chosen as insurance, not as a tuned value: comfortably below
    # where exploration std has been observed to settle on a healthy run,
    # so it should never bind in practice and only engages if std really
    # is running away to zero.
    LOG_STD_MIN = float(np.log(0.10))
    LOG_STD_MAX = float(np.log(0.60))

    def eff_log_std(params):
        return jnp.clip(params['log_std'], LOG_STD_MIN, LOG_STD_MAX)

    def log_prob(params, mean, action):
        log_std = eff_log_std(params)
        std = jnp.exp(log_std)
        z = (action - mean) / std
        return -0.5 * jnp.sum(z*z + 2.0*log_std + jnp.log(2.0*jnp.pi), axis=-1)

    def entropy(params):
        log_std = eff_log_std(params)
        return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)

    # clip_by_global_norm alone does NOT protect against a NaN gradient --
    # if any element is NaN, the global norm is NaN too, so the clip's
    # scale factor is NaN and it passes the NaN straight through. That is
    # exactly how one bad transition (e.g. an undetected physics
    # divergence -- see environment.py's physics_diverged guard) can turn
    # into a permanently-poisoned network: the optimizer writes that NaN
    # gradient into every weight, forever, with nothing to catch it.
    # apply_if_finite wraps the whole chain: a non-finite update is skipped
    # (params/opt_state held at their last good value) instead of applied,
    # and only forces the update through after max_consecutive_errors
    # skips in a row, so a real, sustained problem still surfaces instead
    # of silently stalling forever.
    # SEPARATE clipping budgets for policy vs value parameters. This
    # matters because right after a sudden reward-regime shift (e.g. the
    # moment of a collapse), the value function's prediction necessarily
    # lags -- it hasn't seen the new regime yet -- producing a value-loss
    # error that can be enormous (squared error against a return that just
    # swung by dozens of points). Under a SHARED clip_by_global_norm across
    # the whole params tree, that spike dominates the combined norm and the
    # clip scale factor collapses -- which shrinks the POLICY's own
    # gradient by the SAME factor, even though pi and vf share zero
    # parameters and the policy gradient (built from normalized advantages)
    # was perfectly well-scaled on its own. Measured directly: a realistic
    # post-collapse scenario suppressed the policy's effective applied
    # gradient to ~2% of what it would get clipped to independently --
    # the exact shape of "a correct, well-scaled corrective signal gets
    # silently neutralized by an unrelated value-function shock". This is
    # a plausible root-cause candidate for why gradient descent alone
    # doesn't self-correct a collapse even with curriculum retreated back
    # to the easiest possible task: the value function's own necessary
    # re-calibration can crowd out the policy's ability to learn from a
    # correctly-computed reward signal, every single update, for as long
    # as returns keep swinging.
    #
    # Also: value_loss now uses Huber (see VALUE_HUBER_DELTA below)
    # instead of raw MSE, which caps each individual sample's contribution
    # to the value gradient -- reduces how large a spike can even become
    # in the first place, independent of and complementary to the
    # separate-clipping fix below.
    def _param_group_labels(p):
        return {'pi': 'policy', 'vf': 'value', 'log_std': 'policy'}
    optimiser = optax.apply_if_finite(
        optax.partition(
            {
                'policy': optax.chain(optax.clip_by_global_norm(float(max_grad_norm)), optax.adam(float(lr))),
                'value': optax.chain(optax.clip_by_global_norm(float(max_grad_norm)), optax.adam(float(lr))),
            },
            _param_group_labels,
        ),
        max_consecutive_errors=100,
    )

    def make_initial_state(curriculum_params):
        key = jax.random.PRNGKey(int(seed))
        key, pk, ek = jax.random.split(key, 3)
        params = init_params(pk)
        opt_state = optimiser.init(params)
        env_state, obs = env.reset(curriculum_params)
        return PPOState(params, opt_state, env_state, obs, ek)

    def policy_sample(params, obs, key, obs_mean, obs_std):
        norm_obs = normalize_obs(obs, obs_mean, obs_std, obs_norm_mask)
        mean, value = apply(params, norm_obs)
        std = jnp.exp(eff_log_std(params))
        raw = mean + std * jax.random.normal(key, mean.shape, dtype=jnp.float32)
        action = jnp.clip(raw, -1.0, 1.0)
        # PPO likelihoods use the actual pre-clip Gaussian sample by default,
        # while only the action handed to the simulator is clipped. This keeps
        # old/new log-probabilities attached to the same random variable.
        # The compatibility flag can deliberately evaluate likelihoods on the
        # clipped action for A/B work. `ppo_action_clip_frac` is logged either
        # way so late-run saturation remains visible.
        lp_action = raw if logp_on_raw_action else action
        lp = log_prob(params, mean, lp_action)
        return action, raw, lp, value

    def rollout_once(state: PPOState, curriculum_params, obs_mean, obs_std):
        def one_step(carry, _):
            s, key = carry
            key, ak = jax.random.split(key)
            action, raw_action, lp, value = policy_sample(s.params, s.obs, ak, obs_mean, obs_std)
            out = env.step(s.env_state, action, curriculum_params)
            trans = {
                'obs': s.obs,
                'action': action,
                'raw_action': raw_action,
                'logp': lp,
                'value': value,
                'reward': out.reward,
                'done': out.done,
                # 'episode_fell' is 1.0 only on the exact step a true
                # physical failure (height/tilt/roll/pitch collapse) ends the
                # episode, and 0.0 on ordinary steps *and* on timeout steps.
                # This lets compute_gae tell a real death (bootstrap = 0,
                # correct) apart from hitting max_episode_steps (bootstrap
                # should NOT be 0 -- see the comment in compute_gae).
                'terminal': out.metrics['episode_fell'],
                'bootstrap_obs': out.bootstrap_obs,
                'metrics': out.metrics,
            }
            s2 = PPOState(s.params, s.opt_state, out.state, out.obs, key)
            return (s2, key), trans
        (s2, key2), traj = jax.lax.scan(one_step, (state, state.key), None, length=rollout)
        # Value every transition's true physical next state BEFORE the vec-env
        # auto-reset. This makes timeout bootstrapping exact instead of using
        # V(reset_state), while true physical failures still zero bootstrap.
        _, next_values = apply(s2.params, normalize_obs(traj['bootstrap_obs'], obs_mean, obs_std, obs_norm_mask))
        s2 = PPOState(s2.params, s2.opt_state, s2.env_state, s2.obs, key2)
        return s2, traj, next_values

    def compute_gae(traj, next_values):
        """GAE with exact time-limit bootstrapping. `next_values[t]` is
        V(s_{t+1}) from the pre-reset physical state exposed by environment.py.
        A true fall zeros bootstrap; a timeout does not. Any episode boundary
        still resets the eligibility trace so advantages never bleed across
        auto-reset episodes."""
        rewards = traj['reward']
        dones = traj['done']
        terminals = traj['terminal']
        values = traj['value']
        deltas = rewards + gamma * next_values * (1.0 - terminals) - values
        def rev_step(gae, x):
            delta, done = x
            gae = delta + gamma * lam * (1.0 - done) * gae
            return gae, gae
        _, adv_rev = jax.lax.scan(rev_step, jnp.zeros_like(deltas[0]), (deltas[::-1], dones[::-1]))
        adv = adv_rev[::-1]
        return adv, adv + values

    def flatten_rollout(traj, adv, ret):
        flat = {
            'obs': traj['obs'].reshape((batch_size, obs_dim)),
            'action': (traj['raw_action'] if logp_on_raw_action else traj['action']).reshape((batch_size, action_dim)),
            'old_logp': traj['logp'].reshape((batch_size,)),
            'old_value': traj['value'].reshape((batch_size,)),
            'adv': adv.reshape((batch_size,)),
            'ret': ret.reshape((batch_size,)),
        }
        flat['adv'] = (flat['adv'] - jnp.mean(flat['adv'])) / (jnp.std(flat['adv']) + 1e-8)
        return flat

    def loss_fn(params, batch):
        mean, value = apply(params, batch['obs'])
        new_logp = log_prob(params, mean, batch['action'])
        # `jnp.exp(new_logp - old_logp)` is unbounded. Once the policy is
        # under enough stress that new_logp drifts far from old_logp for
        # even one sample in a batch, exp() overflows to inf; multiplied by
        # a negative advantage that becomes -inf, and a single -inf in the
        # batch mean poisons the whole loss. The clipped surrogate (pg2)
        # already makes extreme ratios irrelevant to the actual PPO update
        # once inside jnp.minimum -- clipping the log-ratio before
        # exponentiating changes no optimization semantics, it just keeps
        # the intermediate values finite. Standard practice in most PPO
        # implementations.
        log_ratio = jnp.clip(new_logp - batch['old_logp'], -20.0, 20.0)
        ratio = jnp.exp(log_ratio)
        # Defense in depth: also guard the advantage itself against ever
        # carrying a stray inf/nan into the loss from further upstream
        # (GAE/returns), independent of the ratio fix above.
        adv = jnp.clip(jnp.nan_to_num(batch['adv']), -50.0, 50.0)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -jnp.mean(jnp.minimum(pg1, pg2))
        value_loss = jnp.mean(optax.huber_loss(value, batch['ret'], delta=VALUE_HUBER_DELTA))
        ent = entropy(params)
        approx_kl = 0.5 * jnp.mean(log_ratio ** 2)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
        loss = policy_loss + value_cost * value_loss - entropy_cost * ent
        aux = {'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': ent, 'approx_kl': approx_kl, 'clip_frac': clip_frac,
               # Cheap per-term finite flags -- if a future run ever skips
               # gradients, these say which term to look at first instead
               # of starting the investigation from zero.
               'policy_loss_finite': jnp.isfinite(policy_loss).astype(jnp.float32),
               'value_loss_finite': jnp.isfinite(value_loss).astype(jnp.float32)}
        return loss, aux

    # clip_by_global_norm bounds a step's magnitude, not its direction, so a
    # norm-bounded step can still walk fully in an already-diverged
    # direction. This is a standard PPO KL early-stop (as in SB3): once a
    # minibatch's approx_kl exceeds KL_HARD_STOP, stop applying further
    # updates for the rest of this call.
    #
    # KL_HARD_STOP=1.0 is a safety net, not the trust region itself (PPO's
    # trust region is the clipped surrogate, clip_eps=0.2) -- set high
    # enough that a healthy run essentially never touches it, low enough to
    # still catch a genuine runaway before it does real damage. A much
    # lower hard stop (e.g. 0.03) can saturate instead -- pinning most
    # minibatches to a single applied gradient step per update and feeding
    # the curriculum governor a permanently-saturated stress signal, which
    # is worse than the problem it's meant to prevent. Step-size control is
    # handled separately by LR_TARGET_UPDATE_KL below, a soft continuous
    # controller instead of a cliff.
    KL_HARD_STOP = 1.0
    # Huber delta for the value loss -- errors up to this magnitude are
    # treated quadratically (same as ordinary MSE, preserving normal
    # fine-grained value learning), errors beyond it are treated linearly,
    # capping how large a single sample's contribution to the value
    # gradient can get. 10.0 is sized against this codebase's typical
    # per-step reward scale (a few points), not against the terminal fall
    # penalty specifically -- ordinary TD errors during healthy learning
    # should mostly stay well inside it.
    VALUE_HUBER_DELTA = 10.0

    def update_minibatch(carry, mb_idx):
        params, opt_state, flat, perm, stopped, lr_scale = carry
        ids = jax.lax.dynamic_slice(perm, (mb_idx * mb_size,), (mb_size,))
        batch = {k: v[ids] for k, v in flat.items()}
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        def tree_l2(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(jnp.square(x)) for x in leaves])))
        grad_policy = {'pi': grads['pi'], 'log_std': grads['log_std']}
        grad_value = {'vf': grads['vf']}
        aux = {**aux,
               'grad_norm_policy_preclip': tree_l2(grad_policy),
               'grad_norm_value_preclip': tree_l2(grad_value)}
        new_updates, new_opt_state = optimiser.update(grads, opt_state, params)
        # `lr_scale` is a traced scalar supplied per update by the
        # KL-adaptive controller in train() -- effective lr = args.lr *
        # lr_scale. Applied to the optimiser's OUTPUT rather than by
        # rebuilding the optimiser, so Adam's moment estimates keep tracking
        # the raw gradients (rescaling them would make the controller fight
        # Adam's own normalization) and nothing re-jits when the lr moves.
        new_updates = jax.tree_util.tree_map(lambda u: u * lr_scale, new_updates)
        aux = {**aux,
               'update_norm_policy': tree_l2({'pi': new_updates['pi'], 'log_std': new_updates['log_std']}),
               'update_norm_value': tree_l2({'vf': new_updates['vf']})}
        new_params = optax.apply_updates(params, new_updates)
        apply_this = jnp.logical_not(stopped)
        params = jax.tree_util.tree_map(lambda new, old: jnp.where(apply_this, new, old), new_params, params)
        opt_state = jax.tree_util.tree_map(lambda new, old: jnp.where(apply_this, new, old), new_opt_state, opt_state)
        stopped = jnp.logical_or(stopped, aux['approx_kl'] > KL_HARD_STOP)
        aux = {**aux, 'loss': loss, 'kl_stopped': stopped.astype(jnp.float32)}
        return (params, opt_state, flat, perm, stopped, lr_scale), aux

    def update_epoch(carry, _):
        params, opt_state, flat, key, stopped, lr_scale = carry
        key, pk = jax.random.split(key)
        perm = jax.random.permutation(pk, batch_size)
        (params, opt_state, _, _, stopped, _), aux = jax.lax.scan(
            update_minibatch, (params, opt_state, flat, perm, stopped, lr_scale), jnp.arange(minibatches))
        aux_mean = {k: jnp.mean(v) for k, v in aux.items()}
        # The LAST minibatch's approx_kl, kept unaveraged. Every minibatch
        # measures divergence from the same fixed rollout policy, so this
        # one is "how far has this update walked in total" -- the quantity
        # a trust region is actually about. The mean over all minibatches
        # is a much weaker signal, dominated by the early, still-small
        # steps.
        aux_mean['approx_kl_final'] = aux['approx_kl'][-1]
        return (params, opt_state, flat, key, stopped, lr_scale), aux_mean

    @jax.jit
    def train_update(state: PPOState, curriculum_params, obs_mean, obs_std, lr_scale):
        state, traj, next_values = rollout_once(state, curriculum_params, obs_mean, obs_std)
        # This is the exact policy/optimizer state that GENERATED `traj`.
        # Health/checkpoint decisions made from this rollout must save THIS
        # state, not the post-gradient policy produced below.
        evaluated_state = state
        raw_obs_batch = traj['obs'].reshape((batch_size, obs_dim))
        adv, ret = compute_gae(traj, next_values)
        traj = {**traj, 'obs': normalize_obs(traj['obs'], obs_mean, obs_std, obs_norm_mask)}
        flat = flatten_rollout(traj, adv, ret)
        # explained_variance = 1 - Var[ret-value]/Var[ret], the standard
        # PPO value-function-health diagnostic (logged by Stable-
        # Baselines3, OpenAI Baselines, CleanRL, etc). Computed from
        # ROLLOUT-TIME predictions (old_value) against the actual GAE
        # returns, before this update's gradient steps touch anything --
        # "was the value function tracking reality coming into this
        # update". ~1.0 means well-calibrated; near 0 or negative means
        # it's not tracking at all, which is exactly the scenario the
        # separate policy/value gradient-clipping budgets above guard
        # against. Kept as a pre-update diagnostic deliberately, not
        # recomputed post-update, so it reflects the same "how well was
        # the value function doing" question every time, uncoupled from
        # how many epochs/minibatches happened this call.
        ret_var = jnp.var(flat['ret'])
        explained_variance = 1.0 - jnp.var(flat['ret'] - flat['old_value']) / jnp.maximum(ret_var, 1e-8)
        init_stopped = jnp.asarray(False)
        lr_scale = jnp.asarray(lr_scale, dtype=jnp.float32)
        (params, opt_state, _, key, _, _), aux = jax.lax.scan(
            update_epoch, (state.params, state.opt_state, flat, state.key, init_stopped, lr_scale), None, length=epochs)
        aux_mean = {k: jnp.mean(v) for k, v in aux.items()}
        # end of the last epoch's last minibatch = total divergence this update
        aux_mean['approx_kl_final'] = aux['approx_kl_final'][-1]
        aux_mean['lr_scale'] = lr_scale
        # Two cheap diagnostics:
        #  - action_clip_frac: what fraction of sampled action components hit
        #    the +/-1 clip in policy_sample. That matters because `lp` is the
        #    density of the CLIPPED action under the UNclipped Gaussian, so a
        #    high clip fraction inflates approx_kl for reasons that have
        #    nothing to do with step size. Logging it first (rather than
        #    changing the semantics blind) so the next run can say whether
        #    this is happening at all -- see --logp-on-raw-action.
        #  - adv_tail_frac: fraction of normalized advantages beyond 5 sigma.
        #    The terminal fall penalty is -65 to -110 against a per-step
        #    reward of ~+2.5, so the advantage distribution is expected to be
        #    heavy-tailed; this says how heavy, which bears directly on how
        #    coherent (and therefore how large in KL) a single step is.
        action_clip_mask = jnp.abs(traj['raw_action']) > 1.0
        action_clip_frac = jnp.mean(action_clip_mask.astype(jnp.float32))
        adv_tail_frac = jnp.mean((jnp.abs(flat['adv']) > 5.0).astype(jnp.float32))
        aux_mean['action_clip_frac'] = action_clip_frac
        aux_mean['adv_tail_frac'] = adv_tail_frac
        for ji in range(action_dim):
            aux_mean[f'action_clip_j{ji}'] = jnp.mean(action_clip_mask[..., ji].astype(jnp.float32))
        # Per-regime credit/value diagnostics: a mixed-policy collapse can be
        # hidden by global advantage normalisation or a healthy global EV.
        raw_adv = adv.reshape((batch_size,))
        raw_ret = ret.reshape((batch_size,))
        raw_val = traj['value'].reshape((batch_size,))
        for rn in ('steady', 'transition', 'resilience'):
            mask = traj['metrics'][f'regime_{rn}_frontier'].reshape((batch_size,)) > 0.5
            n = jnp.maximum(1.0, jnp.sum(mask.astype(jnp.float32)))
            def masked_mean_x(x):
                return jnp.sum(jnp.where(mask, x, 0.0)) / n
            a_mu = masked_mean_x(raw_adv)
            a_var = masked_mean_x((raw_adv - a_mu) ** 2)
            r_mu = masked_mean_x(raw_ret)
            rv = masked_mean_x((raw_ret - r_mu) ** 2)
            err_mu = masked_mean_x(raw_ret - raw_val)
            err_var = masked_mean_x(((raw_ret - raw_val) - err_mu) ** 2)
            aux_mean[f'adv_mean_{rn}'] = a_mu
            aux_mean[f'adv_std_{rn}'] = jnp.sqrt(jnp.maximum(a_var, 0.0))
            aux_mean[f'value_ev_{rn}'] = 1.0 - err_var / jnp.maximum(rv, 1e-8)
        new_state = PPOState(params, opt_state, state.env_state, state.obs, key)
        rollout_metrics = traj['metrics']
        # Mean exploration std, so a run can show directly whether
        # entropy is shrinking through a collapse (a real, literature-
        # documented risk in policy-gradient methods -- see LOG_STD_MIN's
        # own comment above) rather than that staying purely theoretical.
        # Taken post-update (this call's new params) since the point is
        # "where did exploration end up", not where it started.
        exploration_std_mean = jnp.mean(jnp.exp(eff_log_std(params)))
        aux_mean = {**aux_mean, 'rollout_reward_mean': jnp.mean(traj['reward']), 'rollout_done_rate': jnp.mean(traj['done']),
                    'explained_variance': explained_variance, 'exploration_std_mean': exploration_std_mean}
        return new_state, rollout_metrics, aux_mean, raw_obs_batch, evaluated_state

    @jax.jit
    def random_rollout(state: PPOState, curriculum_params):
        def one_step(carry, _):
            s, key = carry
            key, ak = jax.random.split(key)
            action = jax.random.uniform(ak, (env.num_envs, action_dim), minval=-1.0, maxval=1.0, dtype=jnp.float32)
            out = env.step(s.env_state, action, curriculum_params)
            s2 = PPOState(s.params, s.opt_state, out.state, out.obs, key)
            return (s2, key), out.metrics
        (s2, key2), metrics = jax.lax.scan(one_step, (state, state.key), None, length=rollout)
        return PPOState(s2.params, s2.opt_state, s2.env_state, s2.obs, key2), metrics

    @jax.jit
    def zero_rollout(state: PPOState, curriculum_params):
        def one_step(carry, _):
            s, key = carry
            action = jnp.zeros((env.num_envs, action_dim), dtype=jnp.float32)
            out = env.step(s.env_state, action, curriculum_params)
            s2 = PPOState(s.params, s.opt_state, out.state, out.obs, key)
            return (s2, key), out.metrics
        (s2, key2), metrics = jax.lax.scan(one_step, (state, state.key), None, length=rollout)
        return PPOState(s2.params, s2.opt_state, s2.env_state, s2.obs, key2), metrics

    @jax.jit
    def tiny_random_rollout(state: PPOState, curriculum_params):
        # Pre-training diagnostic: tiny action noise tests whether the nursery is
        # physically fair without letting a random policy thrash the servos.
        def one_step(carry, _):
            s, key = carry
            key, ak = jax.random.split(key)
            action = 0.08 * jax.random.uniform(ak, (env.num_envs, action_dim), minval=-1.0, maxval=1.0, dtype=jnp.float32)
            out = env.step(s.env_state, action, curriculum_params)
            s2 = PPOState(s.params, s.opt_state, out.state, out.obs, key)
            return (s2, key), out.metrics
        (s2, key2), metrics = jax.lax.scan(one_step, (state, state.key), None, length=rollout)
        return PPOState(s2.params, s2.opt_state, s2.env_state, s2.obs, key2), metrics

    @jax.jit
    def policy_rollout(state: PPOState, curriculum_params):
        # Untrained-policy diagnostic: uses the current policy distribution, not
        # PPO updates, to expose exploration/action-std aggression. Neutral
        # (no-op) normalization — this runs before any real obs stats exist.
        _neutral_mean = jnp.zeros((obs_dim,), dtype=jnp.float32)
        _neutral_std = jnp.ones((obs_dim,), dtype=jnp.float32)

        def one_step(carry, _):
            s, key = carry
            key, ak = jax.random.split(key)
            action, _raw, _logp, _value = policy_sample(s.params, s.obs, ak, _neutral_mean, _neutral_std)
            out = env.step(s.env_state, action, curriculum_params)
            s2 = PPOState(s.params, s.opt_state, out.state, out.obs, key)
            return (s2, key), out.metrics
        (s2, key2), metrics = jax.lax.scan(one_step, (state, state.key), None, length=rollout)
        return PPOState(s2.params, s2.opt_state, s2.env_state, s2.obs, key2), metrics

    def evaluate_rollout(params, curriculum_params, obs_mean, obs_std, eval_key, horizon):
        """Deterministic fixed-seed policy evaluation, side-effect free.
        Evaluation uses un-staggered fresh episodes so first-fall survival is
        directly comparable at every checkpoint. Auto-reset may happen after a
        fall, but `first_fall_step` records only the original cohort."""
        keys = jax.random.split(eval_key, env.num_envs)
        env_state = env._reset_with_keys(keys, curriculum_params)
        obs = env.obs(env_state)
        first_fall0 = jnp.full((env.num_envs,), -1, dtype=jnp.int32)
        def one_step(carry, i):
            es, ob, first_fall = carry
            mean, _ = apply(params, normalize_obs(ob, obs_mean, obs_std, obs_norm_mask))
            action = jnp.clip(mean, -1.0, 1.0)
            out = env.step(es, action, curriculum_params)
            fell = out.metrics['episode_fell'] > 0.5
            first_fall = jnp.where(jnp.logical_and(first_fall < 0, fell), i + 1, first_fall)
            return (out.state, out.obs, first_fall), out.metrics
        (_, _, first_fall), metrics = jax.lax.scan(
            one_step, (env_state, obs, first_fall0), jnp.arange(horizon, dtype=jnp.int32))
        return metrics, first_fall
    evaluate_rollout = jax.jit(evaluate_rollout, static_argnums=(5,))

    return {
        'make_initial_state': make_initial_state,
        'train_update': train_update,
        'random_rollout': random_rollout,
        'zero_rollout': zero_rollout,
        'evaluate_rollout': evaluate_rollout,
        'obs_norm_mask': obs_norm_mask,
        'tiny_random_rollout': tiny_random_rollout,
        'policy_rollout': policy_rollout,
        'batch_size': batch_size,
        'rollout': rollout,
        'envs': env.num_envs,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
    }

# ---------------------------------------------------------------------------
# Compact organic-birth curriculum/governor.
# ---------------------------------------------------------------------------

@dataclass
class RegimeParams:
    """Curriculum state: four 0-1 difficulty scalars (one per regime, plus
    posture) and three regime-allocation fractions. environment.py's own
    _lerp expands each difficulty scalar into a real physical range -- see
    its module doctrine for why one shared linear scalar per regime is
    enough, with no separate per-component rate table.
    """
    steady_difficulty: float = 0.0
    transition_difficulty: float = 0.0
    resilience_difficulty: float = 0.0
    posture_difficulty: float = 0.0
    regime_frac_steady: float = 1.0
    regime_frac_transition: float = 0.0
    regime_frac_resilience: float = 0.0
    regime_command_frac: float = 0.5
    core_rehearsal_frac: float = 0.0
    athletic_rehearsal_frac: float = 0.0
    root_z_offset: float = 0.0
    assist_scale: float = 0.0
    assist_hip_amp: float = 0.14
    assist_knee_amp: float = 0.22
    assist_sign: float = 1.0
    # v5006 adaptive frontier values in physical units. These are not capped
    # 0-1 completion scalars; they are the currently demonstrated boundary.
    frontier_vx_hi: float = 0.60
    frontier_vx_lo: float = -0.20
    frontier_vy: float = 0.22
    frontier_yaw: float = 1.50
    frontier_push_n: float = 0.70
    frontier_sensor_scale: float = 1.50
    frontier_combo_radius: float = 1.00
    frontier_transition_hold_s: float = 0.90
    frontier_ramp_steps: float = 4.0


class RegimeGovernor:
    """Curriculum controller for STEADY, TRANSITION, RESILIENCE and POSTURE.

    Each regime receives live data from the mixed vectorised rollout. Smoothed
    fall rate drives a banded proportional controller: expand below the target,
    hold near it, and retreat above it. Regime allocation favours skills that
    currently need more training while preserving a usable mix.

    Difficulty expansions share one combined per-update budget; retreats do
    not. The governor also watches the seed gait and stepping activity so a
    broad collapse can retreat all difficulty or request checkpoint recovery.
    All dynamic controller state is serialisable because resume/recovery must
    restore the controller as well as the network.
    """

    REGIMES = ('steady', 'transition', 'resilience')

    # One shared step size for ALL FOUR difficulty scalars (steady/
    # transition/resilience/posture) -- see environment.py's module
    # doctrine for the calibration finding that justifies a single shared
    # rate instead of a separately-tuned step per axis.
    DIFFICULTY_BASE_STEP = 0.0014
    FALL_TARGET = 0.30           # hold fall_rate near here (CLI-overridable -- see __init__'s fall_target)
    FALL_EMA_ALPHA = 0.25        # smoothing before the controller reacts, cuts noise-driven jitter
    PUSH_GAIN = 1.0              # push-side step scale at fall_ema == 0 (maximally easy)
    RETREAT_GAIN = 2.5           # retreat-side step scale at fall_ema == 1.0 -- bigger than PUSH_GAIN
                                    # on purpose, same reasoning SeedGovernor always used
    MAX_STEP_MULT = 3.0          # cap on either side's step, in units of DIFFICULTY_BASE_STEP
    MIN_ACTIONABLE_MULT = 0.05   # below this, treat as "at target" and don't move
    ALLOCATION_FLOOR = 0.10      # regime-allocation draw weight = FLOOR + fall_ema[regime] -- keeps a
                                    # currently-solid regime from dropping to near-zero envs forever
    BOOTSTRAP_UPDATES = 70       # steady-only updates before allocation opens up to all three regimes
                                    # (same magnitude as SeedGovernor's own TURN_LENGTH_UPDATES bootstrap)

    COMPETENCE_FLOOR = 0.35              # push also requires this -- don't expand what the robot can't do at all
    CORE_MIN_COMPETENCE = 0.50           # below this, the core (seed-band) is being neglected
    FALL_SEVERE_THRESHOLD = 0.65         # used by core-neglect and the stepping-collapse detector below
    COMPETENCE_SEVERE_THRESHOLD = 0.20
    CORE_RETREAT_STREAK_LIMIT = 20       # consecutive core-neglect updates before a hard retreat fires

    # v5007 command fidelity is measured after initial settling, on the
    # STEADY frontier cohort, and near the CURRENT physical boundary. Tolerance
    # has an absolute floor plus a command-relative term so low-amplitude vy/yaw
    # is not judged more tightly than the deployable estimator can observe.
    PRECISION_TOL_VX = 0.05
    PRECISION_TOL_VY = 0.035
    PRECISION_TOL_YAW = 0.12
    PRECISION_REL_VX = 0.12
    PRECISION_REL_VY = 0.18
    PRECISION_REL_YAW = 0.12
    FRONTIER_EDGE_FRAC = 0.72
    FRONTIER_SETTLE_S = 0.35
    FRONTIER_EXPAND_PRECISION = 0.62
    FRONTIER_RETREAT_PRECISION = 0.28
    FRONTIER_PRECISION_EMA_ALPHA = 0.15
    FRONTIER_BAD_STREAK_LIMIT = 8
    PRECISION_FLOOR = 0.50

    # Never let an exploratory frontier erase the already-demonstrated v5005
    # athletic envelope. Retreat means "back toward known competence", not
    # "delete a skill from the training distribution".
    FRONTIER_FLOOR_VX_HI = 0.60
    FRONTIER_FLOOR_VX_LO = -0.20
    FRONTIER_FLOOR_VY = 0.22
    FRONTIER_FLOOR_YAW = 1.50
    FRONTIER_FLOOR_COMBO = 1.00
    # Experiment guardrails only: these are not claimed capability ceilings;
    # they prevent one overnight run from requesting numerically absurd commands
    # before the edge-tracking gate has enough evidence to react.
    FRONTIER_SAFETY_VX_HI = 1.00
    FRONTIER_SAFETY_VX_LO_MAG = 0.45
    FRONTIER_SAFETY_VY = 0.45
    FRONTIER_SAFETY_YAW = 3.00
    FRONTIER_SAFETY_COMBO = 1.60

    KL_STRESS_EXPAND_FLOOR = 0.10
    KL_STRESS_CHRONIC_WINDOW = 50      # updates of history examined for a stuck gate
    KL_STRESS_CHRONIC_FRACTION = 0.90  # engaged at least this often over that window = stuck

    STEPPING_EVIDENCE_FLOOR = 0.15
    HIGH_STEP_EVIDENCE_FLOOR = 0.60
    STEPPING_COLLAPSE_LIMIT = 100     # consecutive-update fallback when recovery_window=0

    def __init__(self, fall_target: float | None = None, retreat_gain: float | None = None,
                 difficulty_base_step: float | None = None, bootstrap_updates: int | None = None,
                 regime_command_frac: float | None = None, core_rehearsal_frac: float | None = None,
                 athletic_rehearsal_frac: float | None = None,
                 recovery_window: int = 0, recovery_fraction: float = 0.40, d_gain: float = 0.0,
                 curriculum_push_every_updates: int = 1, open_frontier: bool = False,
                 frontier_growth_frac: float = 0.0015, frontier_retreat_frac: float = 0.0040,
                 push_frontier: bool = False, sensor_frontier: bool = False, combo_frontier: bool = False,
                 accel_frontier: bool = False):
        self.params = RegimeParams()
        if regime_command_frac is not None:
            self.params.regime_command_frac = float(regime_command_frac)
        if core_rehearsal_frac is not None:
            self.params.core_rehearsal_frac = float(core_rehearsal_frac)
        if athletic_rehearsal_frac is not None:
            self.params.athletic_rehearsal_frac = float(athletic_rehearsal_frac)
        if bootstrap_updates is not None:
            self.BOOTSTRAP_UPDATES = int(bootstrap_updates)
        self.competence: dict[str, float] = {r: 0.0 for r in self.REGIMES}
        self.core_competence = 1.0  # optimistic prior: no evidence of core neglect yet
        self.best_len = 0.0
        self.core_bad_streak = 0
        # One fall_ema per regime plus one for posture -- each gets its own
        # honest, continuously-updated signal every update, since every
        # regime has a live, continuous slice of envs every update (no
        # turn-taking to cause one regime's signal to go stale between
        # turns).
        self.fall_ema = {r: 1.0 for r in self.REGIMES}
        self.fall_ema['posture'] = 1.0
        # Pooled count EMAs behind fall_ema (see _pooled_fall_ema). Start at
        # zero: fall_ema keeps its conservative 1.0 prior only until the
        # first real episode of that kind completes, then is pure data.
        self._falls_ema = {k: 0.0 for k in self.fall_ema}
        self._episodes_ema = {k: 0.0 for k in self.fall_ema}
        self.stepping_collapse_streak = 0
        self.kl_stress_history: list[float] = []
        self.kl_stress_chronic = False
        self._updates_done_total = 0
        self.history: list[dict[str, float]] = []
        self.frontier_precision_ema = {k: -1.0 for k in ('vx_hi','vx_lo','vy','yaw','combo','accel')}
        self.frontier_bad_streak = {k: 0 for k in ('vx_hi','vx_lo','vy','yaw','combo','accel')}
        if fall_target is not None:
            self.FALL_TARGET = float(fall_target)
        if retreat_gain is not None:
            self.RETREAT_GAIN = float(retreat_gain)
        if difficulty_base_step is not None:
            self.DIFFICULTY_BASE_STEP = float(difficulty_base_step)
        # Rolling-window recovery trigger. A successful recovery clears this
        # history and the consecutive-update fallback together.
        self.recovery_window = int(recovery_window)
        self.recovery_fraction = float(recovery_fraction)
        self.stepping_collapse_history: list[float] = []
        self.D_GAIN = float(d_gain)
        self.curriculum_push_every_updates = max(1, int(curriculum_push_every_updates))
        self.open_frontier = bool(open_frontier)
        self.frontier_growth_frac = float(max(0.0, frontier_growth_frac))
        self.frontier_retreat_frac = float(max(0.0, frontier_retreat_frac))
        self.push_frontier = bool(push_frontier)
        self.sensor_frontier = bool(sensor_frontier)
        self.combo_frontier = bool(combo_frontier)
        self.accel_frontier = bool(accel_frontier)

    def state_dict(self) -> dict[str, Any]:
        """Everything that can affect the governor's next decision.
        Recovery/resume is a temporal rewind only if this dynamic state is
        restored alongside the network and optimizer."""
        return {
            'params': asdict(self.params),
            'competence': dict(self.competence),
            'core_competence': self.core_competence, 'best_len': self.best_len,
            'core_bad_streak': self.core_bad_streak, 'fall_ema': dict(self.fall_ema),
            '_falls_ema': dict(self._falls_ema), '_episodes_ema': dict(self._episodes_ema),
            'stepping_collapse_streak': self.stepping_collapse_streak,
            'stepping_collapse_history': list(self.stepping_collapse_history),
            'kl_stress_history': list(self.kl_stress_history), 'kl_stress_chronic': self.kl_stress_chronic,
            '_updates_done_total': self._updates_done_total,
            'open_frontier': self.open_frontier, 'frontier_growth_frac': self.frontier_growth_frac,
            'frontier_retreat_frac': self.frontier_retreat_frac, 'push_frontier': self.push_frontier,
            'sensor_frontier': self.sensor_frontier, 'combo_frontier': self.combo_frontier, 'accel_frontier': self.accel_frontier,
            'frontier_precision_ema': dict(self.frontier_precision_ema),
            'frontier_bad_streak': dict(self.frontier_bad_streak),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if not d:
            return
        known = {f.name for f in dataclass_fields(RegimeParams)}
        if isinstance(d.get('params'), dict):
            self.params = RegimeParams(**{k: v for k, v in d['params'].items() if k in known})
        for name in ('competence', 'fall_ema', '_falls_ema', '_episodes_ema',
                     'frontier_precision_ema', 'frontier_bad_streak'):
            if isinstance(d.get(name), dict): setattr(self, name, dict(d[name]))
        for name in ('core_competence', 'best_len', 'core_bad_streak', 'stepping_collapse_streak',
                     'kl_stress_chronic', '_updates_done_total'):
            if name in d: setattr(self, name, d[name])
        for name in ('stepping_collapse_history', 'kl_stress_history'):
            if isinstance(d.get(name), list): setattr(self, name, list(d[name]))

    def params_dict(self) -> dict[str, float]:
        return asdict(self.params)

    def _difficulty_field(self, name: str) -> str:
        return 'posture_difficulty' if name == 'posture' else f'{name}_difficulty'

    def _decide_difficulty_move(self, current_fall_ema: float, fall: float, competence: float,
                                 precision_ok: bool, kl_ok_to_expand: bool) -> tuple[str, float]:
        """The decision half of the shared proportional controller: what
        does this scalar want to do, and how strongly -- without applying
        it yet. Split from application because push moves need to be
        capped in aggregate across however many scalars want to push on
        the same update (see _apply_shared_push_budget); retreat doesn't,
        and applies immediately via _apply_retreat instead."""
        error = current_fall_ema - self.FALL_TARGET + self.D_GAIN * (fall - current_fall_ema)  # >0 too hard (retreat), <0 too easy (push)
        if error <= 0.0:
            mult = min(1.0, -error / self.FALL_TARGET) * self.PUSH_GAIN
            can_push = competence > self.COMPETENCE_FLOOR and precision_ok and kl_ok_to_expand
            if can_push and mult > self.MIN_ACTIONABLE_MULT:
                return 'push', mult
            elif mult > self.MIN_ACTIONABLE_MULT:
                return 'hold_push_blocked', 0.0  # would push, but competence/precision/KL vetoed it
            return 'hold_at_target', 0.0
        else:
            mult = min(1.0, error / (1.0 - self.FALL_TARGET)) * self.RETREAT_GAIN
            return ('retreat', mult) if mult > self.MIN_ACTIONABLE_MULT else ('hold_at_target', 0.0)

    def _apply_retreat(self, name: str, mult: float) -> None:
        """Applied immediately and in full -- retreat is never shared
        across scalars, unlike push. Every scalar that wants to retreat
        does so the same update, at full strength, regardless of how many
        others are also retreating. This is consistent with an asymmetry
        already built into the gains (RETREAT_GAIN > PUSH_GAIN): safety
        is never rationed, only ambition is."""
        field = self._difficulty_field(name)
        cur = getattr(self.params, field)
        setattr(self.params, field, max(0.0, cur - self.DIFFICULTY_BASE_STEP * min(mult, self.MAX_STEP_MULT)))

    def _apply_shared_push_budget(self, candidates: list[tuple[str, float]]) -> None:
        """Push candidates share one combined budget instead of each
        moving independently. Originally, nothing stopped several of the
        four scalars (three regimes plus posture) from pushing on the same
        update, each individually calibrated to what a single mover was
        allowed -- but with several able to fire at once, the combined
        change to the task distribution could be several times what any
        one update was designed to survive. A real run showed exactly this
        shape: steady, transition, and resilience difficulty climbing
        together for about 15M steps before a collapse.

        The fix: split whatever ceiling a single push could reach evenly
        across however many scalars want to push this update. Three
        regimes all maximally eager to push now move a third as far each;
        the total equals what one would have reached alone, not three
        times as much. Relative eagerness between candidates is preserved
        -- a scalar with a smaller raw mult still gets proportionally less
        of the shared budget, it's just that every share is scaled down
        by the same factor.

        Allocation (which regimes get envs next update) is untouched by
        this -- every regime still gets a live, continuous slice of data
        every update. Only how fast its difficulty can move is shared."""
        if not candidates:
            return
        n = len(candidates)
        for name, mult in candidates:
            field = self._difficulty_field(name)
            cur = getattr(self.params, field)
            shared_mult = min(mult, self.MAX_STEP_MULT) / n
            setattr(self.params, field, min(1.0, cur + self.DIFFICULTY_BASE_STEP * shared_mult))

    def _pooled_fall_ema(self, key: str, falls: float, episodes: float) -> float:
        """Smoothed fall rate as a ratio of two EMAs (falls, episodes)
        rather than an EMA of per-rollout ratios. Rollouts complete wildly
        different numbers of episodes -- a real run saw 1,344 in one and
        5 in another -- and averaging their ratios lets a 5-episode
        rollout count as much as a 1,344-episode one. Pooling counts
        weights every completed episode equally, which is what "fall rate"
        is supposed to mean. Both EMAs start at zero, so there's no
        initial bias once any data arrives."""
        a = self.FALL_EMA_ALPHA
        self._falls_ema[key] += a * (falls - self._falls_ema[key])
        self._episodes_ema[key] += a * (episodes - self._episodes_ema[key])
        if self._episodes_ema[key] > 1e-9:
            self.fall_ema[key] = self._falls_ema[key] / self._episodes_ema[key]
        return self.fall_ema[key]

    def note_recovery_performed(self) -> None:
        """Clear collapse evidence after an actual rollback.

        Both trigger paths reset together so stale history cannot immediately
        request another recovery from the restored policy.
        """
        self.stepping_collapse_streak = 0
        self.stepping_collapse_history = []

    def update(self, row: dict[str, float]) -> dict[str, float | str]:
        """`row` carries PER-REGIME aggregates (fall_rate_steady,
        ep_len_mean_steady, etc, from summarize_rollout's masked-by-regime
        breakdown) alongside GLOBAL ones -- global is used for posture (a
        dial every env samples regardless of regime) and for cross-regime
        concerns below (core-neglect, stepping-collapse, KL-stress) that
        aren't specific to any one regime."""
        ep_len = float(row.get('ep_len_mean', 0.0))
        fall = float(row.get('fall_rate', 1.0))
        # Pooled GLOBAL fall rate (every completed episode, any regime) --
        # drives posture and stepping-collapse detection. The raw per-
        # rollout `fall` is kept only for the derivative (D_GAIN) term.
        n_ep_all = float(row.get('episodes', 1.0))
        if n_ep_all > 0:
            self._pooled_fall_ema('posture', fall * n_ep_all, n_ep_all)
        global_fall = self.fall_ema['posture']
        self.best_len = max(self.best_len, ep_len)

        kl_stopped_frac = float(row.get('ppo_kl_stopped_frac', 0.0))
        self.kl_stress_history.append(kl_stopped_frac)
        self.kl_stress_history = self.kl_stress_history[-self.KL_STRESS_CHRONIC_WINDOW:]
        engaged = [1.0 for v in self.kl_stress_history if v > self.KL_STRESS_EXPAND_FLOOR]
        was_chronic = self.kl_stress_chronic
        self.kl_stress_chronic = (len(self.kl_stress_history) >= self.KL_STRESS_CHRONIC_WINDOW
                                   and len(engaged) >= self.KL_STRESS_CHRONIC_FRACTION * self.KL_STRESS_CHRONIC_WINDOW)
        kl_stress_now = kl_stopped_frac > self.KL_STRESS_EXPAND_FLOOR
        kl_ok_to_expand = (not kl_stress_now) or self.kl_stress_chronic

        # Stepping-collapse stays a GLOBAL signal -- "revert" (roll back
        # weights) is necessarily a global action, since all three regimes
        # share the one network being rolled back.
        stepping_collapsed = global_fall > self.FALL_SEVERE_THRESHOLD
        if self.recovery_window > 0:
            self.stepping_collapse_history.append(1.0 if stepping_collapsed else 0.0)
            self.stepping_collapse_history = self.stepping_collapse_history[-self.recovery_window:]
            bad = sum(self.stepping_collapse_history)
            needs_recovery = (len(self.stepping_collapse_history) >= self.recovery_window
                              and bad >= self.recovery_fraction * self.recovery_window)
            self.stepping_collapse_streak = self.stepping_collapse_streak + 1 if stepping_collapsed else 0
        else:
            self.stepping_collapse_streak = self.stepping_collapse_streak + 1 if stepping_collapsed else 0
            needs_recovery = self.stepping_collapse_streak >= self.STEPPING_COLLAPSE_LIMIT

        # Core-neglect stays global too, measured in a masked way
        # (episode_done + command-in-seed-band, regime-independent by
        # construction) -- a sustained violation retreats all FOUR scalars
        # together, not just one, since a basic-walk failure means the
        # whole policy needs to back off, not just whichever regime
        # happened to be pushing hardest.
        core_fall = row.get('core_fall_rate', -1.0)
        core_len = row.get('core_ep_len_mean', -1.0)
        if core_fall >= 0.0 and core_len >= 0.0:
            core_survival = min(1.0, core_len / 350.0) * (1.0 - min(1.0, core_fall))
            self.core_competence = float(max(0.0, min(1.0, core_survival)))
        best_competence = max(self.competence.values()) if self.competence else 0.0
        core_neglected = (self.core_competence < self.CORE_MIN_COMPETENCE
                          and best_competence >= self.COMPETENCE_SEVERE_THRESHOLD
                          and fall <= self.FALL_SEVERE_THRESHOLD)
        if core_neglected:
            self.core_bad_streak += 1
        else:
            self.core_bad_streak = 0
        global_retreat = self.core_bad_streak >= self.CORE_RETREAT_STREAK_LIMIT
        if global_retreat:
            for name in (*self.REGIMES, 'posture'):
                field = self._difficulty_field(name)
                cur = getattr(self.params, field)
                setattr(self.params, field, max(0.0, cur - self.DIFFICULTY_BASE_STEP * self.MAX_STEP_MULT))
            self.core_bad_streak = 0

        prec_vx = float(row.get('precision_vx', -1.0))
        precision_ok = prec_vx < 0.0 or prec_vx >= self.PRECISION_FLOOR
        # Expansion is deliberately blockwise when requested. Long episodes
        # can span ~22 PPO updates, so per-update pushing steers on telemetry
        # from older/easier cohorts. Retreat remains immediate; only ambition
        # waits for a fresh-enough cohort boundary.
        push_window = ((self._updates_done_total + 1) % self.curriculum_push_every_updates) == 0

        # THREE regimes, each judged off its OWN live data every update
        # -- no round-robin, no turn, no "which one is active" selection.
        # Every regime gets real attention every update because every
        # regime got a real, continuous slice of this update's own envs
        # (environment.py's per-env regime assignment). Decision and
        # application are SPLIT (see _decide_difficulty_move/
        # _apply_retreat/_apply_shared_push_budget): retreat applies
        # immediately, every time, for every scalar that wants it, but
        # push candidates are collected first and SHARE one aggregate
        # budget, so several regimes wanting to get harder on the same
        # update can't combine into a larger change than any single mover
        # was ever meant to produce.
        events: dict[str, str] = {}
        push_candidates: list[tuple[str, float]] = []
        for name in self.REGIMES:
            # If this regime had NO real episodes complete this rollout
            # (e.g. zero envs allocated to it during the steady-only
            # bootstrap, or just unlucky timing), skip its push/retreat
            # decision and its fall_ema update entirely -- leave everything
            # exactly as it was, rather than falling back to the global
            # fall rate, which would let a different regime's performance
            # silently drive this one's difficulty.
            regime_fall = row.get(f'fall_rate_{name}', -1.0)
            has_data = regime_fall >= 0.0
            if not has_data:
                events[name] = 'no_data'
                continue
            regime_fall = float(regime_fall)
            n_ep = float(row.get(f'episodes_{name}', 1.0))
            n_fall = float(row.get(f'falls_{name}', regime_fall * n_ep))
            self._pooled_fall_ema(name, n_fall, n_ep)
            ep_len_r = float(row.get(f'ep_len_mean_{name}', ep_len))
            cone_r = float(row.get(f'foot_mid_cone_{name}', 0.0))
            vx_r = row.get(f'vx_mean_{name}', -1.0)
            cmd_vx_r = row.get(f'cmd_vx_mean_{name}', -1.0)
            track_err_r = max(0.0, 1.0 - abs(float(vx_r) - float(cmd_vx_r)) / 0.18) if vx_r >= -0.5 and cmd_vx_r >= -0.5 else 0.0
            survival_r = min(1.0, ep_len_r / 350.0) * (1.0 - min(1.0, regime_fall))
            self.competence[name] = float(max(0.0, min(1.0, 0.65 * survival_r + 0.20 * cone_r * survival_r + 0.15 * track_err_r * survival_r)))
            if global_retreat:
                events[name] = 'retreat_core_neglect_all'
                continue
            action, mult = self._decide_difficulty_move(self.fall_ema[name], regime_fall, self.competence[name], precision_ok, kl_ok_to_expand and push_window)
            if action == 'retreat':
                self._apply_retreat(name, mult)
                events[name] = f'retreat_{name}'
            elif action == 'push':
                push_candidates.append((name, mult))
                events[name] = f'push_{name}'  # may be downgraded below if it ends up sharing a very small budget slice
            else:
                events[name] = action

        # Posture stays a global dial, judged on the OVERALL fall rate --
        # every env samples posture variation regardless of which regime
        # it's in, so there's no regime-specific subset to isolate its own
        # performance from. Still part of the same shared push budget as
        # the three regimes -- posture pushing alongside two regimes also
        # pushing is the same aggregate-change risk either way.
        if global_retreat:
            events['posture'] = 'retreat_core_neglect_all'
        else:
            action, mult = self._decide_difficulty_move(self.fall_ema['posture'], fall, best_competence, precision_ok, kl_ok_to_expand and push_window)
            if action == 'retreat':
                self._apply_retreat('posture', mult)
                events['posture'] = 'retreat_posture'
            elif action == 'push':
                push_candidates.append(('posture', mult))
                events['posture'] = 'push_posture'
            else:
                events['posture'] = action

        self._apply_shared_push_budget(push_candidates)
        if len(push_candidates) > 1:
            shared_note = f'push_shared_x{len(push_candidates)}'
            for name, _ in push_candidates:
                events[name] = shared_note

        # v5007 REPAIRED OPEN FRONTIER. Growth is judged on settled tracking
        # close to the CURRENT boundary, not on all rollout steps. This prevents
        # command transitions, low-amplitude centre samples and estimator noise
        # from voting a proven axis downward. A slow EMA + bad-streak hysteresis
        # keeps a single ugly rollout from eating the envelope.
        frontier_events = {}
        if self.open_frontier and not global_retreat:
            axis_specs = (
                ('vx_hi', 'frontier_precision_vx_forward', +1.0, self.FRONTIER_FLOOR_VX_HI, self.FRONTIER_SAFETY_VX_HI),
                ('vx_lo', 'frontier_precision_vx_reverse', -1.0, abs(self.FRONTIER_FLOOR_VX_LO), self.FRONTIER_SAFETY_VX_LO_MAG),
                ('vy', 'frontier_precision_vy', +1.0, self.FRONTIER_FLOOR_VY, self.FRONTIER_SAFETY_VY),
                ('yaw', 'frontier_precision_yaw', +1.0, self.FRONTIER_FLOOR_YAW, self.FRONTIER_SAFETY_YAW),
            )
            easy_enough = self.fall_ema['steady'] <= 0.25 and self.competence['steady'] >= 0.60
            for axis, pkey, sign, floor_mag, safety_mag in axis_specs:
                field = f'frontier_{axis}'
                cur = float(getattr(self.params, field))
                precision = float(row.get(pkey, -1.0))
                if precision < 0.0:
                    frontier_events[axis] = 'no_edge_data'
                    continue
                prev = float(self.frontier_precision_ema.get(axis, -1.0))
                ema = precision if prev < 0.0 else ((1.0-self.FRONTIER_PRECISION_EMA_ALPHA)*prev + self.FRONTIER_PRECISION_EMA_ALPHA*precision)
                self.frontier_precision_ema[axis] = ema
                if ema < self.FRONTIER_RETREAT_PRECISION and precision < 0.35:
                    self.frontier_bad_streak[axis] = int(self.frontier_bad_streak.get(axis, 0)) + 1
                else:
                    self.frontier_bad_streak[axis] = max(0, int(self.frontier_bad_streak.get(axis, 0)) - 1)
                good = ema >= self.FRONTIER_EXPAND_PRECISION and precision >= 0.55
                sustained_bad = self.frontier_bad_streak[axis] >= self.FRONTIER_BAD_STREAK_LIMIT
                if push_window and easy_enough and good:
                    mag = min(safety_mag, max(abs(cur), floor_mag) * (1.0 + self.frontier_growth_frac))
                    setattr(self.params, field, sign * mag)
                    frontier_events[axis] = 'expand_edge_proven'
                elif sustained_bad:
                    mag = max(floor_mag, abs(cur) * (1.0 - self.frontier_retreat_frac))
                    setattr(self.params, field, sign * mag)
                    frontier_events[axis] = 'retreat_sustained_tracking'
                else:
                    frontier_events[axis] = 'hold'

            if self.push_frontier:
                cur = float(self.params.frontier_push_n)
                if push_window and self.fall_ema['resilience'] <= 0.22 and self.competence['resilience'] >= 0.60:
                    self.params.frontier_push_n = min(float(getattr(self, 'frontier_push_safety_max_n', 3.0)), cur * (1.0 + self.frontier_growth_frac))
                    frontier_events['push_n'] = 'expand'
                elif self.fall_ema['resilience'] >= 0.45:
                    self.params.frontier_push_n = max(0.70, cur * (1.0 - self.frontier_retreat_frac))
                    frontier_events['push_n'] = 'retreat'
                else:
                    frontier_events['push_n'] = 'hold'

            if self.sensor_frontier:
                cur = float(self.params.frontier_sensor_scale)
                settled = min(float(row.get('precision_vx', 0.0)),
                              float(row.get('precision_vy', 0.0)),
                              float(row.get('precision_yaw', 0.0)))
                if push_window and global_fall <= 0.25 and settled >= 0.55:
                    self.params.frontier_sensor_scale = min(4.0, cur * (1.0 + 0.5*self.frontier_growth_frac))
                    frontier_events['sensor'] = 'expand'
                elif global_fall >= 0.50:
                    self.params.frontier_sensor_scale = max(1.50, cur * (1.0 - 0.5*self.frontier_retreat_frac))
                    frontier_events['sensor'] = 'retreat'
                else:
                    frontier_events['sensor'] = 'hold'

            if self.combo_frontier:
                cp = float(row.get('frontier_precision_combo', -1.0))
                cur = float(self.params.frontier_combo_radius)
                if cp >= 0.0:
                    prev = float(self.frontier_precision_ema.get('combo', -1.0))
                    ema = cp if prev < 0.0 else ((1.0-self.FRONTIER_PRECISION_EMA_ALPHA)*prev + self.FRONTIER_PRECISION_EMA_ALPHA*cp)
                    self.frontier_precision_ema['combo'] = ema
                    if ema < self.FRONTIER_RETREAT_PRECISION and cp < 0.35:
                        self.frontier_bad_streak['combo'] = int(self.frontier_bad_streak.get('combo', 0)) + 1
                    else:
                        self.frontier_bad_streak['combo'] = max(0, int(self.frontier_bad_streak.get('combo', 0)) - 1)
                    if push_window and easy_enough and ema >= 0.60 and cp >= 0.55:
                        self.params.frontier_combo_radius = min(self.FRONTIER_SAFETY_COMBO, cur * (1.0 + self.frontier_growth_frac))
                        frontier_events['combo'] = 'expand_edge_proven'
                    elif self.frontier_bad_streak['combo'] >= self.FRONTIER_BAD_STREAK_LIMIT:
                        self.params.frontier_combo_radius = max(self.FRONTIER_FLOOR_COMBO, cur * (1.0 - self.frontier_retreat_frac))
                        frontier_events['combo'] = 'retreat_sustained_tracking'
                    else:
                        frontier_events['combo'] = 'hold'
                else:
                    frontier_events['combo'] = 'no_edge_data'

            if self.accel_frontier:
                tp = float(row.get('transition_precision', -1.0))
                if tp >= 0.0:
                    prev = float(self.frontier_precision_ema.get('accel', -1.0))
                    ema = tp if prev < 0.0 else ((1.0-self.FRONTIER_PRECISION_EMA_ALPHA)*prev + self.FRONTIER_PRECISION_EMA_ALPHA*tp)
                    self.frontier_precision_ema['accel'] = ema
                    if ema < 0.25 and tp < 0.32:
                        self.frontier_bad_streak['accel'] = int(self.frontier_bad_streak.get('accel', 0)) + 1
                    else:
                        self.frontier_bad_streak['accel'] = max(0, int(self.frontier_bad_streak.get('accel', 0)) - 1)
                    if push_window and self.fall_ema['transition'] <= 0.25 and self.competence['transition'] >= 0.60 and ema >= 0.55:
                        self.params.frontier_transition_hold_s = max(CONTROL_DT, self.params.frontier_transition_hold_s * (1.0 - self.frontier_growth_frac))
                        self.params.frontier_ramp_steps = max(1.0, self.params.frontier_ramp_steps * (1.0 - 2.0*self.frontier_growth_frac))
                        frontier_events['accel'] = 'expand_transition_proven'
                    elif self.fall_ema['transition'] >= 0.50 or self.frontier_bad_streak['accel'] >= self.FRONTIER_BAD_STREAK_LIMIT:
                        self.params.frontier_transition_hold_s = min(0.90, self.params.frontier_transition_hold_s * (1.0 + self.frontier_retreat_frac))
                        self.params.frontier_ramp_steps = min(4.0, self.params.frontier_ramp_steps * (1.0 + 2.0*self.frontier_retreat_frac))
                        frontier_events['accel'] = 'retreat'
                    else:
                        frontier_events['accel'] = 'hold'
                else:
                    frontier_events['accel'] = 'no_transition_data'

        # Regime ALLOCATION -- what fraction of envs each regime gets
        # NEXT update. Draw weight = FLOOR + fall_ema[x], so a regime
        # currently struggling gets more env-share than one that's solid.
        # Steady-only bootstrap until a foundation gait exists.
        self._updates_done_total += 1
        if self._updates_done_total <= self.BOOTSTRAP_UPDATES:
            alloc = {'steady': 1.0, 'transition': 0.0, 'resilience': 0.0}
        else:
            weights = {r: self.ALLOCATION_FLOOR + self.fall_ema[r] for r in self.REGIMES}
            total = sum(weights.values())
            alloc = {r: w / total for r, w in weights.items()}
        self.params.regime_frac_steady = alloc['steady']
        self.params.regime_frac_transition = alloc['transition']
        self.params.regime_frac_resilience = alloc['resilience']

        report: dict[str, float | str] = {
            'competence_steady': self.competence['steady'],
            'competence_transition': self.competence['transition'],
            'competence_resilience': self.competence['resilience'],
            'core_competence': self.core_competence,
            'best_len': self.best_len,
            'steady_difficulty': self.params.steady_difficulty,
            'transition_difficulty': self.params.transition_difficulty,
            'resilience_difficulty': self.params.resilience_difficulty,
            'posture_difficulty': self.params.posture_difficulty,
            'regime_frac_steady': alloc['steady'],
            'regime_frac_transition': alloc['transition'],
            'regime_frac_resilience': alloc['resilience'],
            'fall_ema_steady': self.fall_ema['steady'],
            'fall_ema_transition': self.fall_ema['transition'],
            'fall_ema_resilience': self.fall_ema['resilience'],
            'fall_ema_posture': self.fall_ema['posture'],
            'event_steady': events['steady'], 'event_transition': events['transition'],
            'event_resilience': events['resilience'], 'event_posture': events['posture'],
            'core_bad_streak': float(self.core_bad_streak),
            'precision_vx': prec_vx,
            'precision_vy': float(row.get('precision_vy', -1.0)),
            'precision_yaw': float(row.get('precision_yaw', -1.0)),
            'frontier_precision_vx_forward': float(row.get('frontier_precision_vx_forward', -1.0)),
            'frontier_precision_vx_reverse': float(row.get('frontier_precision_vx_reverse', -1.0)),
            'frontier_precision_vy': float(row.get('frontier_precision_vy', -1.0)),
            'frontier_precision_yaw': float(row.get('frontier_precision_yaw', -1.0)),
            'frontier_precision_ema_vx_hi': float(self.frontier_precision_ema.get('vx_hi', -1.0)),
            'frontier_precision_ema_vx_lo': float(self.frontier_precision_ema.get('vx_lo', -1.0)),
            'frontier_precision_ema_vy': float(self.frontier_precision_ema.get('vy', -1.0)),
            'frontier_precision_ema_yaw': float(self.frontier_precision_ema.get('yaw', -1.0)),
            'frontier_vx_hi': self.params.frontier_vx_hi,
            'frontier_vx_lo': self.params.frontier_vx_lo,
            'frontier_vy': self.params.frontier_vy,
            'frontier_yaw': self.params.frontier_yaw,
            'frontier_push_n': self.params.frontier_push_n,
            'frontier_sensor_scale': self.params.frontier_sensor_scale,
            'frontier_combo_radius': self.params.frontier_combo_radius,
            'precision_combo': float(row.get('precision_combo', -1.0)),
            'frontier_event_combo': frontier_events.get('combo','off'),
            'frontier_transition_hold_s': self.params.frontier_transition_hold_s,
            'frontier_ramp_steps': self.params.frontier_ramp_steps,
            'frontier_event_accel': frontier_events.get('accel','off'),
            'frontier_event_vx_hi': frontier_events.get('vx_hi','off'),
            'frontier_event_vx_lo': frontier_events.get('vx_lo','off'),
            'frontier_event_vy': frontier_events.get('vy','off'),
            'frontier_event_yaw': frontier_events.get('yaw','off'),
            'kl_stopped_frac_seen': kl_stopped_frac,
            'kl_stress_chronic': float(self.kl_stress_chronic),
            'stepping_collapse_streak': float(self.stepping_collapse_streak),
            'stepping_collapse_bad_frac': (sum(self.stepping_collapse_history) / len(self.stepping_collapse_history)
                                           if self.stepping_collapse_history else 0.0),
            'needs_recovery': needs_recovery,
            'global_retreat': global_retreat,
            'curriculum_push_window': float(push_window),
            'curriculum_push_every_updates': float(self.curriculum_push_every_updates),
        }
        if self.kl_stress_chronic and not was_chronic:
            print(f"{LOG_PREFIX} WARNING: ppo_kl_stopped_frac stuck above {self.KL_STRESS_EXPAND_FLOOR:.2f} for "
                  f"{self.KL_STRESS_CHRONIC_WINDOW} updates -- ignoring the KL-stress gate so it can't freeze "
                  "expansion indefinitely. Retreat is unaffected (driven by fall rate/competence only).", flush=True)
        self.history.append({**row, **report})
        self.history = self.history[-128:]
        return report


# ---------------------------------------------------------------------------
# Utilities / artifacts.
# ---------------------------------------------------------------------------

def configure_jax(device: str, allow_cpu_mjx: bool) -> None:
    if device == 'cpu':
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    elif device in ('cuda', 'gpu'):
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    # device=auto: don't set anything here -- but that does NOT mean
    # JAX_PLATFORM_NAME is unset. It may already be set in the calling
    # shell/environment for reasons unrelated to this process's own
    # --device flag, so the check below looks at the actual environment
    # state, not just what this function itself just set, and acts on it
    # regardless of how it got there.
    if os.environ.get('JAX_PLATFORM_NAME', '').lower() == 'gpu':
        # Forcing JAX_PLATFORM_NAME=gpu when no CUDA jaxlib is installed makes
        # jax.default_backend() raise a hard RuntimeError on first use instead
        # of gracefully falling back to cpu the way it does when nothing
        # forces a platform. And there's no recovering after the fact: jax
        # reads/caches the platform preference at *import* time, so popping
        # the env var or calling jax.clear_backends() after `import jax` has
        # already run doesn't help. So: remove the forcing *before* the
        # probe import, not after.
        forced = os.environ.pop('JAX_PLATFORM_NAME')
        import jax as _probe_jax
        real_backend = _probe_jax.default_backend()
        if real_backend == 'gpu':
            os.environ['JAX_PLATFORM_NAME'] = forced
        else:
            print(f"{LOG_PREFIX} GPU was requested (device={device!r}, or via an externally-set "
                  f"JAX_PLATFORM_NAME) but no CUDA-enabled jaxlib is installed (jax resolves to "
                  f"{real_backend!r} on its own) — continuing without forcing gpu, so it falls "
                  f"back to cpu instead of crashing. --allow-cpu-mjx still gates whether that's allowed.")


def require_backend(allow_cpu_mjx: bool) -> None:
    import jax
    backend = jax.default_backend()
    if backend != 'gpu' and not allow_cpu_mjx:
        raise SystemExit(f'{LOG_PREFIX} JAX backend is {backend!r}, not gpu. Refusing real MJX training. Use --allow-cpu-mjx only for smoke/debug.')


def mean_metric(metrics: dict[str, Any], name: str) -> float:
    if name not in metrics:
        return 0.0
    return float(np.asarray(metrics[name]).mean())


def done_mean(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    # A thin wrapper over masked_mean(metrics, name, 'episode_done',
    # default) -- kept as its own function since "done_mean" reads better
    # than the fully spelled-out call at its 9 call sites.
    return masked_mean(metrics, name, 'episode_done', default)


def masked_mean(metrics: dict[str, Any], value_name: str, mask_name: str, default: float = 0.0) -> float:
    """Mean of `value_name` restricted to steps where `mask_name` > 0.5.

    Used to split ep_len_mean by *why* the episode ended (fall vs. timeout),
    since a single blended ep_len_mean can't distinguish "mostly instant
    deaths" from "mostly late stumbles right before the time limit" -- and
    those call for very different next actions.
    """
    if value_name not in metrics or mask_name not in metrics:
        return default
    mask = np.asarray(metrics[mask_name]) > 0.5
    vals = np.asarray(metrics[value_name])
    if mask.sum() <= 0:
        return default
    return float(vals[mask].mean())


def masked_mean_in_band(metrics: dict[str, Any], value_name: str, mask_name: str, band_name: str,
                         lo: float, hi: float, default: float = 0.0) -> float:
    """The core-holdout source: "core" is a fixed band on the actual
    sampled command, independent of curriculum difficulty and independent
    of which regime produced it. The reference band is SEED_VX_LO..
    SEED_VX_HI (the same constants _sample_task births the robot into) --
    the one capability that must never be lost regardless of how far any
    regime's own frontier has pushed. Filtering by the actual per-episode
    mean commanded vx means this keeps working correctly at any curriculum
    position, from any regime -- any episode that happens to land in the
    seed band counts, any episode outside it doesn't, regardless of label.
    """
    if value_name not in metrics or mask_name not in metrics or band_name not in metrics:
        return default
    band = np.asarray(metrics[band_name])
    mask = (np.asarray(metrics[mask_name]) > 0.5) & (band >= lo) & (band <= hi)
    vals = np.asarray(metrics[value_name])
    if mask.sum() <= 0:
        return default
    return float(vals[mask].mean())


def masked_mean_regime(metrics: dict[str, Any], value_name: str, regime_mask_name: str, default: float = -1.0) -> float:
    """Mean of an episode-outcome field (e.g. episode_fell,
    episode_length) restricted to done steps where `regime_mask_name`
    (regime_steady/regime_transition/regime_resilience, from environment.py's
    _metrics) was active for that env this step. Same masking discipline as
    masked_mean_in_band -- episode_done combined with a second filter --
    just a boolean regime mask instead of a numeric command band. Regime is
    assigned per-env at that env's own reset and held for the whole
    episode, so "was this env in regime X this step" and "was this env in
    regime X when its episode ended" are the same question -- no ambiguity
    from a regime that could change mid-episode."""
    if value_name not in metrics or regime_mask_name not in metrics or 'episode_done' not in metrics:
        return default
    done = np.asarray(metrics['episode_done']) > 0.5
    regime_mask = np.asarray(metrics[regime_mask_name]) > 0.5
    mask = done & regime_mask
    vals = np.asarray(metrics[value_name])
    if mask.sum() <= 0:
        return default
    return float(vals[mask].mean())


def _dynamic_tol(cmd: np.ndarray, abs_floor: float, rel_frac: float) -> np.ndarray:
    return np.maximum(abs_floor, rel_frac * np.abs(cmd))


def _settled_frontier_mask(metrics: dict[str, Any]) -> np.ndarray | None:
    """Stable command-hold steps suitable for frontier competence.

    Frontier expansion is intentionally judged only on the non-rehearsal
    STEADY cohort after an initial physical settling window. TRANSITION has its
    own acceleration frontier and RESILIENCE has its own push frontier; mixing
    those transients into an axis-boundary measurement was the v5006 failure.
    """
    needed = ('regime_steady_frontier', 'age_s')
    if any(k not in metrics for k in needed):
        return None
    steady = np.asarray(metrics['regime_steady_frontier']) > 0.5
    age = np.asarray(metrics['age_s']) >= RegimeGovernor.FRONTIER_SETTLE_S
    return steady & age


def _precision_on_mask(actual: np.ndarray, cmd: np.ndarray, mask: np.ndarray,
                       abs_floor: float, rel_frac: float, default: float = -1.0) -> float:
    if mask.sum() <= 0:
        return default
    tol = _dynamic_tol(cmd, abs_floor, rel_frac)
    return float((np.abs(actual - cmd)[mask] <= tol[mask]).mean())


def frontier_tracking_metrics(metrics: dict[str, Any], frontier_params: dict[str, float] | None,
                              combo_p: float = 1.35) -> dict[str, float]:
    """v5007 settled/edge tracking diagnostics used by the physical frontier.

    The ordinary ``precision_*`` fields are settled whole-envelope fidelity.
    ``frontier_precision_*`` uses only samples in the outer 28% of the current
    axis/radius, which makes expansion evidence actually refer to the boundary.
    """
    needed = ('vx','vy','yaw_rate','cmd_vx','cmd_vy','cmd_yaw')
    if any(k not in metrics for k in needed):
        return {}
    base = _settled_frontier_mask(metrics)
    if base is None:
        return {}
    vx=np.asarray(metrics['vx']); vy=np.asarray(metrics['vy']); yw=np.asarray(metrics['yaw_rate'])
    cvx=np.asarray(metrics['cmd_vx']); cvy=np.asarray(metrics['cmd_vy']); cyw=np.asarray(metrics['cmd_yaw'])
    p = frontier_params or {}
    hi=max(0.05, float(p.get('frontier_vx_hi', RegimeGovernor.FRONTIER_FLOOR_VX_HI)))
    lo=min(-0.02, float(p.get('frontier_vx_lo', RegimeGovernor.FRONTIER_FLOOR_VX_LO)))
    vym=max(0.02, abs(float(p.get('frontier_vy', RegimeGovernor.FRONTIER_FLOOR_VY))))
    ywm=max(0.04, abs(float(p.get('frontier_yaw', RegimeGovernor.FRONTIER_FLOOR_YAW))))
    radius=max(0.20, abs(float(p.get('frontier_combo_radius', RegimeGovernor.FRONTIER_FLOOR_COMBO))))
    edge=RegimeGovernor.FRONTIER_EDGE_FRAC

    out = {
        'precision_vx': _precision_on_mask(vx, cvx, base, RegimeGovernor.PRECISION_TOL_VX, RegimeGovernor.PRECISION_REL_VX),
        'precision_vy': _precision_on_mask(vy, cvy, base, RegimeGovernor.PRECISION_TOL_VY, RegimeGovernor.PRECISION_REL_VY),
        'precision_yaw': _precision_on_mask(yw, cyw, base, RegimeGovernor.PRECISION_TOL_YAW, RegimeGovernor.PRECISION_REL_YAW),
    }
    m_hi=base & (cvx >= edge*hi)
    m_lo=base & (cvx <= edge*lo)
    m_vy=base & (np.abs(cvy) >= edge*vym)
    m_yw=base & (np.abs(cyw) >= edge*ywm)
    out['frontier_precision_vx_forward'] = _precision_on_mask(vx, cvx, m_hi, RegimeGovernor.PRECISION_TOL_VX, RegimeGovernor.PRECISION_REL_VX)
    out['frontier_precision_vx_reverse'] = _precision_on_mask(vx, cvx, m_lo, RegimeGovernor.PRECISION_TOL_VX, RegimeGovernor.PRECISION_REL_VX)
    out['frontier_precision_vy'] = _precision_on_mask(vy, cvy, m_vy, RegimeGovernor.PRECISION_TOL_VY, RegimeGovernor.PRECISION_REL_VY)
    out['frontier_precision_yaw'] = _precision_on_mask(yw, cyw, m_yw, RegimeGovernor.PRECISION_TOL_YAW, RegimeGovernor.PRECISION_REL_YAW)
    out['frontier_edge_frac_vx_forward'] = float(m_hi.mean())
    out['frontier_edge_frac_vx_reverse'] = float(m_lo.mean())
    out['frontier_edge_frac_vy'] = float(m_vy.mean())
    out['frontier_edge_frac_yaw'] = float(m_yw.mean())

    # Mixed-command edge in normalized physical coordinates. Positive and
    # reverse vx use their own scales because the envelope is asymmetric.
    nx=np.where(cvx>=0.0, cvx/hi, cvx/abs(lo))
    ny=cvy/vym; nz=cyw/ywm
    pp=max(1.0, float(combo_p))
    norm=np.power(np.abs(nx)**pp + np.abs(ny)**pp + np.abs(nz)**pp, 1.0/pp)
    active=(np.abs(nx)>0.18).astype(np.int32)+(np.abs(ny)>0.18).astype(np.int32)+(np.abs(nz)>0.18).astype(np.int32)
    cmask=base & (active>=2) & (norm >= edge*radius)
    if cmask.sum() > 0:
        goodx=np.abs(vx-cvx)<=_dynamic_tol(cvx,RegimeGovernor.PRECISION_TOL_VX,RegimeGovernor.PRECISION_REL_VX)
        goody=np.abs(vy-cvy)<=_dynamic_tol(cvy,RegimeGovernor.PRECISION_TOL_VY,RegimeGovernor.PRECISION_REL_VY)
        goodyaw=np.abs(yw-cyw)<=_dynamic_tol(cyw,RegimeGovernor.PRECISION_TOL_YAW,RegimeGovernor.PRECISION_REL_YAW)
        out['frontier_precision_combo']=float((goodx & goody & goodyaw)[cmask].mean())
    else:
        out['frontier_precision_combo']=-1.0
    out['frontier_edge_frac_combo']=float(cmask.mean())

    # Acceleration competence is deliberately separate: score the ramp itself
    # with relaxed (2x) dynamic tolerances and average the three axis success
    # fractions rather than requiring all three to be perfect simultaneously.
    if 'transition_frac' in metrics:
        rmask=np.asarray(metrics['transition_frac'])>0.5
        if rmask.sum()>0:
            scores=[]
            for actual,cmd,af,rf in ((vx,cvx,RegimeGovernor.PRECISION_TOL_VX,RegimeGovernor.PRECISION_REL_VX),
                                     (vy,cvy,RegimeGovernor.PRECISION_TOL_VY,RegimeGovernor.PRECISION_REL_VY),
                                     (yw,cyw,RegimeGovernor.PRECISION_TOL_YAW,RegimeGovernor.PRECISION_REL_YAW)):
                tol=2.0*_dynamic_tol(cmd,af,rf)
                scores.append(float((np.abs(actual-cmd)[rmask]<=tol[rmask]).mean()))
            out['transition_precision']=float(sum(scores)/len(scores))
        else:
            out['transition_precision']=-1.0
    return out


def summarize_rollout(metrics: dict[str, Any], frontier_params: dict[str, float] | None = None, combo_p: float = 1.35) -> dict[str, float]:
    done = np.asarray(metrics.get('episode_done', 0.0)) > 0.5
    episodes = int(done.sum())
    row = {
        'episodes': float(episodes),
        'ep_len_mean': done_mean(metrics, 'episode_length', default=0.0),
        'ep_rew_mean': done_mean(metrics, 'episode_return', default=0.0),
        'fall_rate': done_mean(metrics, 'episode_fell', default=mean_metric(metrics, 'terminal')),
        'timeout_rate': done_mean(metrics, 'episode_timeout', default=0.0),
        'vx_mean': done_mean(metrics, 'episode_vx_mean', default=mean_metric(metrics, 'vx')),
        'cmd_vx_mean': done_mean(metrics, 'episode_cmd_vx_mean', default=mean_metric(metrics, 'cmd_vx')),
        'bad_contact': done_mean(metrics, 'episode_bad_contact', default=mean_metric(metrics, 'bad_contact')),
        'support': done_mean(metrics, 'episode_support', default=mean_metric(metrics, 'support')),
        'fall_height': mean_metric(metrics, 'fall_height'),
        'fall_tilt': mean_metric(metrics, 'fall_tilt'),
        'fall_bad_contact': mean_metric(metrics, 'fall_bad_contact'),
        'fall_roll_rate': mean_metric(metrics, 'fall_roll_rate'),
        'fall_pitch_rate': mean_metric(metrics, 'fall_pitch_rate'),
        'height': mean_metric(metrics, 'height'),
        'gravity_z': mean_metric(metrics, 'gravity_z'),
        'overspeed': mean_metric(metrics, 'overspeed'),
        'underspeed': mean_metric(metrics, 'underspeed'),
        'foot_mid_angle_deg': mean_metric(metrics, 'foot_mid_angle_deg'),
        'foot_mid_cone': mean_metric(metrics, 'foot_mid_cone'),
        'phase_sync': mean_metric(metrics, 'phase_sync'),
        'step_evidence': mean_metric(metrics, 'step_evidence'),
        'organic_step_shape': mean_metric(metrics, 'organic_step_shape'),
        'reward_valid_alive': mean_metric(metrics, 'reward_valid_alive'),
        'reward_command_tracking': mean_metric(metrics, 'reward_command_tracking'),
        'reward_upright_height': mean_metric(metrics, 'reward_upright_height'),
        'reward_feet_only_contact': mean_metric(metrics, 'reward_feet_only_contact'),
        'reward_swing_clearance': mean_metric(metrics, 'reward_swing_clearance'),
        # Fall-conditioned vs. timeout-conditioned episode length.
        # default=-1.0 (not 0.0) means "no episodes of this kind ended this
        # rollout" so it's never confused with a real, alarming zero.
        'fail_ep_len_mean': masked_mean(metrics, 'episode_length', 'episode_fell', default=-1.0),
        'timeout_ep_len_mean': masked_mean(metrics, 'episode_length', 'episode_timeout', default=-1.0),
        # "core" is a fixed band on the actual commanded speed
        # (SEED_VX_LO..SEED_VX_HI), not a regime label -- see
        # masked_mean_in_band's own docstring for why.
        'core_fall_rate': masked_mean_in_band(metrics, 'episode_fell', 'episode_done', 'episode_cmd_vx_mean',
                                               SEED_VX_LO, SEED_VX_HI, default=-1.0),
        'core_ep_len_mean': masked_mean_in_band(metrics, 'episode_length', 'episode_done', 'episode_cmd_vx_mean',
                                                 SEED_VX_LO, SEED_VX_HI, default=-1.0),
        # v5007 precision fields are attached below by
        # frontier_tracking_metrics(), which scores settled physical tracking
        # and separately measures the actual frontier edge.
        # Requested regime fractions are curriculum controller outputs; these
        # are the ACTUAL per-step occupancies PPO consumed this rollout.
        'occupancy_steady': mean_metric(metrics, 'regime_steady'),
        'occupancy_transition': mean_metric(metrics, 'regime_transition'),
        'occupancy_resilience': mean_metric(metrics, 'regime_resilience'),
        'occupancy_core_rehearsal': mean_metric(metrics, 'core_rehearsal'),
        'occupancy_athletic_rehearsal': mean_metric(metrics, 'athletic_rehearsal'),
        # Difficulty attached to each episode at birth. Because 700-step
        # episodes can span ~22 rollout=32 PPO updates, these may materially
        # lag the current governor position when the episode finally ends.
        'birth_steady_d': done_mean(metrics, 'birth_steady_d', default=-1.0),
        'birth_transition_d': done_mean(metrics, 'birth_transition_d', default=-1.0),
        'birth_resilience_d': done_mean(metrics, 'birth_resilience_d', default=-1.0),
        'birth_posture_d': done_mean(metrics, 'birth_posture_d', default=-1.0),
        'core_rehearsal_fall_rate': masked_mean_regime(metrics, 'episode_fell', 'core_rehearsal', default=-1.0),
        'core_rehearsal_ep_len_mean': masked_mean_regime(metrics, 'episode_length', 'core_rehearsal', default=-1.0),
    }
    row.update(frontier_tracking_metrics(metrics, frontier_params, combo_p=combo_p))
    # Backward-compatible aliases used by older report code.
    row['precision_vx_forward'] = row.get('frontier_precision_vx_forward', -1.0)
    row['precision_vx_reverse'] = row.get('frontier_precision_vx_reverse', -1.0)
    row['precision_combo'] = row.get('frontier_precision_combo', -1.0)

    # Command-transition tracking (see environment.py's out_metrics).
    # transition_*_err = mean |vel - cmd| over the steps that were MID-RAMP;
    # holding_*_err = same over the steps that were not (named to avoid
    # colliding with the STEADY REGIME, a different concept -- this is
    # "not currently mid-ramp", not "in the steady regime"). -1.0 means
    # "no such steps this rollout" (e.g. min_hold_s still long so commands
    # almost never change) -- a sentinel, never a real zero-error reading.
    tf = mean_metric(metrics, 'transition_frac')
    row['transition_frac'] = tf
    for axis in ('vx', 'vy', 'yaw'):
        row[f'transition_{axis}_err'] = mean_metric(metrics, f'transition_{axis}_err') / tf if tf > 1e-4 else -1.0
        row[f'holding_{axis}_err'] = mean_metric(metrics, f'steady_{axis}_err') / (1.0 - tf) if (1.0 - tf) > 1e-4 else -1.0
    # Per-REGIME breakdown of the same fields the governor's competence
    # formula reads globally -- ep_len_mean/fall_rate/vx_mean/cmd_vx_mean
    # are done-masked AND regime-masked (via masked_mean_regime, same
    # discipline as core_fall_rate's done+band masking above); foot_mid_cone
    # is a plain per-step regime-masked mean, matching how the GLOBAL
    # foot_mid_cone field itself is computed (mean_metric, not done-masked).
    # -1.0 = no episodes of that regime ended this rollout, never a real zero.
    for regime in ('steady', 'transition', 'resilience'):
        rmask = f'regime_{regime}_frontier'
        row[f'fall_rate_{regime}'] = masked_mean_regime(metrics, 'episode_fell', rmask, default=-1.0)
        row[f'ep_len_mean_{regime}'] = masked_mean_regime(metrics, 'episode_length', rmask, default=-1.0)
        row[f'vx_mean_{regime}'] = masked_mean_regime(metrics, 'episode_vx_mean', rmask, default=-1.0)
        row[f'cmd_vx_mean_{regime}'] = masked_mean_regime(metrics, 'episode_cmd_vx_mean', rmask, default=-1.0)
        row[f'foot_mid_cone_{regime}'] = masked_mean(metrics, 'foot_mid_cone', rmask, default=0.0)
        # Raw counts, so RegimeGovernor can pool episodes across updates
        # (see its _pooled_fall_ema) instead of averaging per-rollout
        # ratios whose denominators can differ by 100x.
        if 'episode_done' in metrics and rmask in metrics:
            ended = (np.asarray(metrics['episode_done']) > 0.5) & (np.asarray(metrics[rmask]) > 0.5)
            row[f'episodes_{regime}'] = float(ended.sum())
            row[f'falls_{regime}'] = float((np.asarray(metrics['episode_fell'])[ended] > 0.5).sum()) if 'episode_fell' in metrics else 0.0
    return row


def summarize_fixed_eval(metrics: dict[str, Any], first_fall: Any, horizon: int) -> dict[str, float]:
    """Compact, branch-comparable held-out metrics. Unlike training
    episode statistics, these describe one fixed un-staggered cohort and are
    therefore meaningful across curriculum positions and across runs."""
    ff = np.asarray(first_fall)
    failed = ff >= 0
    vx = np.asarray(metrics.get('vx', 0.0)); cvx = np.asarray(metrics.get('cmd_vx', 0.0))
    vy = np.asarray(metrics.get('vy', 0.0)); cvy = np.asarray(metrics.get('cmd_vy', 0.0))
    yw = np.asarray(metrics.get('yaw_rate', 0.0)); cyw = np.asarray(metrics.get('cmd_yaw', 0.0))
    return {
        'survival': float(1.0 - failed.mean()),
        'first_fall_step_mean': float(ff[failed].mean()) if failed.any() else float(horizon),
        'vx_mae': float(np.mean(np.abs(vx - cvx))),
        'vy_mae': float(np.mean(np.abs(vy - cvy))),
        'yaw_mae': float(np.mean(np.abs(yw - cyw))),
        'gravity_z': mean_metric(metrics, 'gravity_z'),
        'foot_mid_cone': mean_metric(metrics, 'foot_mid_cone'),
        'step_evidence': mean_metric(metrics, 'step_evidence'),
        'support': mean_metric(metrics, 'support'),
    }


def ensure_run_dir(path: Path) -> None:
    (path / 'checkpoints').mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())
    if exists:
        try:
            with path.open('r', newline='', encoding='utf-8') as f:
                old_header = next(csv.reader(f))
            fields = old_header
        except Exception:
            pass
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_checkpoint(path: Path, state: Any, params: dict[str, float], meta: dict[str, Any],
                     obs_norm: dict[str, Any] | None = None, governor_meta: dict[str, Any] | None = None) -> None:
    """governor_meta carries governor bookkeeping that ISN'T a difficulty
    scalar (so doesn't belong in `params`/curriculum_params) but still
    needs to survive a resume -- specifically, updates_done_total, which
    gates the steady-only bootstrap. Without this, RegimeGovernor.__init__
    always starts updates_done_total at 0, so every resumed run would redo
    the full BOOTSTRAP_UPDATES steady-only bootstrap again, even though the
    checkpoint's own regime_frac_transition/resilience (part of
    curriculum_params, restored separately) already show real, live
    allocation from before. None (default) for callers that don't have a
    governor in scope (e.g. save_rotating_snapshot's own disaster-recovery
    snapshots, lower stakes than the primary resume targets below)."""
    import jax
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault('source_provenance', source_provenance_from_env())
    payload = {'state': jax.device_get(state), 'curriculum_params': params, 'meta': meta,
               'obs_norm': jax.device_get(obs_norm) if obs_norm is not None else None,
               'governor_meta': governor_meta}
    with path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    (path.parent / 'latest.txt').write_text('MJX checkpoint: latest_mjx.pkl\n', encoding='utf-8')


def payload_params_are_finite(payload: dict[str, Any]) -> bool:
    """Was the network in this checkpoint actually healthy?

    A NaN-poisoned run keeps overwriting `latest_mjx.pkl` with garbage --
    the file loads fine, it's just full of NaN weights. --resume checks
    for this instead of trusting any file that merely unpickles.
    """
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(payload['state'].params)
        return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in leaves)
    except Exception:
        return False


def _snapshot_path(checkpoints_dir: Path, step: int) -> Path:
    return checkpoints_dir / f'snapshot_step_{step:012d}.pkl'


def save_rotating_snapshot(checkpoints_dir: Path, state: Any, params: dict[str, float], meta: dict[str, Any],
                            obs_norm: dict[str, Any] | None, step: int, keep: int,
                            governor_meta: dict[str, Any] | None = None) -> None:
    """`latest_mjx.pkl` is always overwritten, so if it gets poisoned
    there is no way back -- the whole run's progress would be gone even
    though most of it was fine. This keeps up to `keep` older, named-by-
    step snapshots so an incident only costs the gap since the last good
    snapshot, not the entire run.
    """
    if keep <= 0:
        return
    save_checkpoint(_snapshot_path(checkpoints_dir, step), state, params, meta, obs_norm, governor_meta=governor_meta)
    existing = sorted(checkpoints_dir.glob('snapshot_step_*.pkl'))
    for stale in existing[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass


def find_newest_valid_snapshot(checkpoints_dir: Path) -> Path | None:
    for candidate in sorted(checkpoints_dir.glob('snapshot_step_*.pkl'), reverse=True):
        try:
            with candidate.open('rb') as f:
                payload = pickle.load(f)
            if payload_params_are_finite(payload):
                return candidate
        except Exception:
            continue
    return None


def _fmt_len(v: float) -> str:
    # sentinel -1.0 from masked_mean means "no episodes of this kind this rollout".
    return '-' if v < 0 else f"{v:.0f}"


def format_line(version: str, step: int, reward: float, row: dict[str, float], report: dict[str, Any], fps: float,
                 nan_grad_skips: int = 0) -> str:
    fail = f"{100*row['fall_height']:.0f}/{100*row['fall_tilt']:.0f}/{100*row['fall_bad_contact']:.0f}/{100*row['fall_roll_rate']:.0f}/{100*row['fall_pitch_rate']:.0f}"
    line1 = (f"[{version}] step={step:,} rew={reward:+.3f} len={row['ep_len_mean']:.0f} "
             f"fall={100*row['fall_rate']:.1f}% fps={fps:.0f} fail(h/t/b/r/p)={fail} "
             f"vx={row['vx_mean']:+.3f}/{row['cmd_vx_mean']:+.3f} cone={row['foot_mid_angle_deg']:.1f}deg "
             f"step={row['step_evidence']:.2f} sync={row['phase_sync']:.2f}")
    # Three regimes' difficulty + live env-allocation fraction, plus
    # posture's own global dial.
    line2 = (f"[{version}]   steady={report['steady_difficulty']:.2f}(env={report['regime_frac_steady']:.2f}) "
             f"transition={report['transition_difficulty']:.2f}(env={report['regime_frac_transition']:.2f}) "
             f"resilience={report['resilience_difficulty']:.2f}(env={report['regime_frac_resilience']:.2f}) "
             f"posture={report['posture_difficulty']:.2f} | "
             f"fall_ema steady/trans/res/post={report['fall_ema_steady']:.2f}/{report['fall_ema_transition']:.2f}/"
             f"{report['fall_ema_resilience']:.2f}/{report['fall_ema_posture']:.2f} target={RegimeGovernor.FALL_TARGET:.2f} | "
             f"fail_len={_fmt_len(row.get('fail_ep_len_mean', -1.0))} "
             f"timeout_len={_fmt_len(row.get('timeout_ep_len_mean', -1.0))} | "
             f"r_track={row['reward_command_tracking']:.2f} r_contact={row['reward_feet_only_contact']:.2f} | "
             f"nan_skips={nan_grad_skips} ev={row.get('explained_variance', 0.0):.2f} "
             f"expl_std={row.get('exploration_std_mean', 0.0):.3f}")
    line3 = (f"[{version}]   events steady/trans/res/post="
             f"{report['event_steady']}/{report['event_transition']}/{report['event_resilience']}/{report['event_posture']}"
             f"{' GLOBAL-RETREAT' if report.get('global_retreat') else ''} "
             f"comp steady/trans/res={report['competence_steady']:.2f}/{report['competence_transition']:.2f}/{report['competence_resilience']:.2f} "
             f"core_comp={report['core_competence']:.2f} core_bad_streak={int(report['core_bad_streak'])}/{RegimeGovernor.CORE_RETREAT_STREAK_LIMIT} "
             f"core_fall={row.get('core_fall_rate', -1.0):.2f} core_len={_fmt_len(row.get('core_ep_len_mean', -1.0))} "
             f"precision_vx={report['precision_vx']:.2f} combo={report.get('precision_combo',-1.0):.2f} kl_stress={report['kl_stopped_frac_seen']:.2f}"
             f"{'(CHRONIC-IGNORED)' if report.get('kl_stress_chronic', 0.0) else ''} "
             f"kl_end={row.get('ppo_approx_kl_final', 0.0):.3f} "
             f"eff_lr={row.get('ppo_effective_lr', 0.0):.2e} "
             f"a_clip={row.get('ppo_action_clip_frac', 0.0):.2f}")
    line4 = (f"[{version}]   frontier vx=[{report.get('frontier_vx_lo',0.0):+.3f},{report.get('frontier_vx_hi',0.0):+.3f}] "
             f"vy=±{report.get('frontier_vy',0.0):.3f} yaw=±{report.get('frontier_yaw',0.0):.3f} "
             f"combo={report.get('frontier_combo_radius',1.0):.3f} push={report.get('frontier_push_n',0.0):.3f}N "
             f"sensor={report.get('frontier_sensor_scale',0.0):.2f}x hold={report.get('frontier_transition_hold_s',0.0):.3f}s "
             f"ramp={report.get('frontier_ramp_steps',0.0):.2f}frames")
    return line1 + "\n" + line2 + "\n" + line3 + "\n" + line4

def env_kwargs_from_args(args) -> dict:
    """One source of truth for the physical/sensor envelope used by training."""
    return dict(
        actuator_realism=args.actuator_realism,
        comm_latency_steps=args.comm_latency_steps,
        push_max_force_n=args.push_max_force_n,
        stepping_activity_target=args.stepping_activity_target,
        speed_tolerance=args.speed_tolerance,
        legacy_rng_reuse=args.legacy_rng_reuse,
        vx_hi_max=args.vx_hi_max, vx_lo_min=args.vx_lo_min,
        vy_max_mps=args.vy_max_mps, yaw_max_rad_s=args.yaw_max_rad_s,
        target_rate_limit_rad_s=args.target_rate_limit_rad_s, disable_target_slew=args.disable_target_slew,
        phase_speed_gain=args.phase_speed_gain, phase_rate_max_hz=args.phase_rate_max_hz,
        sensor_noise_scale=args.sensor_noise_scale, open_frontier=args.open_frontier,
        combo_mix_frac=args.combo_mix_frac, combo_p=args.combo_p, combo_probe_frac=args.combo_probe_frac,
        phase_obs_scale=args.phase_obs_scale,
    )

# ---------------------------------------------------------------------------
# Run modes.
# ---------------------------------------------------------------------------

def run_drift_test(args: argparse.Namespace) -> int:
    configure_jax(args.device, args.allow_cpu_mjx)
    import jax
    require_backend(args.allow_cpu_mjx)
    env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs, seed=args.seed, max_episode_steps=args.max_episode_steps, **env_kwargs_from_args(args))
    trainer = make_trainer(env, seed=args.seed, lr=args.lr, rollout=args.rollout, epochs=1, minibatches=1)
    params = RegimeGovernor().params_dict()
    state = trainer['make_initial_state'](params)
    p = env.params_array(params)
    print(f"{LOG_PREFIX} drift-test backend={jax.default_backend()} envs={args.envs} rollout={args.rollout}", flush=True)
    for name, fn in [('zero', trainer['zero_rollout']), ('tiny', trainer['tiny_random_rollout']), ('policy', trainer['policy_rollout'])]:
        s2, metrics = fn(state, p)
        s2.obs.block_until_ready()
        row = summarize_rollout(metrics)
        print(f"{LOG_PREFIX} drift policy={name} len={row['ep_len_mean']:.0f} fall={100*row['fall_rate']:.1f}% "
              f"fail(h/t/b/r/p)={100*row['fall_height']:.0f}/{100*row['fall_tilt']:.0f}/{100*row['fall_bad_contact']:.0f}/{100*row['fall_roll_rate']:.0f}/{100*row['fall_pitch_rate']:.0f} "
              f"vx={row['vx_mean']:+.3f}/{row['cmd_vx_mean']:+.3f} cone={row['foot_mid_angle_deg']:.1f} step={row['step_evidence']:.2f}", flush=True)
    return 0


def run_speed_test(args: argparse.Namespace) -> int:
    configure_jax(args.device, args.allow_cpu_mjx)
    import jax
    require_backend(args.allow_cpu_mjx)
    env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs, seed=args.seed, max_episode_steps=args.max_episode_steps, **env_kwargs_from_args(args))
    trainer = make_trainer(env, seed=args.seed, lr=args.lr, rollout=args.rollout, epochs=1, minibatches=max(1, args.minibatches))
    state = trainer['make_initial_state'](RegimeGovernor().params_dict())
    p = env.params_array(RegimeGovernor().params_dict())
    print(f"{LOG_PREFIX} speed-test compile envs={args.envs} rollout={args.rollout} device={jax.default_backend()} obs_dim={OBS_DIM} action_dim={ACTION_DIM}", flush=True)
    import jax.numpy as jnp
    obs_mean = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    obs_std = jnp.ones((OBS_DIM,), dtype=jnp.float32)
    state, metrics, aux, _raw_obs, _evaluated_state = trainer['train_update'](state, p, obs_mean, obs_std, 1.0)
    state.obs.block_until_ready()
    total = args.envs * args.rollout
    start = time.time()
    while total < args.steps:
        state, metrics, aux, _raw_obs, _evaluated_state = trainer['train_update'](state, p, obs_mean, obs_std, 1.0)
        state.obs.block_until_ready()
        total += args.envs * args.rollout
        if total % max(args.envs * args.rollout, 65536) == 0:
            fps = total / max(1e-9, time.time() - start)
            print(f"{LOG_PREFIX} speed step={total:,} fps={fps:.0f} backend={jax.default_backend()}", flush=True)
    return 0


def train(args: argparse.Namespace) -> int:
    configure_jax(args.device, args.allow_cpu_mjx)
    import jax
    require_backend(args.allow_cpu_mjx)
    version = ROBOT_VERSION
    run_dir = Path(args.run_dir or os.environ.get('EMU_LAB_RUN_DIR') or (Path.cwd() / 'library' / 'runs' / args.run_name)).resolve()
    ensure_run_dir(run_dir)

    # --resume: 'auto' looks for this run's own latest checkpoint; anything
    # else is treated as a literal path. Restores policy/value params,
    # optimizer state, obs normalization, and curriculum (RegimeParams) --
    # NOT the physics buffer (mjx.Data/env_state), which just re-initializes
    # fresh episodes. That's a deliberate scope cut: mjx.Data is a large,
    # device-specific pytree that's fragile to pickle/restore across
    # processes or CPU<->GPU, and "resume with fresh episodes" is a
    # perfectly normal PPO restart -- only the learned network and
    # curriculum position need to survive.
    resumed_payload = None
    resumed_step = 0
    warm_payload = None
    if getattr(args, 'warm_start_weights', ''):
        wp = Path(args.warm_start_weights)
        if wp.exists():
            try:
                with wp.open('rb') as f: warm_payload = pickle.load(f)
                if not payload_params_are_finite(warm_payload):
                    print(f"{LOG_PREFIX} --warm-start-weights contains non-finite params; ignoring {wp}", flush=True)
                    warm_payload = None
                else:
                    print(f"{LOG_PREFIX} warm-starting network/normalizer from {wp}; optimizer + governor/frontier are fresh", flush=True)
            except Exception as exc:
                print(f"{LOG_PREFIX} warm-start failed ({exc!r}); starting fresh", flush=True)
                warm_payload = None
    if getattr(args, 'resume', ''):
        is_auto = args.resume.lower() == 'auto'
        resume_path = (run_dir / 'checkpoints' / 'latest_mjx.pkl') if is_auto else Path(args.resume)
        if resume_path.exists():
            try:
                with resume_path.open('rb') as f:
                    resumed_payload = pickle.load(f)
                # Don't trust a checkpoint just because it unpickled -- a
                # NaN-poisoned run can keep overwriting `latest_mjx.pkl`
                # with a perfectly loadable file full of NaN weights. Only
                # 'auto' gets the automatic fallback to the newest valid
                # rotating snapshot -- an explicit --resume path is the
                # user's deliberate choice, so a bad file there fails safe
                # to a fresh start instead of silently substituting a
                # different file they didn't ask for.
                if not payload_params_are_finite(resumed_payload):
                    print(f"{LOG_PREFIX} --resume: {resume_path} loaded but contains non-finite "
                          f"(NaN/Inf) params -- this checkpoint is poisoned, not usable.", flush=True)
                    resumed_payload = None
                    if is_auto:
                        fallback = find_newest_valid_snapshot(run_dir / 'checkpoints')
                        if fallback is not None:
                            with fallback.open('rb') as f:
                                resumed_payload = pickle.load(f)
                            print(f"{LOG_PREFIX} --resume: falling back to rotating snapshot {fallback}", flush=True)
                        else:
                            print(f"{LOG_PREFIX} --resume: no valid rotating snapshot found either -- starting fresh", flush=True)
                    else:
                        print(f"{LOG_PREFIX} --resume: starting fresh (pass a specific --resume <snapshot path> "
                              "if you want to try an older rotating snapshot instead)", flush=True)
                if resumed_payload is not None:
                    resumed_step = int(resumed_payload.get('meta', {}).get('step', 0))
                    print(f"{LOG_PREFIX} --resume: loaded checkpoint (was at step={resumed_step:,})", flush=True)
            except Exception as exc:
                print(f"{LOG_PREFIX} --resume: failed to load {resume_path} ({exc!r}) -- starting fresh instead", flush=True)
                resumed_payload = None
        else:
            print(f"{LOG_PREFIX} --resume: no checkpoint found at {resume_path} -- starting fresh", flush=True)

        # Seed THIS run's own best-tier snapshots from wherever we resumed
        # from, if it has any and we don't already. Auto-recovery (below,
        # "needs_recovery") only ever looks in run_dir/checkpoints/ -- a
        # resumed run (run_ladder.py's phase 2 included) starts with NONE
        # of that, so if it degraded early there would be nothing to roll
        # back to until it re-earned a tier from zero. Skipped when
        # resuming from THIS run's own directory (--resume auto after a
        # restart): source and destination are the same folder, nothing to
        # seed.
        if resumed_payload is not None:
            src_ckpt_dir = resume_path.parent
            dst_ckpt_dir = run_dir / 'checkpoints'
            if src_ckpt_dir.resolve() != dst_ckpt_dir.resolve():
                for tier_name in ('best_known_good.pkl', 'best_effort.pkl', 'best_seen.pkl'):
                    src_tier = src_ckpt_dir / tier_name
                    dst_tier = dst_ckpt_dir / tier_name
                    if src_tier.exists() and not dst_tier.exists():
                        dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_tier, dst_tier)
                        print(f"{LOG_PREFIX} --resume: seeded {tier_name} from the source run "
                              f"({src_tier}) -- auto-recovery has a real snapshot to roll back to "
                              f"from step 0 of this run, instead of needing to re-earn one first.",
                              flush=True)

    meta = {'version': version, 'robot_version': ROBOT_VERSION, 'env_version': ENV_VERSION, 'program_version': PROGRAM_VERSION,
            'source_provenance': source_provenance_from_env(),
            'args': vars(args), 'resumed_from_step': resumed_step if resumed_payload is not None else None}
    write_json(run_dir / 'run_meta.json', meta)

    env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs, seed=args.seed,
                    max_episode_steps=args.max_episode_steps, **env_kwargs_from_args(args))
    minibatches = args.minibatches if args.minibatches > 0 else (8 if args.envs * args.rollout >= 8192 else 4)
    trainer = make_trainer(env, seed=args.seed, lr=args.lr, rollout=args.rollout, epochs=args.epochs, minibatches=minibatches,
                           entropy_cost=args.entropy_cost, value_cost=args.value_cost,
                           logp_on_raw_action=bool(args.logp_on_raw_action),
                           obs_norm_mode=args.obs_norm_mode)
    # Fixed continuity evaluator deliberately does NOT inherit the live open
    # frontier/blob sampler. Otherwise dynamic50/dynamic80 would silently move
    # with the training frontier and cease to be longitudinal rulers. It keeps
    # the branch's actor/actuator observation contract, but v5005's fixed 0.60/
    # -0.20/0.22/1.50 command design range and 0.70N disturbance ceiling.
    fixed_eval_env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs,
                    seed=args.eval_seed, max_episode_steps=args.max_episode_steps,
                    actuator_realism=args.actuator_realism, comm_latency_steps=args.comm_latency_steps,
                    push_max_force_n=0.70, vx_hi_max=0.60, vx_lo_min=-0.20, vy_max_mps=0.22, yaw_max_rad_s=1.50,
                    target_rate_limit_rad_s=args.target_rate_limit_rad_s, disable_target_slew=args.disable_target_slew, phase_speed_gain=args.phase_speed_gain,
                    phase_rate_max_hz=args.phase_rate_max_hz, sensor_noise_scale=args.sensor_noise_scale,
                    open_frontier=False, combo_mix_frac=0.0, phase_obs_scale=args.phase_obs_scale)
    fixed_eval_trainer = make_trainer(fixed_eval_env, seed=args.eval_seed, lr=args.lr, rollout=args.rollout,
                           epochs=1, minibatches=1, entropy_cost=args.entropy_cost, value_cost=args.value_cost,
                           logp_on_raw_action=bool(args.logp_on_raw_action), obs_norm_mode=args.obs_norm_mode)
    # Fixed mixed-command ruler: same physical world as continuity eval, but
    # every birth is a genuine vx+vy+yaw p-norm command rather than axis-isolated
    # or independent rectangular sampling. This is the direct exam for the new
    # command-combination curriculum.
    combo_eval_env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs,
                    seed=args.eval_seed + 606, max_episode_steps=args.max_episode_steps,
                    actuator_realism=args.actuator_realism, comm_latency_steps=args.comm_latency_steps,
                    push_max_force_n=0.70, vx_hi_max=0.60, vx_lo_min=-0.20, vy_max_mps=0.22, yaw_max_rad_s=1.50,
                    target_rate_limit_rad_s=args.target_rate_limit_rad_s, disable_target_slew=args.disable_target_slew, phase_speed_gain=args.phase_speed_gain,
                    phase_rate_max_hz=args.phase_rate_max_hz, sensor_noise_scale=args.sensor_noise_scale,
                    open_frontier=False, combo_mix_frac=1.0, combo_p=1.35, combo_probe_frac=0.0,
                    phase_obs_scale=args.phase_obs_scale)
    combo_eval_trainer = make_trainer(combo_eval_env, seed=args.eval_seed + 606, lr=args.lr, rollout=args.rollout,
                           epochs=1, minibatches=1, entropy_cost=args.entropy_cost, value_cost=args.value_cost,
                           logp_on_raw_action=bool(args.logp_on_raw_action), obs_norm_mode=args.obs_norm_mode)
    # v5007 growth rulers deliberately extend beyond the v5005 envelope. They
    # stay FIXED across training: growth means a policy can control commands it
    # was never asked to master by the old 0.60/-0.20/0.22/1.50 benchmark.
    growth_eval_env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs,
                    seed=args.eval_seed + 811, max_episode_steps=args.max_episode_steps,
                    actuator_realism=args.actuator_realism, comm_latency_steps=args.comm_latency_steps,
                    push_max_force_n=0.70, vx_hi_max=0.75, vx_lo_min=-0.28, vy_max_mps=0.30, yaw_max_rad_s=1.90,
                    target_rate_limit_rad_s=args.target_rate_limit_rad_s, disable_target_slew=args.disable_target_slew,
                    phase_speed_gain=args.phase_speed_gain, phase_rate_max_hz=args.phase_rate_max_hz,
                    sensor_noise_scale=args.sensor_noise_scale, open_frontier=False, combo_mix_frac=0.0,
                    phase_obs_scale=args.phase_obs_scale)
    growth_eval_trainer = make_trainer(growth_eval_env, seed=args.eval_seed + 811, lr=args.lr, rollout=args.rollout,
                           epochs=1, minibatches=1, entropy_cost=args.entropy_cost, value_cost=args.value_cost,
                           logp_on_raw_action=bool(args.logp_on_raw_action), obs_norm_mode=args.obs_norm_mode)
    growth_combo_env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs,
                    seed=args.eval_seed + 919, max_episode_steps=args.max_episode_steps,
                    actuator_realism=args.actuator_realism, comm_latency_steps=args.comm_latency_steps,
                    push_max_force_n=0.70, vx_hi_max=0.75, vx_lo_min=-0.28, vy_max_mps=0.30, yaw_max_rad_s=1.90,
                    target_rate_limit_rad_s=args.target_rate_limit_rad_s, disable_target_slew=args.disable_target_slew,
                    phase_speed_gain=args.phase_speed_gain, phase_rate_max_hz=args.phase_rate_max_hz,
                    sensor_noise_scale=args.sensor_noise_scale, open_frontier=False, combo_mix_frac=1.0, combo_p=1.35,
                    combo_probe_frac=0.0, phase_obs_scale=args.phase_obs_scale)
    growth_combo_trainer = make_trainer(growth_combo_env, seed=args.eval_seed + 919, lr=args.lr, rollout=args.rollout,
                           epochs=1, minibatches=1, entropy_cost=args.entropy_cost, value_cost=args.value_cost,
                           logp_on_raw_action=bool(args.logp_on_raw_action), obs_norm_mode=args.obs_norm_mode)
    # Held-out transfer evaluator: every branch is tested in the same fixed,
    # less-friendly physical model (actuator inertia/friction, one-step latency
    # and stronger pushes). It measures transfer rather than letting a branch
    # grade itself in the exact environment it trained on.
    robust_eval_env = EmuMJXEnv(str(Path(__file__).with_name('robot.xml')), num_envs=args.envs,
                    seed=args.eval_seed + 404, max_episode_steps=args.max_episode_steps,
                    actuator_realism=True, comm_latency_steps=max(1, int(args.comm_latency_steps)),
                    push_max_force_n=0.70,
                    vx_hi_max=args.vx_hi_max, vx_lo_min=args.vx_lo_min,
                    vy_max_mps=args.vy_max_mps, yaw_max_rad_s=args.yaw_max_rad_s,
                    target_rate_limit_rad_s=args.target_rate_limit_rad_s, disable_target_slew=args.disable_target_slew,
                    phase_speed_gain=args.phase_speed_gain, phase_rate_max_hz=args.phase_rate_max_hz,
                    sensor_noise_scale=max(1.5, float(args.sensor_noise_scale)), open_frontier=False,
                    combo_mix_frac=0.0, combo_p=args.combo_p, combo_probe_frac=0.0,
                    phase_obs_scale=args.phase_obs_scale)
    robust_eval_trainer = make_trainer(robust_eval_env, seed=args.eval_seed + 404, lr=args.lr,
                           rollout=args.rollout, epochs=1, minibatches=1,
                           entropy_cost=args.entropy_cost, value_cost=args.value_cost,
                           logp_on_raw_action=bool(args.logp_on_raw_action),
                           obs_norm_mode=args.obs_norm_mode)
    # No track-level flags to validate here -- a regime's difficulty is judged directly by RegimeGovernor, with no scheduling layer in between.
    governor = RegimeGovernor(fall_target=args.fall_target, retreat_gain=args.retreat_gain,
                              difficulty_base_step=args.difficulty_base_step,
                              bootstrap_updates=args.bootstrap_updates,
                              regime_command_frac=args.regime_command_frac,
                              core_rehearsal_frac=args.core_rehearsal_frac,
                              athletic_rehearsal_frac=args.athletic_rehearsal_frac,
                              recovery_window=args.recovery_window, recovery_fraction=args.recovery_fraction,
                              d_gain=args.fall_derivative_gain,
                              curriculum_push_every_updates=args.curriculum_push_every_updates,
                              open_frontier=args.open_frontier, frontier_growth_frac=args.frontier_growth_frac,
                              frontier_retreat_frac=args.frontier_retreat_frac, push_frontier=args.push_frontier,
                              sensor_frontier=args.sensor_frontier, combo_frontier=(args.combo_mix_frac > 0.0), accel_frontier=args.accel_frontier)
    if resumed_payload is not None:
        try:
            saved_curriculum = resumed_payload.get('curriculum_params', {}) or {}
            known_fields = {f.name for f in dataclass_fields(RegimeParams)}
            governor.params = RegimeParams(**{k: v for k, v in saved_curriculum.items() if k in known_fields})
        except Exception as exc:
            print(f"{LOG_PREFIX} --resume: could not restore curriculum params ({exc!r}) -- using fresh RegimeParams", flush=True)
        gmeta = resumed_payload.get('governor_meta') or {}
        # Current checkpoints carry complete governor state. Keep the compact
        # updates_done_total-only path so older checkpoints remain loadable.
        try:
            if 'params' in gmeta or 'fall_ema' in gmeta:
                governor.load_state_dict(gmeta)
                print(f"{LOG_PREFIX} --resume: restored full governor state "
                      f"(updates={governor._updates_done_total}, core_bad_streak={governor.core_bad_streak}, "
                      f"kl_history={len(governor.kl_stress_history)}).", flush=True)
            elif 'updates_done_total' in gmeta:
                governor._updates_done_total = int(gmeta['updates_done_total'])
                print(f"{LOG_PREFIX} --resume: restored legacy updates_done_total={governor._updates_done_total}.", flush=True)
        except Exception as exc:
            print(f"{LOG_PREFIX} --resume: governor dynamic-state restore failed ({exc!r}); "
                  "curriculum scalars were still restored.", flush=True)
    params = governor.params_dict()
    state = trainer['make_initial_state'](params)
    if resumed_payload is not None:
        try:
            loaded_state = resumed_payload['state']
            state = state._replace(params=jax.device_put(loaded_state.params),
                                    opt_state=jax.device_put(loaded_state.opt_state))
            print(f"{LOG_PREFIX} --resume: restored policy/value params + optimizer state + curriculum "
                  f"(steady={governor.params.steady_difficulty:.2f} transition={governor.params.transition_difficulty:.2f} "
                  f"resilience={governor.params.resilience_difficulty:.2f} posture={governor.params.posture_difficulty:.2f} "
                  f"env-alloc={governor.params.regime_frac_steady:.2f}/{governor.params.regime_frac_transition:.2f}/"
                  f"{governor.params.regime_frac_resilience:.2f}); episode buffers re-initialize fresh.", flush=True)
        except Exception as exc:
            print(f"{LOG_PREFIX} --resume: could not restore network/optimizer state ({exc!r}) -- starting fresh instead", flush=True)
            resumed_payload = None
            resumed_step = 0
    if warm_payload is not None and resumed_payload is None:
        try:
            loaded_state = warm_payload['state']
            # Deliberately keep the freshly-created optimizer state so every
            # v5006 hypothesis starts with identical weights but no inherited
            # optimiser/curriculum momentum.
            state = state._replace(params=jax.device_put(loaded_state.params))
            print(f"{LOG_PREFIX} warm-start: restored policy/value params only; fresh optimizer/frontier retained", flush=True)
        except Exception as exc:
            print(f"{LOG_PREFIX} warm-start network restore failed ({exc!r}); using fresh weights", flush=True)
            warm_payload = None
    p = env.params_array(params)

    print(f"{LOG_PREFIX} training start run={args.run_name} backend=mjx steps={args.steps:,} envs={args.envs} rollout={args.rollout} epochs={args.epochs} device={args.device} jax_backend={jax.default_backend()}", flush=True)
    print(f"{LOG_PREFIX} obs_dim={OBS_DIM} action_dim={ACTION_DIM} regimes=steady/transition/resilience + posture", flush=True)
    print(f"{LOG_PREFIX} compiling MJX/JAX update...", flush=True)
    import jax.numpy as jnp
    obs_mean = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    obs_var = jnp.ones((OBS_DIM,), dtype=jnp.float32)
    obs_count = 1e-4  # small epsilon, standard init for a running-variance accumulator
    if warm_payload is not None and resumed_payload is None and warm_payload.get('obs_norm') is not None:
        try:
            saved_norm = warm_payload['obs_norm']
            obs_mean = jnp.asarray(saved_norm['mean'], dtype=jnp.float32)
            obs_var = jnp.asarray(saved_norm['var'], dtype=jnp.float32)
            obs_count = float(saved_norm['count'])
            print(f"{LOG_PREFIX} warm-start: restored observation normalizer (count={obs_count:,.0f})", flush=True)
        except Exception as exc:
            print(f"{LOG_PREFIX} warm-start normalizer failed ({exc!r}); using fresh stats", flush=True)
    if resumed_payload is not None and resumed_payload.get('obs_norm') is not None:
        try:
            saved_norm = resumed_payload['obs_norm']
            obs_mean = jnp.asarray(saved_norm['mean'], dtype=jnp.float32)
            obs_var = jnp.asarray(saved_norm['var'], dtype=jnp.float32)
            obs_count = float(saved_norm['count'])
            print(f"{LOG_PREFIX} --resume: restored obs normalization (count={obs_count:,.0f})", flush=True)
        except Exception as exc:
            print(f"{LOG_PREFIX} --resume: could not restore obs normalization ({exc!r}) -- using fresh running stats", flush=True)
    # KL-adaptive learning-rate controller (standard form: Schulman
    # adaptive-KL PPO, rl_games, Brax) -- a soft, multiplicative, self-
    # tuning complement to the hard KL_HARD_STOP ceiling above. KL cost
    # scales as 1/sigma^2 and sigma moves over a run, so a fixed step size
    # alone can't track that. Starts at 0.25x (not 1.0x), not 1.0x full
    # scale: entering at full lr would be a large, untested jump in
    # per-rollout policy movement before the controller has had a chance
    # to find the right scale on its own.
    LR_TARGET_UPDATE_KL = float(args.lr_target_kl)   # desired total KL from rollout policy per update
    LR_SCALE_MIN, LR_SCALE_MAX = float(args.lr_scale_min), 4.0
    LR_ADJUST = 1.5
    lr_scale = 0.25
    if resumed_payload is not None:
        try:
            lr_scale = float(resumed_payload.get('meta', {}).get('lr_scale', lr_scale))
            print(f"{LOG_PREFIX} --resume: restored lr_scale={lr_scale:.4f} "
                  f"(effective lr={args.lr * lr_scale:.3e})", flush=True)
        except Exception:
            pass
    if args.fixed_lr_scale > 0.0:
        lr_scale = float(args.fixed_lr_scale)
        print(f"{LOG_PREFIX} fixed LR controller mode: lr_scale={lr_scale:.4f}; adaptive KL LR changes disabled.", flush=True)

    # Associate rollout-derived health metrics with the exact policy,
    # normalizer, curriculum and governor state that GENERATED them.
    evaluated_obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
    rollout_params_for_metrics = governor.params_dict()
    rollout_governor_state_for_metrics = governor.state_dict()
    rollout_lr_scale_for_metrics = lr_scale
    norm_diag = {'obs_norm_update_mean_l2': 0.0, 'obs_norm_update_std_l2': 0.0,
                 'obs_norm_anchor_mean_l2': 0.0, 'obs_norm_anchor_std_l2': 0.0,
                 'obs_norm_frozen': 0.0}
    obs_anchor = None

    t0 = time.time()
    old_mean, old_std = obs_mean, obs_var ** 0.5
    state, rollout_metrics, aux, raw_obs, evaluated_state = trainer['train_update'](state, p, obs_mean, old_std, lr_scale)
    warmup_end_steps = resumed_step + args.envs * args.rollout
    norm_frozen = args.freeze_obs_norm_after_steps > 0 and resumed_step >= args.freeze_obs_norm_after_steps
    if not norm_frozen:
        obs_mean, obs_var, obs_count = update_running_stats(obs_mean, obs_var, obs_count, raw_obs)
    norm_diag['obs_norm_update_mean_l2'] = float(np.linalg.norm(np.asarray(obs_mean - old_mean)))
    norm_diag['obs_norm_update_std_l2'] = float(np.linalg.norm(np.asarray(obs_var ** 0.5 - old_std)))
    norm_diag['obs_norm_frozen'] = float(norm_frozen)
    if warmup_end_steps >= args.obs_norm_anchor_steps:
        obs_anchor = (np.asarray(obs_mean).copy(), np.asarray(obs_var ** 0.5).copy())
    state.obs.block_until_ready()
    print(f"{LOG_PREFIX} compile+first_update={time.time()-t0:.2f}s", flush=True)

    # --steps is the absolute target step count, not "how many more steps
    # to run" -- so a resumed run's warm-up update above is counted against
    # the restored step total, and log/save cadence continues from where
    # it left off instead of immediately firing at time zero.
    total_steps = resumed_step + args.envs * args.rollout
    start = time.time()
    last_log = resumed_step
    last_save = resumed_step
    last_snapshot = resumed_step
    last_seen_nan_skips = [0]  # mutable box so the log block above can update it
    # Best-known-good snapshot tracking for auto-recovery, separate from
    # the recency-based rotating snapshots. Saved only when competence
    # clears a real bar over a trailing window of updates (a median, not a
    # consecutive-update streak -- an unsmoothed per-rollout estimate means
    # "N in a row" resets on any single bad rollout and is a much higher
    # bar than it looks), not on one lucky rollout.
    BEST_SNAPSHOT_FLOOR = 0.60
    BEST_SNAPSHOT_WINDOW = 8
    best_competence_saved = 0.0
    # Unconditional high-water-mark tier, so recovery always has somewhere
    # to go even on a run that never reaches BEST_SNAPSHOT_FLOOR.
    BEST_SEEN_FLOOR = 0.10
    best_seen_competence_saved = 0.0
    comp_window: list[tuple[float, float]] = []
    # A lower-bar tier between the two above, for a run that's had a real
    # non-collapsed stretch but never a confirmed-healthy one.
    BEST_EFFORT_FLOOR = 0.20
    best_effort_competence_saved = 0.0
    recent_rewards: list[float] = []
    log_every_steps = args.envs * args.rollout * max(1, int(args.log_every))
    save_every_steps = args.envs * args.rollout * max(1, int(args.checkpoint_every))
    snapshot_every_steps = max(0, int(args.snapshot_every_steps))

    # Fixed held-out evaluator. Same deterministic cohorts at every checkpoint
    # and every branch, independent of whatever curriculum happens to be live.
    eval_suites = {
        'core': dict(steady_difficulty=0.0, transition_difficulty=0.0, resilience_difficulty=0.0,
                     posture_difficulty=0.0, regime_frac_steady=1.0, regime_frac_transition=0.0,
                     regime_frac_resilience=0.0, regime_command_frac=0.5, core_rehearsal_frac=0.0),
        'stress30': dict(steady_difficulty=0.30, transition_difficulty=0.30, resilience_difficulty=0.30,
                         posture_difficulty=0.20, regime_frac_steady=1/3, regime_frac_transition=1/3,
                         regime_frac_resilience=1/3, regime_command_frac=0.5, core_rehearsal_frac=0.0),
        # Fixed capability rulers. dynamic50 emphasises command changes;
        # resilience50 emphasises disturbance recovery. Their distributions are
        # constant across checkpoints and branches.
        'dynamic50': dict(steady_difficulty=0.45, transition_difficulty=0.60, resilience_difficulty=0.25,
                          posture_difficulty=0.25, regime_frac_steady=0.20, regime_frac_transition=0.65,
                          regime_frac_resilience=0.15, regime_command_frac=0.75, core_rehearsal_frac=0.0),
        'resilience50': dict(steady_difficulty=0.35, transition_difficulty=0.35, resilience_difficulty=0.60,
                             posture_difficulty=0.40, regime_frac_steady=0.15, regime_frac_transition=0.20,
                             regime_frac_resilience=0.65, regime_command_frac=0.65, core_rehearsal_frac=0.0),
        'dynamic80': dict(steady_difficulty=0.75, transition_difficulty=0.85, resilience_difficulty=0.35,
                          posture_difficulty=0.35, regime_frac_steady=0.15, regime_frac_transition=0.70,
                          regime_frac_resilience=0.15, regime_command_frac=0.90, core_rehearsal_frac=0.0),
        'combo70': dict(steady_difficulty=0.70, transition_difficulty=0.72, resilience_difficulty=0.30,
                        posture_difficulty=0.35, regime_frac_steady=0.35, regime_frac_transition=0.50,
                        regime_frac_resilience=0.15, regime_command_frac=0.90, core_rehearsal_frac=0.0),
        'resilience75': dict(steady_difficulty=0.55, transition_difficulty=0.55, resilience_difficulty=0.80,
                             posture_difficulty=0.55, regime_frac_steady=0.10, regime_frac_transition=0.20,
                             regime_frac_resilience=0.70, regime_command_frac=0.80, core_rehearsal_frac=0.0),
    }
    eval_fields = ('survival','first_fall_step_mean','vx_mae','vy_mae','yaw_mae','gravity_z','foot_mid_cone','step_evidence','support')
    latest_eval = {f'eval_{suite}_{field}': -1.0 for suite in eval_suites for field in eval_fields}
    latest_eval.update({f'eval_robust40_{field}': -1.0 for field in eval_fields})
    latest_eval.update({f'eval_growth110_{field}': -1.0 for field in eval_fields})
    latest_eval.update({f'eval_growthcombo110_{field}': -1.0 for field in eval_fields})
    latest_eval['eval_policy_step'] = -1.0
    latest_eval['eval_broad_score'] = -1e9
    latest_eval['eval_broad_tail3'] = -1e9
    last_eval_step = resumed_step
    best_broad_eval_score = -1e9
    broad_eval_history: list[float] = []

    def _mean_eval_summaries(items: list[dict[str, float]]) -> dict[str, float]:
        if not items:
            return {}
        return {k: float(sum(float(x[k]) for x in items) / len(items)) for k in items[0]}

    def _broad_eval_score(out: dict[str, float]) -> float:
        # Survival remains the primary ruler, but command obedience now has a
        # real vote. In particular combo survival cannot win by ignoring yaw.
        surv = (0.05*out.get('eval_core_survival',0.0) +
                0.08*out.get('eval_stress30_survival',0.0) +
                0.10*out.get('eval_dynamic50_survival',0.0) +
                0.16*out.get('eval_dynamic80_survival',0.0) +
                0.12*out.get('eval_combo70_survival',0.0) +
                0.12*out.get('eval_resilience50_survival',0.0) +
                0.14*out.get('eval_resilience75_survival',0.0) +
                0.11*out.get('eval_robust40_survival',0.0) +
                0.06*out.get('eval_growth110_survival',0.0) +
                0.06*out.get('eval_growthcombo110_survival',0.0))
        track_pen = (0.10*min(1.0, out.get('eval_dynamic80_vx_mae',1.0)/0.12) +
                     0.10*min(1.0, out.get('eval_dynamic80_yaw_mae',2.0)/0.40) +
                     0.08*min(1.0, out.get('eval_combo70_vy_mae',1.0)/0.12) +
                     0.10*min(1.0, out.get('eval_combo70_yaw_mae',2.0)/0.40) +
                     0.06*min(1.0, out.get('eval_growthcombo110_yaw_mae',2.0)/0.55))
        return float(surv - track_pen)

    def maybe_fixed_eval(step: int, force: bool = False) -> None:
        nonlocal last_eval_step, latest_eval, best_broad_eval_score
        every = max(0, int(args.eval_every_steps))
        if not force and (every <= 0 or step - last_eval_step < every):
            return
        base_key = jax.random.PRNGKey(int(args.eval_seed))
        cohorts = max(1, int(args.eval_cohorts))
        out = {'eval_policy_step': float(step)}

        def run_suite(name: str, env_obj, trainer_obj, ep: dict, salt: int) -> dict[str, float]:
            summaries=[]
            for cohort in range(cohorts):
                ek = jax.random.fold_in(base_key, salt + 10007*cohort)
                em, ff = trainer_obj['evaluate_rollout'](
                    evaluated_state.params, env_obj.params_array(ep),
                    evaluated_obs_norm['mean'], evaluated_obs_norm['var'] ** 0.5,
                    ek, int(args.eval_horizon))
                ff.block_until_ready()
                summaries.append(summarize_fixed_eval(em, ff, int(args.eval_horizon)))
            return _mean_eval_summaries(summaries)

        for idx, (name, ep) in enumerate(eval_suites.items()):
            etr = combo_eval_trainer if name == 'combo70' else fixed_eval_trainer
            eenv = combo_eval_env if name == 'combo70' else fixed_eval_env
            summary = run_suite(name, eenv, etr, ep, 100 + idx)
            out.update({f'eval_{name}_{k}': v for k, v in summary.items()})

        robust_ep = dict(steady_difficulty=0.40, transition_difficulty=0.45, resilience_difficulty=0.50,
                         posture_difficulty=0.35, regime_frac_steady=0.25, regime_frac_transition=0.30,
                         regime_frac_resilience=0.45, regime_command_frac=0.65, core_rehearsal_frac=0.0,
                         athletic_rehearsal_frac=0.0)
        rsummary = run_suite('robust40', robust_eval_env, robust_eval_trainer, robust_ep, 404)
        out.update({f'eval_robust40_{k}': v for k, v in rsummary.items()})

        # Beyond-v5005 fixed growth exams. Same evaluator forever; only the
        # policy changes. One is axis/rectangular, one is all-mixed p-norm.
        growth_ep = dict(steady_difficulty=1.0, transition_difficulty=0.80, resilience_difficulty=0.30,
                         posture_difficulty=0.35, regime_frac_steady=0.55, regime_frac_transition=0.35,
                         regime_frac_resilience=0.10, regime_command_frac=0.90, core_rehearsal_frac=0.0,
                         athletic_rehearsal_frac=0.0)
        gsummary = run_suite('growth110', growth_eval_env, growth_eval_trainer, growth_ep, 811)
        gcsummary = run_suite('growthcombo110', growth_combo_env, growth_combo_trainer, growth_ep, 919)
        out.update({f'eval_growth110_{k}': v for k, v in gsummary.items()})
        out.update({f'eval_growthcombo110_{k}': v for k, v in gcsummary.items()})

        out['eval_broad_score'] = _broad_eval_score(out)
        broad_eval_history.append(float(out['eval_broad_score']))
        out['eval_broad_tail3'] = float(sum(broad_eval_history[-3:]) / len(broad_eval_history[-3:]))
        latest_eval.update(out)
        append_metrics_csv(run_dir / 'fixed_eval.csv', {'step': step, **out})
        last_eval_step = step

        # Keep the best broad held-out checkpoint, independent of latest/final.
        # This is the antidote to v5006's 130-150M specialization collapse.
        if out['eval_broad_tail3'] > best_broad_eval_score:
            best_broad_eval_score = out['eval_broad_tail3']
            save_checkpoint(run_dir / 'checkpoints' / 'best_broad_eval.pkl', evaluated_state,
                            rollout_params_for_metrics,
                            {'step': step, 'version': version, 'best_broad_eval_score': best_broad_eval_score, 'instant_broad_eval_score': out['eval_broad_score'],
                             'evaluated_policy': True}, evaluated_obs_norm, governor.state_dict())

        print(f"{LOG_PREFIX} fixed-eval step={step:,} score={out['eval_broad_score']:.3f} tail3={out['eval_broad_tail3']:.3f} "
              f"core={out['eval_core_survival']:.3f} dyn80={out['eval_dynamic80_survival']:.3f} "
              f"combo70={out['eval_combo70_survival']:.3f}/{out['eval_combo70_yaw_mae']:.3f}yaw "
              f"res75={out['eval_resilience75_survival']:.3f} robust40={out['eval_robust40_survival']:.3f} "
              f"growth={out['eval_growth110_survival']:.3f}/{out['eval_growthcombo110_survival']:.3f}", flush=True)

    def augment_rollout_row(row: dict[str, Any], aux_now: dict[str, Any]) -> None:
        """Attach diagnostics with the exact rollout provenance.

        `row` describes the rollout collected by evaluated_state, under
        rollout_params_for_metrics/evaluated_obs_norm -- not the policy and
        curriculum that exist after the gradient/governor update. Keeping
        these aligned is what makes collapse forensics trustworthy.
        """
        for axis in ('steady', 'transition', 'resilience', 'posture'):
            cur = float(rollout_params_for_metrics.get(f'{axis}_difficulty', 0.0))
            born = float(row.get(f'birth_{axis}_d', -1.0))
            row[f'rollout_{axis}_d'] = cur
            row[f'birth_lag_{axis}'] = (cur - born) if born >= 0.0 else -1.0
        # What allocation generated the samples, distinct from the governor's
        # allocation for NEXT update and from measured per-step occupancy.
        for rn in ('steady', 'transition', 'resilience'):
            row[f'rollout_regime_frac_{rn}'] = float(rollout_params_for_metrics.get(f'regime_frac_{rn}', 0.0))
        row['rollout_core_rehearsal_frac'] = float(rollout_params_for_metrics.get('core_rehearsal_frac', 0.0))
        row['rollout_athletic_rehearsal_frac'] = float(rollout_params_for_metrics.get('athletic_rehearsal_frac', 0.0))
        row['rollout_lr_scale'] = float(rollout_lr_scale_for_metrics)
        row['obs_norm_mode'] = args.obs_norm_mode
        row['logp_on_raw_action'] = float(bool(args.logp_on_raw_action))
        row['fixed_lr_scale'] = float(args.fixed_lr_scale)
        row.update(norm_diag)
        row.update(latest_eval)
        scalar_aux = {
            'ppo_kl_stopped_frac': 'kl_stopped',
            'explained_variance': 'explained_variance',
            'exploration_std_mean': 'exploration_std_mean',
            'ppo_approx_kl_final': 'approx_kl_final',
            'ppo_action_clip_frac': 'action_clip_frac',
            'ppo_adv_tail_frac': 'adv_tail_frac',
            'ppo_grad_norm_policy_preclip': 'grad_norm_policy_preclip',
            'ppo_grad_norm_value_preclip': 'grad_norm_value_preclip',
            'ppo_update_norm_policy': 'update_norm_policy',
            'ppo_update_norm_value': 'update_norm_value',
        }
        for dst, src in scalar_aux.items():
            row[dst] = float(np.asarray(aux_now.get(src, 0.0)))
        # Effective LR belongs to the rollout/update pair, not the LR scale
        # that may be selected for the NEXT update by the controller below.
        row['ppo_lr_scale'] = float(np.asarray(aux_now.get('lr_scale', rollout_lr_scale_for_metrics)))
        row['ppo_effective_lr'] = float(args.lr) * row['ppo_lr_scale']
        for ji in range(ACTION_DIM):
            row[f'ppo_action_clip_j{ji}'] = float(np.asarray(aux_now.get(f'action_clip_j{ji}', 0.0)))
        for rn in ('steady', 'transition', 'resilience'):
            for key in ('adv_mean', 'adv_std', 'value_ev'):
                row[f'ppo_{key}_{rn}'] = float(np.asarray(aux_now.get(f'{key}_{rn}', 0.0)))

    while total_steps < args.steps:
        maybe_fixed_eval(total_steps)
        row = summarize_rollout(rollout_metrics, rollout_params_for_metrics, combo_p=args.combo_p)
        reward_mean = float(np.asarray(aux.get('rollout_reward_mean', 0.0)))
        recent_rewards.append(reward_mean)
        recent_rewards = recent_rewards[-64:]
        # Add PPO, normalisation, evaluator and birth-cohort diagnostics
        # before the governor sees the row. Everything here describes the
        # SAME rollout/policy/task distribution.
        augment_rollout_row(row, aux)
        curriculum_frozen = (args.freeze_curriculum_after_steps > 0 and
                             total_steps >= args.freeze_curriculum_after_steps)
        frozen_params = governor.params_dict() if curriculum_frozen else None
        report = governor.update(row)
        if curriculum_frozen and frozen_params is not None:
            # Keep the governor's observational state (EMAs, collapse
            # detection, competence) alive, but prevent it from moving either
            # difficulty or sampling distribution. This makes a clean causal
            # intervention: PPO keeps learning while the task distribution is
            # stationary from this point onward.
            governor.params = RegimeParams(**frozen_params)
            for axis in ('steady', 'transition', 'resilience', 'posture'):
                report[f'{axis}_difficulty'] = float(frozen_params[f'{axis}_difficulty'])
                report[f'event_{axis}'] = 'hold_curriculum_frozen'
            for rn in ('steady', 'transition', 'resilience'):
                report[f'regime_frac_{rn}'] = float(frozen_params[f'regime_frac_{rn}'])
        report['curriculum_frozen'] = float(curriculum_frozen)
        params = governor.params_dict()
        p = env.params_array(params)

        # Best-known-good snapshot tracking (see the init comment above
        # for why this is separate from the routine rotating snapshots).
        # Sustained health required, not a single good rollout. Competence
        # is three numbers (one per regime), and a snapshot only counts as
        # genuinely good if it's good ACROSS ALL THREE, not just whichever
        # regime happens to be easiest right now -- so this uses the
        # MINIMUM of the three, the same "judge the worst case, not the
        # average" principle core_competence already applies to the
        # original seed-band walk.
        overall_competence = min(report['competence_steady'], report['competence_transition'], report['competence_resilience'])
        comp_window.append((overall_competence, float(report['core_competence'])))
        comp_window = comp_window[-BEST_SNAPSHOT_WINDOW:]
        if len(comp_window) >= BEST_SNAPSHOT_WINDOW:
            comp_med = float(np.median([c for c, _ in comp_window]))
            core_med = float(np.median([c for _, c in comp_window]))
        else:
            comp_med = core_med = -1.0  # not enough history to judge yet

        def _save_tier(filename: str, label: str, extra: str = '') -> None:
            # `row` was generated by evaluated_state BEFORE this rollout's PPO
            # update and by rollout_params_for_metrics BEFORE governor.update.
            # Saving post-update `state` here would attribute old-policy health
            # to unmeasured new weights -- exactly the temporal checkpoint bug
            # this generation is designed to eliminate.
            save_checkpoint(run_dir / 'checkpoints' / filename, evaluated_state, rollout_params_for_metrics,
                             {'step': total_steps, 'run_name': args.run_name,
                              'competence': comp_med, 'core_competence': core_med,
                              'lr_scale': rollout_lr_scale_for_metrics,
                              'evaluated_policy': True}, evaluated_obs_norm,
                             governor_meta=rollout_governor_state_for_metrics)
            print(f"{LOG_PREFIX} new {label} snapshot saved at step={total_steps:,} "
                  f"(median-of-{BEST_SNAPSHOT_WINDOW} competence={comp_med:.2f}, "
                  f"core={core_med:.2f}){extra}", flush=True)

        if comp_med >= BEST_SNAPSHOT_FLOOR and core_med >= BEST_SNAPSHOT_FLOOR and comp_med > best_competence_saved:
            best_competence_saved = comp_med
            _save_tier('best_known_good.pkl', 'best-known-good')

        # Optional early stop, for building a shared bootstrap checkpoint
        # that a later batch of runs all --resume from, instead of every
        # run separately re-paying the early "learn to walk at all" cost.
        # Reuses BEST_SNAPSHOT_FLOOR's existing criterion (median-of-8
        # competence AND core_competence both >= 0.60 -- holistic: survival,
        # cone accuracy and tracking error together, not just "didn't fall")
        # rather than inventing a second, cruder one. Gated behind
        # --min-steps-before-stop so a lucky early window can't end the run
        # before the curriculum has even reached the later regimes.
        if (args.stop_after_best_known_good and total_steps >= args.min_steps_before_stop
                and best_competence_saved >= BEST_SNAPSHOT_FLOOR):
            print(f"{LOG_PREFIX} --stop-after-best-known-good: stopping at step={total_steps:,} "
                  f"(>= --min-steps-before-stop={args.min_steps_before_stop:,}, "
                  f"best_known_good.pkl saved at competence={best_competence_saved:.2f}). "
                  f"Would otherwise have continued to steps={args.steps:,}.", flush=True)
            break

        # A lower bar than confirmed-good, for a run with a real but less sustained improvement.
        if comp_med >= BEST_EFFORT_FLOOR and core_med >= BEST_EFFORT_FLOOR and comp_med > best_effort_competence_saved:
            best_effort_competence_saved = comp_med
            _save_tier('best_effort.pkl', 'best-effort',
                       f" -- below the confirmed-good bar ({BEST_SNAPSHOT_FLOOR:.2f}) but the best "
                       f"sustained state seen so far this run.")

        # Unconditional high-water mark, so recovery is never empty-handed.
        if comp_med >= BEST_SEEN_FLOOR and comp_med > best_seen_competence_saved:
            best_seen_competence_saved = comp_med
            _save_tier('best_seen.pkl', 'best-seen (high-water mark)',
                       " -- not a health claim, just the best this run has managed.")

        # Auto-recovery. The governor can only see the symptom (stepping-
        # collapse sustained past ordinary retreat's ability to fix it); it
        # has no access to network weights, so the actual rollback happens
        # here. This guards against a real failure mode: a policy getting
        # stuck for tens of millions of steps after apparently learning to
        # balance-and-avoid-stepping rather than walk, with ordinary
        # per-regime retreat unable to pull it back out on its own.
        if report['needs_recovery']:
            # Prefer the confirmed-good tier; fall back to the best-effort
            # tier when this run has never cleared BEST_SNAPSHOT_FLOOR.
            # Without this, a run that never reached that bar would hit
            # `needs_recovery` and have nothing to recover to, silently
            # reducing to "continuing with ordinary retreat only" forever.
            best_path = run_dir / 'checkpoints' / 'best_known_good.pkl'
            recovery_tier = 'best-known-good'
            if not best_path.exists():
                best_path = run_dir / 'checkpoints' / 'best_effort.pkl'
                recovery_tier = 'best-effort'
            if not best_path.exists():
                # Third tier, the high-water mark. See its save block above.
                best_path = run_dir / 'checkpoints' / 'best_seen.pkl'
                recovery_tier = 'best-seen'
            recovered = False
            if best_path.exists():
                try:
                    with best_path.open('rb') as f:
                        best_payload = pickle.load(f)
                    if payload_params_are_finite(best_payload):
                        loaded_state = best_payload['state']
                        state = state._replace(params=jax.device_put(loaded_state.params),
                                                opt_state=jax.device_put(loaded_state.opt_state))
                        saved_curriculum = best_payload.get('curriculum_params', {}) or {}
                        known_fields = {f.name for f in dataclass_fields(RegimeParams)}
                        # Full temporal rewind: network, optimiser, curriculum,
                        # governor histories/EMAs, observation normaliser and LR
                        # controller. Older compact checkpoints use the fallback.
                        gmeta = best_payload.get('governor_meta') or {}
                        if 'params' in gmeta or 'fall_ema' in gmeta:
                            governor.load_state_dict(gmeta)
                        else:
                            governor.params = RegimeParams(**{k: v for k, v in saved_curriculum.items() if k in known_fields})
                            if 'updates_done_total' in gmeta:
                                try:
                                    governor._updates_done_total = int(gmeta['updates_done_total'])
                                except (TypeError, ValueError):
                                    pass
                        if best_payload.get('obs_norm') is not None:
                            obs_mean = jnp.asarray(best_payload['obs_norm']['mean'], dtype=jnp.float32)
                            obs_var = jnp.asarray(best_payload['obs_norm']['var'], dtype=jnp.float32)
                            obs_count = float(best_payload['obs_norm']['count'])
                        try:
                            lr_scale = float(best_payload.get('meta', {}).get('lr_scale', lr_scale))
                        except (TypeError, ValueError):
                            pass
                        if args.fixed_lr_scale > 0.0:
                            lr_scale = float(args.fixed_lr_scale)
                        # Revert to the previous good state AND shift the RNG --
                        # rolling back only params/opt_state/curriculum would
                        # leave every env's live physical state exactly as the
                        # COLLAPSED policy left it (mid-fall, tangled, whatever
                        # it was), for the just-reverted policy to inherit.
                        # env.reset() alone wouldn't fix this either: it always
                        # resets from the same fixed self.key0, so calling it
                        # naively here would replay IDENTICAL initial conditions
                        # every time recovery fires, not shift anything.
                        # jax.random.fold_in with total_steps (monotonically
                        # increasing, so never coincidentally repeats) derives a
                        # genuinely different key each time, deterministically.
                        fresh_key = jax.random.fold_in(env.key0, total_steps)
                        fresh_params_array = env.params_array(governor.params_dict())
                        fresh_env_state = env.reset_jit(fresh_key, fresh_params_array)
                        state = state._replace(env_state=fresh_env_state, obs=env.obs(fresh_env_state))
                        best_step = best_payload.get('meta', {}).get('step', '?')
                        print(f"{LOG_PREFIX} AUTO-RECOVERY: stepping-collapse sustained for "
                              f"{int(report['stepping_collapse_streak'])} updates despite repeated retreats -- "
                              f"rolled back network + curriculum to the {recovery_tier} snapshot from "
                              f"step={best_step} (competence={best_payload.get('meta', {}).get('competence', '?')}). "
                              f"Every env RESET with a fresh RNG derived from step={total_steps:,} (not the run's "
                              f"original seed) -- a genuinely different attempt, not a replay of the same luck "
                              f"that led here, and no env inherits whatever physical state the collapsed policy "
                              f"left it in.", flush=True)
                        recovered = True
                    else:
                        print(f"{LOG_PREFIX} AUTO-RECOVERY triggered but {best_path.name} contains non-finite "
                              f"params -- skipping recovery, continuing with ordinary retreat only.", flush=True)
                except Exception as exc:
                    print(f"{LOG_PREFIX} AUTO-RECOVERY triggered but failed to load {best_path.name} ({exc!r}) "
                          f"-- continuing with ordinary retreat only.", flush=True)
            else:
                print(f"{LOG_PREFIX} AUTO-RECOVERY triggered (stepping-collapse for "
                      f"{int(report['stepping_collapse_streak'])} updates) but no best-known-good or "
                      f"best-effort or best-seen snapshot exists yet (median-of-{BEST_SNAPSHOT_WINDOW} competence "
                      f"has never reached even {BEST_SEEN_FLOOR:.2f} this run) -- continuing with ordinary "
                      f"retreat only.", flush=True)
            # Only clear the streak when a rollback actually happened.
            # Clearing it unconditionally would mean a FAILED attempt (e.g.
            # no snapshot to load, the case just above) forces a blind wait
            # for a whole new streak before trying again, even though
            # nothing about the situation changed. Leaving the streak alone
            # lets update()'s own normal per-update bookkeeping (+1 if
            # still collapsed, reset to 0 the moment a rollout genuinely
            # isn't) decide when to stop retrying.
            if recovered:
                governor.note_recovery_performed()
            params = governor.params_dict()
            p = env.params_array(params)
            if recovered:
                # don't immediately re-save a "best" from the state we just
                # loaded: drop the trailing window so the next save has to be
                # earned from BEST_SNAPSHOT_WINDOW fresh post-recovery rollouts.
                comp_window = []

        if total_steps - last_log >= log_every_steps:
            last_log = total_steps
            elapsed = time.time() - start
            # fps must be steps-*this-process* / elapsed-*this-process*.
            # Using absolute total_steps (which starts at resumed_step, not
            # 0, after --resume) divided by this process's short elapsed
            # time would produce nonsense fps spikes right after a resume.
            fps = (total_steps - resumed_step) / max(1e-9, elapsed)
            # Surfaces optax.apply_if_finite's running counter, so a
            # skipped (non-finite) gradient update shows up in the log
            # instead of silently vanishing. Any nonzero value here means
            # something upstream produced a NaN/Inf gradient this run --
            # worth investigating even though the skip itself kept the
            # network safe.
            nan_grad_skips = int(np.asarray(state.opt_state.total_notfinite))
            if nan_grad_skips > last_seen_nan_skips[0]:
                print(f"{LOG_PREFIX} WARNING: optimizer skipped a non-finite gradient update "
                      f"(total_notfinite={nan_grad_skips}, was {last_seen_nan_skips[0]}) at step={total_steps:,}. "
                      f"Network was NOT corrupted -- the update was discarded instead of applied. "
                      f"Worth checking whether this correlates with a specific curriculum/task extreme.", flush=True)
                last_seen_nan_skips[0] = nan_grad_skips
            # Surfaces the KL-early-stopping guard firing. A nonzero
            # fraction here means at least one minibatch this update
            # exceeded KL_HARD_STOP and further updates were skipped for
            # the rest of this rollout's epochs -- the policy was kept
            # closer to the data it was actually collected under. Worth
            # watching: if it fires repeatedly and steadily, that's a sign
            # of real, ongoing instability (lr/epochs/minibatches too
            # aggressive for the current reward/task regime), not a one-off.
            kl_stopped_frac = float(np.asarray(aux.get('kl_stopped', 0.0)))
            if kl_stopped_frac > 0.0:
                print(f"{LOG_PREFIX} WARNING: KL early-stopping fired this update (stopped_frac={kl_stopped_frac:.2f}, "
                      f"approx_kl={float(np.asarray(aux.get('approx_kl', 0.0))):.4f}) at step={total_steps:,}. "
                      f"Further updates were skipped this rollout to avoid walking the policy too far from "
                      f"the data it was collected under. Fine as a one-off; investigate lr/epochs/minibatches "
                      f"if this keeps firing.", flush=True)
            log_row = {
                'step': total_steps, 'elapsed_s': elapsed, 'steps_per_sec': fps,
                'reward_mean': float(np.mean(recent_rewards)), **row, **report,
                'backend': 'mjx', 'jax_backend': jax.default_backend(), 'num_envs': args.envs, 'rollout': args.rollout,
                'ppo_policy_loss': float(np.asarray(aux.get('policy_loss', 0.0))),
                'ppo_value_loss': float(np.asarray(aux.get('value_loss', 0.0))),
                'ppo_entropy': float(np.asarray(aux.get('entropy', 0.0))),
                'ppo_approx_kl': float(np.asarray(aux.get('approx_kl', 0.0))),
                'ppo_policy_loss_finite_frac': float(np.asarray(aux.get('policy_loss_finite', 1.0))),
                'ppo_value_loss_finite_frac': float(np.asarray(aux.get('value_loss_finite', 1.0))),
                'ppo_kl_stopped_frac': float(np.asarray(aux.get('kl_stopped', 0.0))),
                'ppo_approx_kl_final': float(np.asarray(aux.get('approx_kl_final', 0.0))),
                'ppo_lr_scale': float(np.asarray(aux.get('lr_scale', 1.0))),
                'ppo_effective_lr': float(args.lr) * float(np.asarray(aux.get('lr_scale', 1.0))),
                'ppo_action_clip_frac': float(np.asarray(aux.get('action_clip_frac', 0.0))),
                'ppo_adv_tail_frac': float(np.asarray(aux.get('adv_tail_frac', 0.0))),
                'explained_variance': float(np.asarray(aux.get('explained_variance', 0.0))),
                'exploration_std_mean': float(np.asarray(aux.get('exploration_std_mean', 0.0))),
                'nan_grad_skips_total': nan_grad_skips,
            }
            append_metrics_csv(run_dir / 'metrics.csv', log_row)
            write_json(run_dir / 'envelope_summary.json', {'step': total_steps, 'params': params, 'report': report, 'last_row': log_row})
            print(format_line(version, total_steps, float(np.mean(recent_rewards)), row, report, fps, nan_grad_skips), flush=True)

        if save_every_steps > 0 and total_steps - last_save >= save_every_steps:
            last_save = total_steps
            obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
            # apply_if_finite should make this unreachable in practice (a
            # non-finite update is skipped before it ever reaches
            # state.params), but checking again here, right at the point of
            # writing the checkpoint, costs almost nothing and closes the
            # loop completely rather than relying on a single guard.
            params_ok = all(bool(np.all(np.isfinite(np.asarray(leaf))))
                             for leaf in jax.tree_util.tree_leaves(state.params))
            if params_ok:
                save_checkpoint(run_dir / 'checkpoints' / 'latest_mjx.pkl', state, params,
                                 {'step': total_steps, 'run_name': args.run_name, 'lr_scale': lr_scale}, obs_norm,
                                 governor_meta=governor.state_dict())
            else:
                print(f"{LOG_PREFIX} WARNING: refusing to overwrite checkpoints/latest_mjx.pkl at step={total_steps:,} "
                      f"-- state.params contains non-finite values despite the optimizer guard. "
                      f"This should not be reachable; please report it. Falling back to the rotating "
                      f"snapshots below for recovery.", flush=True)

        if snapshot_every_steps > 0 and total_steps - last_snapshot >= snapshot_every_steps:
            last_snapshot = total_steps
            obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
            save_rotating_snapshot(run_dir / 'checkpoints', state, params,
                                    {'step': total_steps, 'run_name': args.run_name, 'lr_scale': lr_scale},
                                    obs_norm, total_steps, keep=max(0, int(args.snapshot_keep)),
                                    governor_meta=governor.state_dict())

        # Freeze provenance for the rollout we are ABOUT to collect. The
        # next loop iteration's row must be attributed to these exact values,
        # even though PPO/governor/normalizer may all move afterwards.
        rollout_params_for_metrics = governor.params_dict()
        rollout_governor_state_for_metrics = governor.state_dict()
        rollout_lr_scale_for_metrics = float(lr_scale)
        evaluated_obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
        old_mean, old_std = obs_mean, obs_var ** 0.5
        state, rollout_metrics, aux, raw_obs, evaluated_state = trainer['train_update'](
            state, p, obs_mean, old_std, lr_scale)
        # Controller step. `approx_kl_final` is the last minibatch of the
        # last epoch, i.e. total divergence from the policy this rollout
        # was collected under -- the quantity a trust region is about. The
        # dead band (0.5x-2x of target) stops the lr oscillating on noise.
        kl_final = float(np.asarray(aux.get('approx_kl_final', 0.0)))
        if args.fixed_lr_scale > 0.0:
            lr_scale = float(args.fixed_lr_scale)
        elif np.isfinite(kl_final):
            if kl_final > 2.0 * LR_TARGET_UPDATE_KL:
                lr_scale /= LR_ADJUST
            elif kl_final < 0.5 * LR_TARGET_UPDATE_KL:
                lr_scale *= LR_ADJUST
            lr_scale = float(min(LR_SCALE_MAX, max(LR_SCALE_MIN, lr_scale)))
        # Optional baseline pull, separate from and much weaker than the
        # reactive step above. An adaptive learning-rate controller that
        # only reacts to current KL can get stuck at an arbitrarily low,
        # self-consistent rate: a policy that's stopped doing much of
        # anything (collapsed, not actively thrashing) can show LOW
        # per-step KL while performing badly, landing lr_scale in the dead
        # band indefinitely with no path back up tied to actual task
        # performance. This pulls a constant FRACTION of the remaining gap
        # to 1.0 (== --lr unmodified, the natural "no correction active"
        # point) every update, applied AFTER the reactive step so it never
        # fights an active shrink/grow, only nudges when nothing else is
        # moving it. 0.0 (default) = off.
        if args.fixed_lr_scale <= 0.0 and args.lr_scale_baseline_pull > 0.0:
            lr_scale += args.lr_scale_baseline_pull * (1.0 - lr_scale)
            lr_scale = float(min(LR_SCALE_MAX, max(LR_SCALE_MIN, lr_scale)))

        # Observation-normalizer experiment. A curriculum can change the
        # meaning of normalised coordinates even with frozen policy weights;
        # branches may freeze this transformation after a chosen step.
        norm_frozen = (args.freeze_obs_norm_after_steps > 0 and
                       total_steps >= args.freeze_obs_norm_after_steps)
        if not norm_frozen:
            obs_mean, obs_var, obs_count = update_running_stats(obs_mean, obs_var, obs_count, raw_obs)
        new_std = obs_var ** 0.5
        norm_diag = {
            'obs_norm_update_mean_l2': float(np.linalg.norm(np.asarray(obs_mean - old_mean))),
            'obs_norm_update_std_l2': float(np.linalg.norm(np.asarray(new_std - old_std))),
            'obs_norm_frozen': float(norm_frozen),
            'obs_norm_anchor_mean_l2': 0.0,
            'obs_norm_anchor_std_l2': 0.0,
        }
        next_total_steps = total_steps + args.envs * args.rollout
        if obs_anchor is None and next_total_steps >= args.obs_norm_anchor_steps:
            obs_anchor = (np.asarray(obs_mean).copy(), np.asarray(new_std).copy())
        if obs_anchor is not None:
            norm_diag['obs_norm_anchor_mean_l2'] = float(np.linalg.norm(np.asarray(obs_mean) - obs_anchor[0]))
            norm_diag['obs_norm_anchor_std_l2'] = float(np.linalg.norm(np.asarray(new_std) - obs_anchor[1]))
        state.obs.block_until_ready()
        total_steps = next_total_steps

    # final write. Evaluate and report the exact policy that generated the
    # final rollout before allowing that rollout to update governor stats.
    maybe_fixed_eval(total_steps, force=True)
    row = summarize_rollout(rollout_metrics, rollout_params_for_metrics, combo_p=args.combo_p)
    augment_rollout_row(row, aux)
    curriculum_frozen = (args.freeze_curriculum_after_steps > 0 and
                         total_steps >= args.freeze_curriculum_after_steps)
    frozen_params = governor.params_dict() if curriculum_frozen else None
    report = governor.update(row)
    if curriculum_frozen and frozen_params is not None:
        governor.params = RegimeParams(**frozen_params)
        for axis in ('steady', 'transition', 'resilience', 'posture'):
            report[f'{axis}_difficulty'] = float(frozen_params[f'{axis}_difficulty'])
            report[f'event_{axis}'] = 'hold_curriculum_frozen'
        for rn in ('steady', 'transition', 'resilience'):
            report[f'regime_frac_{rn}'] = float(frozen_params[f'regime_frac_{rn}'])
    report['curriculum_frozen'] = float(curriculum_frozen)
    elapsed = time.time() - start
    fps = (total_steps - resumed_step) / max(1e-9, elapsed)
    nan_grad_skips = int(np.asarray(state.opt_state.total_notfinite))
    # This summary row carries every ppo_* diagnostic field the main
    # loop's log_row has (policy/value loss, approx_kl, the finite-fraction
    # and kl_stopped_frac safety diagnostics) -- the CSV's very last row is
    # exactly the moment you'd most want full information. Mirrors
    # information, silently had blanks for all of them. Mirrors log_row's
    # construction now instead of its own hand-trimmed subset.
    final_row = {'step': total_steps, 'elapsed_s': elapsed, 'steps_per_sec': fps, 'reward_mean': float(np.mean(recent_rewards) if recent_rewards else 0.0), **row, **report,
                 'backend': 'mjx', 'jax_backend': jax.default_backend(), 'nan_grad_skips_total': nan_grad_skips,
                 'ppo_policy_loss': float(np.asarray(aux.get('policy_loss', 0.0))),
                 'ppo_value_loss': float(np.asarray(aux.get('value_loss', 0.0))),
                 'ppo_entropy': float(np.asarray(aux.get('entropy', 0.0))),
                 'ppo_approx_kl': float(np.asarray(aux.get('approx_kl', 0.0))),
                 'ppo_policy_loss_finite_frac': float(np.asarray(aux.get('policy_loss_finite', 1.0))),
                 'ppo_value_loss_finite_frac': float(np.asarray(aux.get('value_loss_finite', 1.0))),
                 'ppo_kl_stopped_frac': float(np.asarray(aux.get('kl_stopped', 0.0))),
                 'ppo_approx_kl_final': float(np.asarray(aux.get('approx_kl_final', 0.0))),
                 'ppo_lr_scale': float(np.asarray(aux.get('lr_scale', 1.0))),
                 'ppo_effective_lr': float(args.lr) * float(np.asarray(aux.get('lr_scale', 1.0))),
                 'ppo_action_clip_frac': float(np.asarray(aux.get('action_clip_frac', 0.0))),
                 'ppo_adv_tail_frac': float(np.asarray(aux.get('adv_tail_frac', 0.0))),
                 'explained_variance': float(np.asarray(aux.get('explained_variance', 0.0))),
                 'exploration_std_mean': float(np.asarray(aux.get('exploration_std_mean', 0.0)))}
    append_metrics_csv(run_dir / 'metrics.csv', final_row)
    write_json(run_dir / 'envelope_summary.json', {'step': total_steps, 'params': governor.params_dict(), 'report': report, 'last_row': final_row})
    save_checkpoint(run_dir / 'checkpoints' / 'latest_mjx.pkl', state, governor.params_dict(),
                     {'step': total_steps, 'run_name': args.run_name, 'lr_scale': lr_scale},
                     {'mean': obs_mean, 'var': obs_var, 'count': obs_count},
                     governor_meta=governor.state_dict())
    print(format_line(version, total_steps, final_row['reward_mean'], row, report, fps, nan_grad_skips), flush=True)
    print(f'{LOG_PREFIX} training complete', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='EMU PPO trainer')
    p.add_argument('--steps', type=int, default=30_000_000)
    p.add_argument('--envs', type=int, default=2048)
    p.add_argument('--rollout', type=int, default=32)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatches', type=int, default=8)
    p.add_argument('--seed', type=int, default=733)
    # Base learning rate. The adaptive KL controller scales this per update, so
    # `ppo_effective_lr` in metrics.csv is the value that actually matters during
    # training.
    p.add_argument('--lr', type=float, default=2e-4)
    # Default PPO likelihoods are evaluated on the raw Gaussian sample; the
    # simulator still receives the clipped action. A compatibility switch is
    # retained for controlled comparisons with older checkpoints/semantics.
    p.add_argument('--logp-on-raw-action', action=argparse.BooleanOptionalAction, default=True,
                    help='use the pre-clip Gaussian sample for PPO log-probs (correct default). '
                         '--no-logp-on-raw-action uses the older clipped-action likelihood semantics for A/B only')
    p.add_argument('--entropy-cost', type=float, default=0.002)
    p.add_argument('--value-cost', type=float, default=0.5)
    p.add_argument('--device', choices=['auto','cpu','cuda','gpu'], default='auto')
    p.add_argument('--allow-cpu-mjx', action='store_true')
    p.add_argument('--run-name', default=f'emu_{VERSION}_run')
    p.add_argument('--run-dir', default='')
    p.add_argument('--log-every', type=int, default=1, help='log every N updates')
    p.add_argument('--checkpoint-every', type=int, default=10, help='save every N updates')
    # Rotating snapshots, separate from the frequent latest_mjx.pkl save
    # above -- a recovery floor in case that file ever gets overwritten
    # with a poisoned (e.g. NaN) state, erasing a run's whole progress.
    # Kept in raw step counts (not update counts) since that's the unit
    # people actually think in when deciding "how much am I willing to lose".
    p.add_argument('--snapshot-every-steps', type=int, default=2_000_000,
                    help='save a rotating, step-numbered checkpoint snapshot every N steps (0 disables)')
    p.add_argument('--snapshot-keep', type=int, default=10,
                    help='how many rotating snapshots to keep (oldest deleted first)')
    p.add_argument('--max-episode-steps', type=int, default=700)
    p.add_argument('--speed-test', action='store_true')
    p.add_argument('--drift-test', '--neutral-test', action='store_true', dest='drift_test')
    # --resume: 'auto' resumes this run's own latest checkpoint
    # (<run-dir>/checkpoints/latest_mjx.pkl); anything else is treated as a
    # literal .pkl path. Restores policy/value network, optimizer state, obs
    # normalization, and curriculum (RegimeParams); episodes/physics buffers
    # re-initialize fresh either way. --steps is the absolute target step
    # count when resuming, not an incremental amount.
    p.add_argument('--resume', default='', help="'auto' or a checkpoint .pkl path to resume from; default '' starts fresh")
    p.add_argument('--actuator-realism', action='store_true',
                    help='correct the servo\'s armature (1:345 real gear ratio, not the XML\'s direct-drive '
                         'assumption) and add Coulomb friction. Off by default (direct-drive, frictionless).')
    p.add_argument('--comm-latency-steps', type=int, default=0,
                    help='delay the slew-limiter\'s target reference by this many control steps, modeling '
                         'the time a real servo takes to receive a new target (0 = no delay, instant '
                         'delivery). 0-MAX_COMM_LATENCY_STEPS (4).')
    p.add_argument('--fall-target', type=float, default=None,
                    help='override RegimeGovernor.FALL_TARGET (class default 0.30) -- the fall rate '
                         'the proportional push/retreat controller holds each regime near. Lower = curriculum '
                         'stays more conservative; higher = tolerates more failure while expanding faster.')
    p.add_argument('--retreat-gain', type=float, default=None,
                    help='override RegimeGovernor.RETREAT_GAIN (class default 2.5, vs. PUSH_GAIN=1.0 '
                         '-- retreat is deliberately more aggressive than push by design). Lower values test '
                         'whether that asymmetry over-corrects under a noisier fall signal (e.g. latency).')
    p.add_argument('--difficulty-base-step', type=float, default=None,
                    help='override RegimeGovernor.DIFFICULTY_BASE_STEP (class default 0.0014) -- the '
                         'shared per-update step size for all four difficulty scalars (steady/transition/'
                         'resilience/posture), expressed as a fraction of each one\'s own 0-1 range -- see '
                         'environment.py\'s module doctrine for why one shared rate is enough.')
    p.add_argument('--bootstrap-updates', type=int, default=None,
                    help='override RegimeGovernor.BOOTSTRAP_UPDATES (class default 70) -- how many '
                         'updates run STEADY-only (100%% of envs) before TRANSITION/RESILIENCE open up at all.')
    p.add_argument('--regime-command-frac', type=float, default=None,
                    help='override RegimeParams.regime_command_frac (dataclass default 0.5) -- what '
                         'fraction of STEADY\'s current command envelope TRANSITION and RESILIENCE spawn at. '
                         'A fixed design constant, not a progressed difficulty scalar (see environment.py\'s '
                         'REGIME_COMMAND_FRAC_DEFAULT for the full rationale: isolating which axis is under '
                         'test from the aggregate difficulty of everything else at once). Untested at any '
                         'value other than its own default.')
    p.add_argument('--core-rehearsal-frac', type=float, default=0.0,
                    help='fraction of newly-reset envs reserved for permanent seed-band STEADY rehearsal. '
                         'These envs are excluded from frontier competence statistics. 0 disables; try 0.15.')
    p.add_argument('--athletic-rehearsal-frac', type=float, default=0.0,
                    help='v5007: fraction of births reserved for full proven v5005 athletic-envelope rehearsal. '
                         'Keeps vx/vy/yaw skills alive while frontier exploration moves beyond them; excluded from edge evidence.')
    p.add_argument('--curriculum-push-every-updates', type=int, default=1,
                    help='only allow curriculum EXPANSION every N updates; retreat remains immediate. '
                         'With rollout=32 and 700-step episodes, N=24 roughly matches one full episode cohort.')
    p.add_argument('--freeze-curriculum-after-steps', type=int, default=0,
                    help='freeze all curriculum difficulty + regime-allocation parameters after this absolute '
                         'training step while PPO and governor diagnostics continue (0 = never). Diagnostic for '
                         'task-distribution non-stationarity; the v5006 ladder uses 25M.')
    p.add_argument('--obs-norm-mode', choices=['all','continuous'], default='all',
                    help='all normalizes every observation channel; continuous leaves bounded/categorical '
                         'regime/phase/contact/action channels in natural coordinates.')
    p.add_argument('--freeze-obs-norm-after-steps', type=int, default=0,
                    help='freeze running observation mean/variance after this absolute training step (0 = never). '
                         'Diagnostic for curriculum-coupled coordinate drift; try 25000000.')
    p.add_argument('--obs-norm-anchor-steps', type=int, default=20_000_000,
                    help='capture normalization stats at/after this step and log later L2 drift from that anchor.')
    p.add_argument('--eval-every-steps', type=int, default=10_000_000,
                    help='run deterministic fixed-seed held-out evaluation every N training steps (0 disables).')
    p.add_argument('--eval-horizon', type=int, default=350,
                    help='control steps in each fixed evaluator cohort; first-fall survival is measured without timeout staggering.')
    p.add_argument('--eval-seed', type=int, default=1701,
                    help='fixed seed for held-out evaluator cohorts; identical across checkpoints/branches.')
    p.add_argument('--eval-cohorts', type=int, default=2,
                    help='number of deterministic held-out seed cohorts averaged per evaluator; v5007 default 2 reduces single-cohort quirks.')
    p.add_argument('--fixed-lr-scale', type=float, default=0.0,
                    help='>0 disables the adaptive-KL LR controller and holds ppo_lr_scale at this value. '
                         'Useful A/B for nested-controller oscillation; 0 keeps adaptive behaviour.')

    p.add_argument('--push-max-force-n', type=float, default=None,
                    help='override PUSH_MAX_FORCE_N (module default 0.35N) -- the resilience regime\'s '
                         'real-unit ceiling, both in the curriculum (how far resilience_push_n walks) and '
                         'the environment (the actual force applied). Raising it directly tests whether the '
                         'survivable push-force envelope can be widened past the current design ceiling.')
    p.add_argument('--vx-hi-max', type=float, default=0.45, help='forward design ceiling m/s; curriculum interpolates toward it')
    p.add_argument('--vx-lo-min', type=float, default=-0.15, help='reverse design ceiling m/s')
    p.add_argument('--vy-max-mps', type=float, default=0.18, help='lateral design ceiling m/s')
    p.add_argument('--yaw-max-rad-s', type=float, default=1.20, help='yaw-rate design ceiling rad/s')
    p.add_argument('--target-rate-limit-rad-s', type=float, default=1.75, help='max servo target slew rate before simulated servo physics')
    p.add_argument('--disable-target-slew', action='store_true',
                    help='v5007: deliver position targets directly and let geared-servo physics, not an extra software slew clamp, limit motion')
    p.add_argument('--phase-speed-gain', type=float, default=0.0, help='extra gait-clock Hz at full translational speed; 0 keeps v5004 cadence scaffold')
    p.add_argument('--phase-rate-max-hz', type=float, default=2.6, help='hard ceiling for speed-adaptive phase scaffold')
    p.add_argument('--sensor-noise-scale', type=float, default=1.0, help='multiplier for deployable sensor/estimator noise model')
    p.add_argument('--open-frontier', action='store_true', help='v5007: use repaired settled edge-tracking physical frontiers')
    p.add_argument('--frontier-growth-frac', type=float, default=0.0015, help='fractional growth of a proven physical frontier on an eligible curriculum push')
    p.add_argument('--frontier-retreat-frac', type=float, default=0.0040, help='fractional local retreat when survival/tracking says a frontier is too hard')
    p.add_argument('--push-frontier', action='store_true', help='let recoverable disturbance force expand by competence')
    p.add_argument('--sensor-frontier', action='store_true', help='let deployable estimator noise severity expand by competence')
    p.add_argument('--accel-frontier', action='store_true', help='let transition hold/ramp timing tighten by demonstrated tracking competence down to control-step limits')
    p.add_argument('--combo-mix-frac', type=float, default=0.0, help='fraction of command births that are mixed vx+vy+yaw blob samples; remainder isolate axes')
    p.add_argument('--combo-p', type=float, default=1.35, help='p-norm exponent for combined-command feasible blob (1=diamond, 2=ellipse)')
    p.add_argument('--combo-probe-frac', type=float, default=0.08, help='how far mixed-command probes may poke beyond the current blob radius')
    p.add_argument('--phase-obs-scale', type=float, default=1.0, help='scale actor phase sin/cos channels; 0 removes explicit gait-clock observation')
    p.add_argument('--warm-start-weights', default='', help='load network params + observation normalizer from a checkpoint but start a fresh optimizer/governor/frontier')
    p.add_argument('--stepping-activity-target', type=float, default=None,
                    help='override STEPPING_ACTIVITY_TARGET (module default 0.15) -- the reward\'s '
                         'anti-freezing gate threshold (see _reward\'s stepping_gate). Lower = easier to '
                         'satisfy (less pressure to keep stepping); higher = stricter.')
    p.add_argument('--speed-tolerance', type=float, default=None,
                    help='override the reward\'s overspeed/underspeed tolerance band (module default '
                         '0.025 m/s, both terms). Wider tolerance trades tracking precision for an easier '
                         'reward landscape -- tests whether that helps the policy explore a wider effective '
                         'command envelope under latency.')
    p.add_argument('--lr-target-kl', type=float, default=0.05,
                    help='override LR_TARGET_UPDATE_KL (default 0.05) -- the KL-trust-region '
                         'controller\'s per-update target that ppo_lr_scale adjusts toward (see train()\'s '
                         'lr_scale block). Tighter (lower) reacts to instability sooner/harder.')
    p.add_argument('--lr-scale-min', type=float, default=0.01,
                    help='override LR_SCALE_MIN (default 0.01) -- the floor ppo_lr_scale can be cut '
                         'to. A real run was observed ending with lr_scale still actively collapsing '
                         '(4.0 -> 0.351 in its last ~5M steps), a materially different, unresolved '
                         'optimizer-stability signal that episode-length alone didn\'t show. A higher '
                         'floor tests whether preventing the collapse changes anything, without '
                         'touching the controller\'s reaction logic itself.')
    p.add_argument('--lr-scale-baseline-pull', type=float, default=0.0,
                    help='every update, pull lr_scale a FRACTION of the remaining gap toward 1.0 '
                         '(unmodified --lr), on top of and after the existing reactive KL-based adjustment. '
                         '0.0 (default) = off. An adaptive controller that only reacts to current KL can '
                         'get stuck at an arbitrarily low, self-consistent rate: a policy that has stopped '
                         'doing much of anything can hold KL low while still performing badly, so it has '
                         'no path back up tied to whether the robot is actually walking. Try 0.01-0.03 '
                         'first -- deliberately much weaker than a single reactive step (which can move '
                         'lr_scale by 1.5x in one update).')
    p.add_argument('--recovery-window', type=int, default=0,
                    help='0 (default) = a simple consecutive-updates trigger (STEPPING_COLLAPSE_LIMIT=100 '
                         'in a row with fall_rate>0.65). That is a high bar -- 100 consecutive is ~6.5M '
                         'steps of UNBROKEN severe falling, and the governor\'s own retreats can visibly '
                         'interrupt that during a realistic, noisy oscillating collapse, so the recovery '
                         'mechanism (snapshot rollback included) may never fire in exactly the collapse it '
                         'exists to catch. Set >0 (try 150-250) to switch to a rolling-window trigger '
                         'instead: needs_recovery fires once --recovery-fraction of the last N updates had '
                         'severe falling (matching the pattern the KL-stress-chronic gate already uses) -- '
                         'no longer reset to zero by a single good rollout in the middle of a bad stretch.')
    p.add_argument('--recovery-fraction', type=float, default=0.40,
                    help='only used when --recovery-window > 0. Fraction of that window\'s updates '
                         'that must have had fall_rate > FALL_SEVERE_THRESHOLD (0.65) for needs_recovery to '
                         'fire. Default 0.40 -- deliberately looser than KL-stress-chronic\'s 0.90, since '
                         'stepping collapse is the more acute failure the snapshot rollback exists for.')
    p.add_argument('--fall-derivative-gain', type=float, default=0.0,
                    help='0.0 (default) leaves the controller exactly as before -- purely proportional '
                         'to current_fall_ema vs FALL_TARGET. Nonzero adds a term proportional to '
                         '(fall - current_fall_ema): the EMA update is fall_ema += ALPHA*(fall - fall_ema), so '
                         'this IS algebraically the "how fast is fall_ema currently moving" signal, not a '
                         'separate mechanism -- a faster-reacting point on the same reactive controller\'s '
                         'design space, not a fundamentally different one. Retreats harder (or holds off '
                         'pushing) when a single rollout is unusually bad relative to the recent smoothed '
                         'trend, before the EMA itself has caught up. Try 0.3-0.6 first.')
    p.add_argument('--legacy-rng-reuse', action='store_true',
                    help='compatibility/A-B switch that deliberately reproduces two old RNG key-reuse paths. '
                         'Off by default; see history.md for the original failure details.')
    p.add_argument('--stop-after-best-known-good', action='store_true',
                    help='stop training as soon as best_known_good.pkl is saved (median-of-8 '
                         'competence AND core_competence both >= 0.60 -- sustained, holistic health, the '
                         'same existing criterion that tier already uses, not a new cruder one) AND '
                         '--min-steps-before-stop has been reached. For building a shared BOOTSTRAP '
                         'checkpoint a later batch of runs all --resume from, instead of each one '
                         'separately re-paying the cost of learning to walk at all. Off by default -- a '
                         'normal run always trains for the full --steps.')
    p.add_argument('--min-steps-before-stop', type=int, default=15_000_000,
                    help='floor for --stop-after-best-known-good -- a lucky early window can\'t end '
                         'the run before the curriculum has even reached the later regimes. Ignored unless '
                         '--stop-after-best-known-good is set. --steps still applies as a hard ceiling '
                         'either way -- set it generously above this floor as a safety margin in case a '
                         'run is unusually slow to reach best_known_good.')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.speed_test:
        return run_speed_test(args)
    if args.drift_test:
        return run_drift_test(args)
    return train(args)

if __name__ == '__main__':
    raise SystemExit(main())
