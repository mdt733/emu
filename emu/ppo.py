"""EMU PPO implementation and observation normalisation.

This is the learning engine only. It owns network/optimizer state, rollout and
PPO updates, and policy evaluation kernels. It does not own curriculum,
checkpoint policy, experiment orchestration, or training-health decisions.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .contracts import ACTION_DIM, OBS_DIM, BASE_OBS_FIELDS, HISTORY_LEN, OBS_NORMALIZED_FIELDS
from .environment import T
from .gait_exam import contact_pair_mask
from .directional import heading_rate_request
from .contracts import CONTROL_DT
from .tracking_contract import (
    VX_ABS_TOL, VX_REL_TOL, VY_ABS_TOL, VY_REL_TOL, YAW_ABS_TOL, YAW_REL_TOL,
    UPRIGHT_GRAVITY_Z_MIN, SUPPORT_MIN, NODE_STRUCTURAL_FRACTION_MIN,
)

# Fixed current PPO doctrine. These are not experiment knobs in modern EMU;
# training_config records them with each run, while this module owns their use.
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.20
PPO_MAX_GRAD_NORM = 1.0
PPO_HIDDEN_SIZES = (256, 256)
PPO_LOG_STD_INIT = -1.55
PPO_STD_MIN = 0.10
PPO_STD_MAX = 0.60
PPO_MAX_CONSECUTIVE_NONFINITE = 100
PPO_KL_HARD_STOP = 1.0
PPO_VALUE_HUBER_DELTA = 10.0

PPO_DOCTRINE = {
    'PPO_GAMMA': PPO_GAMMA,
    'PPO_GAE_LAMBDA': PPO_GAE_LAMBDA,
    'PPO_CLIP_EPS': PPO_CLIP_EPS,
    'PPO_MAX_GRAD_NORM': PPO_MAX_GRAD_NORM,
    'PPO_HIDDEN_SIZES': PPO_HIDDEN_SIZES,
    'PPO_LOG_STD_INIT': PPO_LOG_STD_INIT,
    'PPO_STD_MIN': PPO_STD_MIN,
    'PPO_STD_MAX': PPO_STD_MAX,
    'PPO_MAX_CONSECUTIVE_NONFINITE': PPO_MAX_CONSECUTIVE_NONFINITE,
    'PPO_KL_HARD_STOP': PPO_KL_HARD_STOP,
    'PPO_VALUE_HUBER_DELTA': PPO_VALUE_HUBER_DELTA,
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


def build_obs_norm_mask():
    """Normalise physical-scale channels; keep bounded/categorical channels raw."""
    import numpy as _np
    base = _np.asarray(
        [1.0 if name in OBS_NORMALIZED_FIELDS else 0.0 for name in BASE_OBS_FIELDS],
        dtype=_np.float32,
    )
    return _np.tile(base, HISTORY_LEN)

# ---------------------------------------------------------------------------
# PPO implementation.
# ---------------------------------------------------------------------------

class PPOState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: Any
    obs: Any
    key: Any


def parameter_group_labels(_params):
    """Prefix labels consumed by optax.partition.

    Optax explicitly permits this tree to be a prefix of the parameter tree,
    so the labels apply recursively to every leaf below ``pi`` and ``vf``.
    """
    return {'pi': 'policy', 'vf': 'value', 'log_std': 'policy'}


def expected_param_shapes(obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
    """Pure-Python shape contract for the current policy/value parameter tree."""
    def mlp(out_dim: int):
        sizes = (int(obs_dim),) + tuple(PPO_HIDDEN_SIZES) + (int(out_dim),)
        return [
            {'w': (fan_in, fan_out), 'b': (fan_out,)}
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
        ]
    return {'pi': mlp(action_dim), 'vf': mlp(1), 'log_std': (int(action_dim),)}


def make_trainer(env, *, seed: int, lr: float, rollout: int, epochs: int, minibatches: int,
                 entropy_cost: float = 0.002, value_cost: float = 0.5):
    import jax
    import jax.numpy as jnp
    import optax

    obs_norm_mask = jnp.asarray(build_obs_norm_mask(), dtype=jnp.float32)

    _tmp_state, _tmp_obs = env.reset({})
    obs_dim = int(_tmp_obs.shape[-1])
    action_dim = ACTION_DIM
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
        sizes = (int(obs_dim),) + PPO_HIDDEN_SIZES + (int(out_dim),)
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
            'log_std': jnp.full((action_dim,), PPO_LOG_STD_INIT, dtype=jnp.float32),
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
    def eff_log_std(params):
        return jnp.clip(params['log_std'], float(np.log(PPO_STD_MIN)), float(np.log(PPO_STD_MAX)))

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
    # Also: value_loss now uses Huber (see PPO_VALUE_HUBER_DELTA below)
    # instead of raw MSE, which caps each individual sample's contribution
    # to the value gradient -- reduces how large a spike can even become
    # in the first place, independent of and complementary to the
    # separate-clipping fix below.
    optimiser = optax.apply_if_finite(
        optax.partition(
            {
                'policy': optax.chain(optax.clip_by_global_norm(PPO_MAX_GRAD_NORM), optax.adam(float(lr))),
                'value': optax.chain(optax.clip_by_global_norm(PPO_MAX_GRAD_NORM), optax.adam(float(lr))),
            },
            parameter_group_labels,
        ),
        max_consecutive_errors=PPO_MAX_CONSECUTIVE_NONFINITE,
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
        # `ppo_action_clip_frac` is logged so late-run saturation remains visible.
        lp = log_prob(params, mean, raw)
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
        deltas = rewards + PPO_GAMMA * next_values * (1.0 - terminals) - values
        def rev_step(gae, x):
            delta, done = x
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * (1.0 - done) * gae
            return gae, gae
        _, adv_rev = jax.lax.scan(rev_step, jnp.zeros_like(deltas[0]), (deltas[::-1], dones[::-1]))
        adv = adv_rev[::-1]
        return adv, adv + values

    def flatten_rollout(traj, adv, ret):
        flat = {
            'obs': traj['obs'].reshape((batch_size, obs_dim)),
            'action': traj['raw_action'].reshape((batch_size, action_dim)),
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
        pg2 = jnp.clip(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv
        policy_loss = -jnp.mean(jnp.minimum(pg1, pg2))
        value_loss = jnp.mean(optax.huber_loss(value, batch['ret'], delta=PPO_VALUE_HUBER_DELTA))
        ent = entropy(params)
        approx_kl = 0.5 * jnp.mean(log_ratio ** 2)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > PPO_CLIP_EPS).astype(jnp.float32))
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
    # minibatch's approx_kl exceeds PPO_KL_HARD_STOP, stop applying further
    # updates for the rest of this call.
    #
    # PPO_KL_HARD_STOP=1.0 is a safety net, not the trust region itself (PPO's
    # trust region is the clipped surrogate, clip_eps=0.2) -- set high
    # enough that a healthy run essentially never touches it, low enough to
    # still catch a genuine runaway before it does real damage. A much
    # lower hard stop (e.g. 0.03) can saturate instead -- pinning most
    # minibatches to a single applied gradient step per update and feeding
    # the curriculum governor a permanently-saturated stress signal, which
    # is worse than the problem it's meant to prevent. Step-size control is
    # handled separately by LR_TARGET_UPDATE_KL below, a soft continuous
    # controller instead of a cliff.
    # Huber delta for the value loss -- errors up to this magnitude are
    # treated quadratically (same as ordinary MSE, preserving normal
    # fine-grained value learning), errors beyond it are treated linearly,
    # capping how large a single sample's contribution to the value
    # gradient can get. 10.0 is sized against this codebase's typical
    # per-step reward scale (a few points), not against the terminal fall
    # penalty specifically -- ordinary TD errors during healthy learning
    # should mostly stay well inside it.

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
        stopped = jnp.logical_or(stopped, aux['approx_kl'] > PPO_KL_HARD_STOP)
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
        rollout_policy_state = state
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
        return new_state, rollout_metrics, aux_mean, raw_obs_batch, rollout_policy_state

    def evaluate_summary(params, curriculum_params, obs_mean, obs_std, eval_key, horizon):
        """Same fixed evaluator semantics, but aggregate on-device.

        Keep only current environment state, first-fall status, and running
        sums on device, returning the compact summary consumed by held-out
        evaluation rather than transferring per-step histories to the host.

        Output order:
          survival, first_fall_step_mean, vx_mae, vy_mae, yaw_mae,
          gravity_z, foot_mid_cone, step_evidence, support,
          pitch_mae, height_shortfall_m
        """
        keys = jax.random.split(eval_key, env.num_envs)
        env_state = env._reset_with_keys(keys, curriculum_params)
        obs = env.obs(env_state)
        first_fall0 = jnp.full((env.num_envs,), -1, dtype=jnp.int32)
        sums0 = jnp.zeros((9,), dtype=jnp.float32)

        def one_step(i, carry):
            es, ob, first_fall, sums = carry
            mean, _ = apply(params, normalize_obs(ob, obs_mean, obs_std, obs_norm_mask))
            action = jnp.clip(mean, -1.0, 1.0)
            out = env.step(es, action, curriculum_params)
            m = out.metrics
            fell = m['episode_fell'] > 0.5
            first_fall = jnp.where(jnp.logical_and(first_fall < 0, fell), i + 1, first_fall)
            sums = sums + jnp.stack((
                jnp.sum(jnp.abs(m['vx'] - m['cmd_vx'])),
                jnp.sum(jnp.abs(m['vy'] - m['cmd_vy'])),
                jnp.sum(jnp.abs(m['yaw_rate'] - m['cmd_yaw'])),
                jnp.sum(m['gravity_z']),
                jnp.sum(m['foot_mid_cone']),
                jnp.sum(m['step_evidence']),
                jnp.sum(m['support']),
                jnp.sum(jnp.abs(m['body_pitch'] - m['cmd_pitch'])),
                jnp.sum(jnp.maximum(0.0, -m['height_error'])),
            ))
            return out.state, out.obs, first_fall, sums

        _, _, first_fall, sums = jax.lax.fori_loop(
            0, horizon, one_step, (env_state, obs, first_fall0, sums0))
        failed = first_fall >= 0
        failed_count = jnp.sum(failed.astype(jnp.float32))
        survival = 1.0 - failed_count / float(env.num_envs)
        first_fall_sum = jnp.sum(jnp.where(failed, first_fall, 0).astype(jnp.float32))
        first_fall_mean = jnp.where(
            failed_count > 0.0,
            first_fall_sum / jnp.maximum(failed_count, 1.0),
            jnp.asarray(float(horizon), dtype=jnp.float32),
        )
        denom = jnp.asarray(float(env.num_envs * horizon), dtype=jnp.float32)
        means = sums / denom
        return jnp.concatenate((jnp.stack((survival, first_fall_mean)), means))

    evaluate_summary = jax.jit(evaluate_summary, static_argnums=(5,))

    def evaluate_trace(params, curriculum_params, obs_mean, obs_std, eval_key,
                       horizon, heading_kp=0.0):
        """Pre-reset physics trace, optionally with IMU-relative heading hold.

        ON/OFF use the exact same operator commands, initial states and seeds.
        The ON controller integrates measured body gyro-Z (not simulator yaw
        truth) and modifies ONLY the yaw-rate request delivered to PPO.
        The examiner always grades against the original operator command.
        This first integration supports fixed STEADY commands, not transitions.
        """
        keys = jax.random.split(eval_key, env.num_envs)
        initial = env._reset_with_keys(keys, curriculum_params)
        obs = env.obs(initial)
        # Fixed requested commands must survive any controller correction.
        operator = jnp.stack((initial.task[:, T.vx], initial.task[:, T.vy],
                              initial.task[:, T.yaw]), axis=-1)
        yaw_field = BASE_OBS_FIELDS.index('cmd_yaw')
        kp = float(heading_kp)
        if not 0.0 <= kp <= 2.0:
            raise ValueError('heading_kp must be in [0, 2]')

        def record_state(es):
            data = es.data
            rotation = env._rotation_body_to_world(data)
            yaw = jnp.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
            contacts = contact_pair_mask(
                jnp, data.contact.geom1, data.contact.geom2,
                data.contact.dist, int(env.ids['floor_geom']),
                tuple(int(x) for x in env.ids['foot_geoms']))
            return data.qpos[:, :2], yaw, data.xpos[:, env.foot_bodies, :], contacts

        def step(carry, _):
            es, ob, target, estimated = carry
            xy, yaw, foot_xyz, contacts = record_state(es)
            # Body gyro integration: no perfect MuJoCo yaw oracle is used
            # to control the trained actor. This is noiseless simulated gyro;
            # sensor bias/latency robustness remains a separate future gate.
            body_gyro_z = env._body_packets(es.data)[2][:, 2]
            next_target = target + operator[:, 2] * CONTROL_DT
            next_estimated = estimated + body_gyro_z * CONTROL_DT
            corrected, error = heading_rate_request(
                jnp, operator[:, 2], next_target, next_estimated,
                kp=kp, max_rate=env.yaw_max_rad_s)
            if kp > 0:
                delivered_task = es.task.at[:, T.yaw].set(corrected)
                # The latest history frame is the current command received
                # by PPO. Earlier frames remain real history, not rewritten.
                hist = es.hist.at[:, -1, yaw_field].set(corrected)
                stepped_state = es._replace(task=delivered_task, hist=hist)
                policy_obs = hist.reshape((env.num_envs, OBS_DIM))
            else:
                stepped_state, policy_obs = es, ob
            mean, _ = apply(params, normalize_obs(
                policy_obs, obs_mean, obs_std, obs_norm_mask))
            out = env.step(stepped_state, jnp.clip(mean, -1.0, 1.0),
                           curriculum_params)
            sample = dict(position_xy=xy, heading_rad=yaw,
                          foot_xyz=foot_xyz, floor_contact=contacts,
                          command=operator, issued_yaw_rate=corrected,
                          estimated_heading_rad=next_estimated,
                          requested_heading_rad=next_target,
                          controller_heading_error_rad=error,
                          done=out.done, fell=out.metrics['episode_fell'])
            return (out.state, out.obs, next_target, next_estimated), sample

        zeros = jnp.zeros((env.num_envs,), dtype=jnp.float32)
        (end_state, _, _, _), frames = jax.lax.scan(
            step, (initial, obs, zeros, zeros), xs=None, length=horizon)
        last_xy, last_yaw, last_feet, last_contact = record_state(end_state)
        return dict(**frames, last_position_xy=last_xy,
                    last_heading_rad=last_yaw, last_foot_xyz=last_feet,
                    last_floor_contact=last_contact)

    evaluate_trace = jax.jit(evaluate_trace, static_argnums=(5, 6))

    def evaluate_recovery(params, curriculum_params, obs_mean, obs_std, eval_key, horizon,
                          stable_steps=10, recovery_not_before=0):
        """Return fall/recovery times for capability probes.

        Recovery means sustained healthy dynamic command tracking: alive,
        commanded-attitude-compatible, supported, and tracking the three
        movement axes for ``stable_steps``.
        """
        keys = jax.random.split(eval_key, env.num_envs)
        env_state = env._reset_with_keys(keys, curriculum_params)
        obs = env.obs(env_state)
        first_fall0 = jnp.full((env.num_envs,), -1, dtype=jnp.int32)
        first_recovery0 = jnp.full((env.num_envs,), -1, dtype=jnp.int32)
        streak0 = jnp.zeros((env.num_envs,), dtype=jnp.int32)

        def one_step(i, carry):
            es, ob, first_fall, first_recovery, streak = carry
            mean, _ = apply(params, normalize_obs(ob, obs_mean, obs_std, obs_norm_mask))
            action = jnp.clip(mean, -1.0, 1.0)
            out = env.step(es, action, curriculum_params)
            m = out.metrics
            fell = m['episode_fell'] > 0.5
            first_fall = jnp.where(jnp.logical_and(first_fall < 0, fell), i + 1, first_fall)
            # Slightly forgiving instantaneous gates plus a consecutive-frame
            # requirement are a better definition of dynamic recovery than
            # demanding low body rates during a walking gait.
            vx_tol = jnp.maximum(VX_ABS_TOL, VX_REL_TOL * jnp.abs(m['cmd_vx']))
            vy_tol = jnp.maximum(VY_ABS_TOL, VY_REL_TOL * jnp.abs(m['cmd_vy']))
            yaw_tol = jnp.maximum(YAW_ABS_TOL, YAW_REL_TOL * jnp.abs(m['cmd_yaw']))
            stable = (
                (first_fall < 0) &
                (m['structural_attitude_ok'] > .5) &  # requested chassis attitude, not rigid upright
                (jnp.abs(m['vx'] - m['cmd_vx']) <= vx_tol) &
                (jnp.abs(m['vy'] - m['cmd_vy']) <= vy_tol) &
                (jnp.abs(m['yaw_rate'] - m['cmd_yaw']) <= yaw_tol) &
                (m['support'] > SUPPORT_MIN)
            )
            streak = jnp.where(stable, streak + 1, 0)
            newly = ((first_recovery < 0) & (streak >= stable_steps) &
                     (i >= jnp.maximum(4, recovery_not_before + stable_steps - 1)))
            first_recovery = jnp.where(newly, i + 1, first_recovery)
            return out.state, out.obs, first_fall, first_recovery, streak

        _, _, first_fall, first_recovery, _ = jax.lax.fori_loop(
            0, horizon, one_step,
            (env_state, obs, first_fall0, first_recovery0, streak0))
        return first_fall, first_recovery

    evaluate_recovery = jax.jit(evaluate_recovery, static_argnums=(5, 6))

    def evaluate_node_window(params, curriculum_params, obs_mean, obs_std, eval_key, horizon,
                             window_steps=50, settle_steps=25):
        """Held-out node exam matching the capability-field window contract.

        After settling, compare mean absolute tracking error with mean dynamic
        tolerance and require structural health for the configured window fraction.
        The result is one Bernoulli verdict per environment.
        """
        keys=jax.random.split(eval_key,env.num_envs)
        env_state=env._reset_with_keys(keys,curriculum_params); obs=env.obs(env_state)
        err0=jnp.zeros((env.num_envs,3),dtype=jnp.float32)
        tol0=jnp.zeros((env.num_envs,3),dtype=jnp.float32)
        good0=jnp.zeros((env.num_envs,),dtype=jnp.float32)
        count0=jnp.zeros((env.num_envs,),dtype=jnp.float32)
        end_step=jnp.minimum(jnp.asarray(horizon,dtype=jnp.int32),jnp.asarray(settle_steps+window_steps,dtype=jnp.int32))
        def one_step(i,carry):
            es,ob,err,tol,good,count=carry
            mean,_=apply(params,normalize_obs(ob,obs_mean,obs_std,obs_norm_mask)); action=jnp.clip(mean,-1.0,1.0)
            out=env.step(es,action,curriculum_params); m=out.metrics
            use=(i>=settle_steps)&(i<end_step)
            usef=use.astype(jnp.float32)
            vx_tol=jnp.maximum(VX_ABS_TOL,VX_REL_TOL*jnp.abs(m['cmd_vx'])); vy_tol=jnp.maximum(VY_ABS_TOL,VY_REL_TOL*jnp.abs(m['cmd_vy'])); yaw_tol=jnp.maximum(YAW_ABS_TOL,YAW_REL_TOL*jnp.abs(m['cmd_yaw']))
            e=jnp.stack((jnp.abs(m['vx']-m['cmd_vx']),jnp.abs(m['vy']-m['cmd_vy']),jnp.abs(m['yaw_rate']-m['cmd_yaw'])),axis=1)
            tt=jnp.stack((vx_tol,vy_tol,yaw_tol),axis=1)
            structural=((m['structural_attitude_ok']>.5)&(m['support']>SUPPORT_MIN)&(m['bad_contact']<.5)&(m['episode_fell']<.5)).astype(jnp.float32)
            err=err+usef*e; tol=tol+usef*tt; good=good+usef*structural; count=count+usef
            return out.state,out.obs,err,tol,good,count
        _,_,err,tol,good,count=jax.lax.fori_loop(0,horizon,one_step,(env_state,obs,err0,tol0,good0,count0))
        c=jnp.maximum(count,1.0); tracking=jnp.all((err/c[:,None]) <= (tol/c[:,None]),axis=1); structural=(good/c)>=NODE_STRUCTURAL_FRACTION_MIN
        enough=count>=jnp.minimum(jnp.asarray(float(window_steps)),jnp.asarray(float(max(1,horizon-settle_steps))))
        return tracking & structural & enough

    evaluate_node_window = jax.jit(evaluate_node_window, static_argnums=(5,6,7))

    return {
        'make_initial_state': make_initial_state,
        'train_update': train_update,
        'evaluate_summary': evaluate_summary,
        'evaluate_trace': evaluate_trace,
        'evaluate_recovery': evaluate_recovery,
        'evaluate_node_window': evaluate_node_window,
        'obs_dim': obs_dim,
    }

