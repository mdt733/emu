"""Optional environment, sensor and PPO diagnostics for the EMU CLI."""
from __future__ import annotations

import os
import sys
from .contracts import VERSION, ROBOT_XML

def doctor(args) -> int:
    print(f'EMU {VERSION} doctor')
    print(f'Python: {sys.version.split()[0]}')
    failures=[]
    try:
        import numpy as np; print(f'NumPy: {np.__version__}')
    except Exception as exc: failures.append(f'NumPy: {exc}')
    try:
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        import jax; print(f'JAX: {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}')
        if args.require_gpu and jax.default_backend() not in ('gpu','cuda'): failures.append('GPU required but JAX is not using GPU')
    except Exception as exc: failures.append(f'JAX: {exc}')
    try:
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        print(f'MuJoCo: {mujoco.__version__}  robot nq={m.nq} nv={m.nv} nu={m.nu}')
    except Exception as exc: failures.append(f'MuJoCo/XML: {exc}')
    try:
        import numpy as np
        from .curriculum import CapabilityField, FIELD_VERSION
        f=CapabilityField(); T,N=96,8; z=np.zeros((T,N),np.float32); o=np.ones((T,N),np.float32)
        age=(np.arange(T,dtype=np.float32)[:,None]+1)*.02+z
        m={'cmd_vx':z+.1,'cmd_vy':z,'cmd_yaw':z,'target_cmd_vx':z+.1,'target_cmd_vy':z,'target_cmd_yaw':z,
           'transition_ramp_speed':z,'vx':z+.1,'vy':z,'yaw_rate':z,'gravity_z':o,'support':o*.3,'bad_contact':z,
           'episode_fell':z,'episode_done':z,'age_s':age,'regime_resilience':z,'birth_resilience_d':z,'transition_frac':z,'cmd_resampled':z}
        d=f.update_from_rollout(m,step=65536); CapabilityField.from_state_dict(f.state_dict())
        if d['field_node_events_update']<=0: raise RuntimeError('no semantic node event')
        print(f'Capability field schema {FIELD_VERSION}: PASS')
    except Exception as exc: failures.append(f'Capability field: {exc}')
    if not args.no_mjx_smoke:
        try:
            import jax.numpy as jnp
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            e=EmuMJXEnv(str(ROBOT_XML),num_envs=8,seed=7,max_episode_steps=40,)
            p=RegimeGovernor().params_dict(); st,_=e.reset(p); out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),p); out.obs.block_until_ready()
            print(f'MJX reset/step: PASS obs={tuple(out.obs.shape)}')
        except Exception as exc: failures.append(f'MJX reset/step: {exc}')
    if getattr(args, 'v5015_smoke', False):
        # Explicit integration smoke: actually JIT-compile and step an
        # experimental four-command task, then compile the held-out summary.
        # No training/checkpoint/output files are created by this probe.
        try:
            import jax
            import jax.numpy as jnp
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            from .ppo import make_trainer
            from .contracts import OBS_DIM
            n = 4
            env = EmuMJXEnv(str(ROBOT_XML), num_envs=n, seed=5015,
                            max_episode_steps=40, attitude_profile='v5015',
                            pitch_command_max_deg=45.0)
            params = RegimeGovernor().params_dict()
            state, obs = env.reset(params)
            if state.cmd_target.shape != (n, 4) or obs.shape != (n, OBS_DIM):
                raise RuntimeError(f'v5015 reset contract: target={state.cmd_target.shape}, obs={obs.shape}')
            stepped = env.step(state, jnp.zeros((n, 6), dtype=jnp.float32), params)
            stepped.obs.block_until_ready()
            for key in ('body_pitch', 'cmd_pitch', 'robot_lean_forward',
                        'robot_lean_side', 'structural_attitude_ok', 'height_error'):
                arr = stepped.metrics[key]
                if arr.shape != (n,) or not bool(jax.device_get(jnp.all(jnp.isfinite(arr)))):
                    raise RuntimeError(f'v5015 invalid metric {key}: {arr.shape}')
            learner = make_trainer(env, seed=5015, lr=2e-4, rollout=4,
                                   epochs=1, minibatches=1)
            initial = learner['make_initial_state'](params)
            compact = learner['evaluate_summary'](
                initial.params, env.params_array(params),
                jnp.zeros((OBS_DIM,), dtype=jnp.float32),
                jnp.ones((OBS_DIM,), dtype=jnp.float32),
                jax.random.PRNGKey(5015), 3)
            compact.block_until_ready()
            if compact.shape != (11,) or not bool(jax.device_get(jnp.all(jnp.isfinite(compact)))):
                raise RuntimeError(f'v5015 held-out summary invalid: {compact.shape}')
            # Also JIT-compile one *real* PPO gradient/update, not just forward
            # inference: this catches reward/rollout/optimizer shape drift
            # before spending 20 million GPU environment steps.
            updated, rolled, diagnostics, _, _ = learner['train_update'](
                initial, env.params_array(params),
                jnp.zeros((OBS_DIM,), dtype=jnp.float32),
                jnp.ones((OBS_DIM,), dtype=jnp.float32),
                jnp.asarray(.25, dtype=jnp.float32))
            if not all(bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
                       for leaf in jax.tree_util.tree_leaves(updated.params)):
                raise RuntimeError('v5015 PPO smoke produced nonfinite parameters')
            if not bool(jax.device_get(jnp.isfinite(diagnostics['rollout_reward_mean']))):
                raise RuntimeError('v5015 PPO smoke produced nonfinite rollout reward')
            print('v5015 MJX + PPO update + evaluation smoke: PASS (4 envs, one optimizer update, 11 evaluation fields)')
        except Exception as exc:
            failures.append(f'v5015 MJX + PPO smoke: {exc}')
    if getattr(args, 'v5016_smoke', False):
        # Explicit sensor-schema acceptance smoke, no writes or long run.
        try:
            import jax
            import jax.numpy as jnp
            from .contracts import OBS_SCHEMA_SENSOR, SENSOR_OBS_DIM
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            from .ppo import make_trainer, build_obs_norm_mask
            n = 2
            env = EmuMJXEnv(str(ROBOT_XML), num_envs=n, seed=5016,
                            max_episode_steps=30, attitude_profile='v5015',
                            contact_slope_deg=2.,
                            observation_schema=OBS_SCHEMA_SENSOR)
            params = RegimeGovernor().params_dict()
            state, obs = env.reset(params)
            if obs.shape != (n, SENSOR_OBS_DIM) or state.hist.shape != (n, 4, 58):
                raise RuntimeError(f'v5016 sensor observation shape mismatch: {obs.shape}')
            if build_obs_norm_mask(OBS_SCHEMA_SENSOR).shape != (SENSOR_OBS_DIM,):
                raise RuntimeError('v5016 observation normalizer mask mismatch')
            out = env.step(state, jnp.zeros((n, 6), dtype=jnp.float32), params)
            out.obs.block_until_ready()
            if out.bootstrap_obs.shape != (n, SENSOR_OBS_DIM):
                raise RuntimeError('v5016 bootstrap observation shape mismatch')
            trainer = make_trainer(env, seed=5016, lr=2e-4, rollout=2,
                                   epochs=1, minibatches=1)
            init = trainer['make_initial_state'](params)
            packed = env.params_array(params)
            mean = jnp.zeros((SENSOR_OBS_DIM,), jnp.float32)
            var = jnp.ones((SENSOR_OBS_DIM,), jnp.float32)
            updated, _, diagnostics, _, _ = trainer['train_update'](
                init, packed, mean, var, jnp.asarray(.25, dtype=jnp.float32))
            if not all(bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
                       for leaf in jax.tree_util.tree_leaves(updated.params)):
                raise RuntimeError('v5016 PPO update produced nonfinite parameters')
            if not bool(jax.device_get(jnp.isfinite(diagnostics['rollout_reward_mean']))):
                raise RuntimeError('v5016 PPO update produced nonfinite rollout reward')
            print('v5016 sensor MJX + 232-input PPO gradient smoke: PASS '
                  '(2 worlds, 2-frame rollout, one optimizer update)')
        except Exception as exc:
            failures.append(f'v5016 sensor PPO smoke: {exc}')
    if args.domain_smoke:
        try:
            import jax
            import jax.numpy as jnp
            from .environment import DomainRandomizationConfig
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            cfg=DomainRandomizationConfig(enabled=True,robot_radius=.6,terrain_radius=.6,hardware_radius=.6)
            e=EmuMJXEnv(str(ROBOT_XML),num_envs=8,seed=5014,max_episode_steps=40,domain_config=cfg)
            p=RegimeGovernor().params_dict(); st,_=e.reset(p); out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),p); out.obs.block_until_ready()
            info=jax.device_get(e.domain_info)
            max_slope=max(float(abs(info['slope_roll_rad']).max()),float(abs(info['slope_pitch_rad']).max()))
            print(f'Domain MJX smoke: PASS obs={tuple(out.obs.shape)} max_sampled_slope={max_slope*57.2957795:.2f}deg')
        except Exception as exc: failures.append(f'Domain MJX smoke: {exc}')
    if failures:
        print('\nNOT READY:'); [print('  -',x) for x in failures]; return 1
    print('\nReady to train.'); return 0


