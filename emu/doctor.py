"""Optional environment, sensor and PPO diagnostics for the EMU CLI."""
from __future__ import annotations

import os
import sys
from .contracts import VERSION, ROBOT_XML, OBS_SCHEMA_TERRAIN

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
        from .capability_field import CapabilityField, FIELD_VERSION
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
            e=EmuMJXEnv(str(ROBOT_XML),num_envs=8,seed=7,max_episode_steps=700,
                        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
                        unified_capability_focus=True,academy_stream=True,
                        terrain_carpet=True,terrain_carpet_prisms=True)
            p=RegimeGovernor().params_dict()
            st,_=e.reset(p)
            out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),e.params_array(p))
            out.obs.block_until_ready()
            if out.obs.shape != (8,360):
                raise RuntimeError(f'Expected [8,360] from real prisms, got {out.obs.shape}')
            print(f'MJX reset/step: PASS obs={tuple(out.obs.shape)}')
        except Exception as exc: failures.append(f'MJX reset/step: {exc}')
    if args.domain_smoke:
        try:
            import jax
            import jax.numpy as jnp
            from .domain_randomization import DomainRandomizationConfig
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            # Geometry comes only from the prism carpet. Randomize robot
            # inertia, friction and hardware; do not rotate a retired plane.
            cfg=DomainRandomizationConfig(
                enabled=True,robot_radius=.6,terrain_radius=0.,hardware_radius=.6)
            e=EmuMJXEnv(str(ROBOT_XML),num_envs=8,seed=5014,max_episode_steps=700,
                        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
                        unified_capability_focus=True,academy_stream=True,
                        terrain_carpet=True,terrain_carpet_prisms=True,
                        domain_config=cfg)
            p=RegimeGovernor().params_dict()
            st,_=e.reset(p)
            out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),e.params_array(p))
            out.obs.block_until_ready()
            if out.obs.shape != (8,360):
                raise RuntimeError('Domain MJX actor has the wrong 360 input shape')
            print(f'Domain MJX robot/hardware noise: PASS obs={tuple(out.obs.shape)}')
        except Exception as exc: failures.append(f'Domain MJX smoke: {exc}')
    if failures:
        print('\nNOT READY:'); [print('  -',x) for x in failures]; return 1
    print('\nReady to train.'); return 0


