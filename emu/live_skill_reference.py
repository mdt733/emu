"""Fixed recovery practice embedded in ordinary PPO worlds.

All six cases have identical .16m/s forward command, flat physical terrain,
no extra birth tilt or later push, and a signed longitudinal velocity
offset. Random gait phase and normal sensor/joint noise remain. Reference
births replace only a bounded fraction of mature RESILIENCE births.
"""
REFERENCE_KICKS_MPS = (-.40,-.25,-.10,.10,.25,.40)
REFERENCE_VX_MPS = .16
SETTLE_DEADLINE_S = 3.
SETTLE_FRAMES = 25  # .5 seconds continuous good physical tracking


def reference_assignment(jax,jnp,keys,is_resilience,active,athletic,mix,share):
    if share <= 0.:
        return jnp.full(is_resilience.shape,-1,dtype=jnp.int32)
    draw=jax.vmap(lambda k:jax.random.uniform(jax.random.fold_in(k,50601),()))(keys)
    case=jax.vmap(lambda k:jax.random.randint(jax.random.fold_in(k,50602),(),0,6))(keys)
    eligible=is_resilience & active & (athletic<.5) & (mix>=.999999)
    return jnp.where(eligible & (draw<share),case,-1).astype(jnp.int32)


def reference_task(jnp,task,T,ids,vx_hi,phase_gain):
    from .environment import command_phase_rate
    mask=ids>=0
    for col in (T.vy,T.yaw,T.height,T.pitch,T.roll,T.resilience,T.birth_resilience_d,T.field_taught):
        task=task.at[:,col].set(jnp.where(mask,0.,task[:,col]))
    task=task.at[:,T.vx].set(jnp.where(mask,REFERENCE_VX_MPS,task[:,T.vx]))
    task=task.at[:,T.regime].set(jnp.where(mask,2.,task[:,T.regime]))
    phase=command_phase_rate(jnp.full(ids.shape,REFERENCE_VX_MPS),jnp.zeros(ids.shape),vx_hi,phase_gain,jnp.ones(ids.shape))
    return task.at[:,T.phase_rate].set(jnp.where(mask,phase,task[:,T.phase_rate]))


def reference_velocity(jnp,qvel,ids):
    mask=ids>=0
    kick=jnp.asarray(REFERENCE_KICKS_MPS,dtype=qvel.dtype)[jnp.maximum(ids,0)]
    target=jnp.zeros_like(qvel[:,:6]).at[:,0].set(REFERENCE_VX_MPS+kick)
    return qvel.at[:,:6].set(jnp.where(mask[:,None],target,qvel[:,:6]))


def reference_progress(jnp,ids,streak,settled,metrics,fell,age_s):
    active=(ids>=0)&(age_s<=SETTLE_DEADLINE_S+1e-6)&(~fell)
    good=(active & (jnp.abs(metrics['vx']-REFERENCE_VX_MPS)<=.08)
          & (jnp.abs(metrics['vy'])<=.06) & (jnp.abs(metrics['yaw_rate'])<=.25)
          & (metrics['structural_attitude_ok']>.5) & (metrics['support']>.10)
          & (metrics['bad_contact']<.5))
    streak=jnp.where(good,streak+1,0)
    return streak,settled | (streak>=SETTLE_FRAMES)
