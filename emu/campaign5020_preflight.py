"""Non-destructive v5020 MJX campaign gate: exercise the *promotion examiner*.

Unlike the historical GPU smoke (which checked training alone), this calls
JIT-compiled frozen-policy real-contact step and slope evaluators, including
servo-power telemetry, using a tiny randomly initialized schema-9 actor.
Success proves runnable interfaces and finite measurements, NOT walking.
All temporary XML stays outside the Git checkout and campaign root.
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def run_preflight(*, seed: int = 733, device: str = "cuda",
                  worlds: int = 8, horizon: int = 6) -> dict:
    if worlds < 2 or worlds > 64:
        raise ValueError("preflight worlds must be 2..64")
    if horizon < 2 or horizon > 100:
        raise ValueError("preflight horizon must be 2..100")
    from .ppo import configure_jax, require_backend
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    require_backend(device == "cpu")
    from .campaign5020 import config, ARMS
    from .contracts import ROBOT_XML, OBS_SCHEMA_PREVIEW
    from .training_config import training_sensor_contract
    from .evaluation import _screen_world, _screen_episode
    from .terrain import adaptive_height_course

    with tempfile.TemporaryDirectory(prefix="emu-v5020-preflight-") as scratch:
        home=Path(scratch)
        foundation=config(home/"foundation20",stage="foundation",arm=None,
                          seed=seed,envs=worlds,device=device)
        contract=training_sensor_contract(foundation)
        # Confirm all preflighted candidates have the EXACT same 312 sensor
        # and normalizer interface as the shared parent, without touching runs.
        for arm in ARMS:
            for stage in ("round1","round2","round3"):
                cfg=config(home/stage/arm,stage=stage,arm=arm,
                           seed=seed,envs=worlds,device=device)
                if (cfg.observation_schema != OBS_SCHEMA_PREVIEW or
                    training_sensor_contract(cfg) != contract):
                    raise RuntimeError(f"312 sensor contract mismatch: {stage}/{arm}")

        course=adaptive_height_course(ROBOT_XML,home/"physical_course.xml",immutable=True)
        course_env,course_ppo=_screen_world(
            foundation,envs=worlds,seed_offset=1771,
            limits=(.40,-.10,.12,.80),push_n=.35,
            sensor_scale=1.5,combo_mix=.05,
            terrain_radius=.22,terrain_mode="contact",course_path=course)
        exam=_screen_episode(.25,.25,.15,.25,(.65,.25,.10),.75)
        exam.update(frontier_terrain_height_m=.020,
                    probe_cmd_vx=.20,probe_cmd_vy=0.,probe_cmd_yaw=0.)
        p=course_env.params_array(exam)
        initial=course_ppo["make_initial_state"](p)
        if initial.obs.shape != (worlds,312):
            raise RuntimeError(f"incorrect preview observations {initial.obs.shape}")
        mu=jnp.zeros((312,),dtype=jnp.float32)
        sig=jnp.ones((312,),dtype=jnp.float32)
        key=jax.random.PRNGKey(seed+1771)
        real=np.asarray(jax.device_get(course_ppo["evaluate_terrain_summary"](
            initial.params,p,mu,sig,key,horizon)),dtype=float)
        if (real.shape != (3,) or not np.isfinite(real).all() or
            real[0] < 1 or not (0<=real[1]<=1) or not (0<=real[2]<=1)):
            raise RuntimeError(f"real step exam failed: {real.tolist()}")

        slope_env,slope_ppo=_screen_world(
            foundation,envs=worlds,seed_offset=1667,
            limits=(.65,-.20,.24,1.50),push_n=.80,
            sensor_scale=1.5,combo_mix=.35,
            terrain_radius=.30,terrain_mode="contact")
        slope_p=slope_env.params_array(
            _screen_episode(.65,.65,.45,.60,(.35,.40,.25),.75))
        summary=np.asarray(jax.device_get(slope_ppo["evaluate_summary"](
            initial.params,slope_p,mu,sig,jax.random.PRNGKey(seed+1667),horizon)),dtype=float)
        power=np.asarray(jax.device_get(slope_ppo["evaluate_efficiency"](
            initial.params,slope_p,mu,sig,jax.random.PRNGKey(seed+1883),horizon)),dtype=float)
        if (summary.shape != (11,) or power.shape != (5,) or
            not np.isfinite(summary).all() or not np.isfinite(power).all()):
            raise RuntimeError(f"contact-slope/power exam failed: {summary}, {power}")

        result={"status":"PASS","device":jax.default_backend(),"worlds":worlds,
                "horizon":horizon,"obs_dim":312,"physical_course_attempts":int(real[0]),
                "physical_contact_rate":float(real[1]),
                "physical_crossing_rate":float(real[2]),
                "slope_survival_random_actor":float(summary[0]),
                "shaft_power_random_actor_w":float(power[0])}
        return result
