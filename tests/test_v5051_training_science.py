"""v5051 source-pinned PPO experimental controls; no CUDA required."""
from __future__ import annotations
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from emu.cli import main as emu_cli
from emu.step_autopsy import inspect_missions
from emu.rewards import _gait_maturity, _physical_qualities
from scripts.v5051_train_campaign import (
    ARMS, PARENT_STEPS, FINAL_STEPS, PHYSICAL_PANELS,
    cli_command, exam_protocol, select_arms)
from emu.evaluation import PANELS, broad_score


def trace(*, second_foot=True, continued=True, heading=0.):
    n=100
    cells=np.zeros((n,1,2),dtype=np.int32)
    cells[10:,:,0]=1
    top=np.zeros((n,1,4,4))
    top[:,:,1,:]=.02
    pairs=np.zeros((n,1,8,2),dtype=bool)
    vertical=np.zeros((n,1,8,2))
    pairs[10:,:,2:4,0]=True
    vertical[10:,:,2:4,0]=1.
    if second_foot:
        pairs[20:,:,2:4,1]=True
        vertical[20:,:,2:4,1]=1.
    feet=np.zeros((n,1,2,3))
    feet[:,:,:,0]=.85
    feet[:,:,:,1]=.375
    feet[:,:,:,2]=.038
    # The source-to-target crossing at frame 10 is supported for >=5
    # frames; body continues from x=.80 to x=1.35 (if requested).
    xy=np.zeros((n,1,2))
    xy[:,:,0]=.8
    if continued:
        xy[30:,:,0]=np.linspace(.8,1.35,n-30)[:,None]
    yaw=np.full((n,1),heading)
    return dict(
        carpet_root_cell=cells,carpet_prism_corner_z=top,
        carpet_raw_contact_pairs=pairs,
        carpet_contact_vertical=vertical,foot_xyz=feet,
        initial_foot_radii_m=np.array([.018,.018]),
        done=np.zeros((n,1),bool),fell=np.zeros((n,1),bool),
        position_xy=xy,heading_rad=yaw,
    )


class ScienceTrainingTests(unittest.TestCase):
    def test_fifteen_independent_60m_forks(self):
        self.assertEqual(len(ARMS),15)
        self.assertEqual(len({a.name for a in ARMS}),len(ARMS))
        self.assertEqual(PARENT_STEPS,60_000_000)
        self.assertEqual(FINAL_STEPS,120_000_000)
        for arm in ARMS:
            self.assertEqual(len(arm.overrides)%2,0)

    def test_compact_retention_and_physical_protocol(self):
        self.assertEqual(set(PANELS),
                         {"core","dynamic80","combo70","resilience75"})
        self.assertEqual({v[0] for v in PANELS.values()},{"fixed","combo"})
        # Four retention guards cannot silently include an old/gone panel
        # in best-policy scoring, or claim to certify stairs.
        baseline={f"eval_{name}_survival":1. for name in PANELS}
        self.assertAlmostEqual(broad_score(baseline),1.)
        self.assertEqual([x[0] for x in PHYSICAL_PANELS],[
            "standing","slow_walk","fast_walk","slope","step_3cm","step_4cm"])
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=exam_protocol(root,8)
            self.assertEqual(first["worlds"],8)
            self.assertEqual(exam_protocol(root,8),first)
            with self.assertRaisesRegex(RuntimeError,"Physical exam"):
                exam_protocol(root,16)

    def test_all_arms_parse_to_real_trainer_without_training(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/"v5051"
            for arm in (None,*ARMS):
                cmd=cli_command(root,seed=735,envs=2048,arm=arm,
                                resume=(root/"foundation/checkpoints/final.pkl"
                                        if arm is not None else None))
                self.assertIn("--steps",cmd)
                self.assertEqual(cmd[cmd.index("--steps")+1],
                                 str(FINAL_STEPS if arm else PARENT_STEPS))
                self.assertIn("--carpet-prisms",cmd)
                self.assertIn("--academy-stream",cmd)
                with redirect_stdout(io.StringIO()):
                    rc=emu_cli([*cmd[3:],"--plan"])
                self.assertEqual(rc,0,arm.name if arm else "foundation")

    def test_four_world_actual_mjx_science_update(self):
        # Compile the *real physics step* including new transfer event
        # memory and all reward-side research branches, not a toy reward
        # calculation. Four worlds are deliberately CPU-bounded.
        from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN
        from emu.environment import EmuMJXEnv
        import jax.numpy as jnp
        import jax
        env=EmuMJXEnv(
            str(ROBOT_XML),num_envs=4,seed=735,
            terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
            unified_capability_focus=True,terrain_carpet=True,
            terrain_carpet_prisms=True,academy_stream=True,
            terrain_lesson="coherent-careful",
            solver_iterations=2,solver_ls_iterations=4,
            science_gate_mode="zero-only",
            science_phase_lowcmd_scale=.25,
            science_phase_sync_strength=0.,
            science_crouch_allowance_m=.03,
            science_stride_efficiency_weight=.08,
            science_supported_cross_bonus=6.,
            science_foothold_bonus=3.,
            science_bilateral_support_bonus=4.,
            science_followthrough_bonus=10.,
            science_sensor_teacher_bonus=3.,
        )
        packet=env.params_array(dict(specialist_mix=1.,
                                     focus_weight_step=1.,
                                     academy_episode_probability=1.,
                                     terrain_height_m=.02,
                                     academy_height_up_m=.02,
                                     academy_height_down_m=.02))
        state,obs=env.reset(dict(specialist_mix=1.,
                                 focus_weight_step=1.,
                                 academy_episode_probability=1.,
                                 terrain_height_m=.02,
                                 academy_height_up_m=.02,
                                 academy_height_down_m=.02))
        out=env.step(state,jnp.zeros((4,6),jnp.float32),packet)
        self.assertEqual(out.obs.shape,(4,360))
        self.assertTrue(np.isfinite(np.asarray(jax.device_get(out.reward))).all())
        self.assertEqual(out.state.academy_progress_origin_xy.shape,(4,2))
        self.assertIn("academy_followthrough_new",out.metrics)

    def test_real_stair_mission_requires_bilateral_support_and_progress(self):
        # Keep one positive and three negative physical verdicts. They
        # exercise REAL prism contact traces, not PPO reward counters.
        good=inspect_missions(trace(),max_followup_s=4.)
        self.assertEqual(good["eligible_supported_contacts"],1)
        self.assertEqual(good["completed_missions"],1)
        self.assertAlmostEqual(good["events"][0]["signed_actual_height_cm"],2.)
        for label,kwargs in (
            ("trailing",dict(second_foot=False)),
            ("no_forward",dict(continued=False)),
            ("bad_heading",dict(heading=np.pi/2)),
        ):
            with self.subTest(reason=label):
                verdict=inspect_missions(trace(**kwargs),max_followup_s=4.)
                self.assertEqual(verdict["completed_missions"],0)
                self.assertEqual(verdict["eligible_supported_contacts"],1)

    def test_fixed_wall_uses_real_mujoco_collision(self):
        from scripts.v5051_fixed_wall_screen import (
            make_wall_model, test_swing as wall_trial)
        model,geom=make_wall_model(.03)
        self.assertAlmostEqual(float(model.geom_pos[geom,2] +
                                     model.geom_size[geom,2]),.03,places=7)
        result=wall_trial(model,geom,.03,"left",.2,.4,settle=2,frames=12)
        self.assertEqual(result["actual_height_cm"],3.)
        self.assertLessEqual(result["max_servo_torque_nm"],1.914)

    def test_no_duplicate_arm_selection(self):
        self.assertEqual(select_arms("00_control")[0].name,"00_control")
        with self.assertRaises(ValueError):
            select_arms("00_control,00_control")
        with self.assertRaises(ValueError):
            select_arms("nonexistent")

    def test_gate_zero_only_keeps_motion_requirement(self):
        m=dict(age_s=np.array([2.,2.]),
               stepping_activity_so_far=np.array([0.,0.]),
               cmd_vx=np.array([0.,.25]),cmd_vy=np.zeros(2),
               cmd_yaw=np.zeros(2))
        _,legacy=_gait_maturity(m,np)
        _,zero=_gait_maturity(m,np,gate_mode="zero-only")
        _,ungated=_gait_maturity(m,np,gate_mode="off")
        np.testing.assert_allclose(legacy,[.3,.3])
        np.testing.assert_allclose(zero,[1.,.3])
        np.testing.assert_allclose(ungated,[1.,1.])

    def test_sync_off_reduces_only_phase_payment(self):
        m=dict(cmd_pitch=np.ones(1)*0.,height_error=np.zeros(1),
               attitude_quality=np.ones(1),foot_mid_cone=np.ones(1),
               bad_contact=np.zeros(1),support=np.ones(1),
               stance_slip=np.zeros(1),swing_clearance=np.zeros(1),
               target_step_height=np.ones(1)*.03,
               organic_step_shape=np.array([.7]),
               phase_sync=np.array([1.]),step_evidence=np.array([1.]),
               action_delta_norm=np.zeros(1),
               joint_limit_pressure=np.zeros(1),scissor_pressure=np.zeros(1))
        baseline=_physical_qualities(m,np)
        stripped=_physical_qualities(m,np,phase_sync_strength=0.)
        self.assertAlmostEqual(float(baseline[0][0]),float(stripped[0][0]))
        self.assertAlmostEqual(float(baseline[3][0]-stripped[3][0]),.135)
        # Crouch tolerance uses a teacher-only STEP flag; non-STEP unchanged.
        low=dict(m,height_error=np.array([-.045]),
                 science_step_focus=np.array([1.]))
        step=_physical_qualities(low,np,crouch_allowance_m=.03)
        plain=_physical_qualities(low,np)
        self.assertGreater(float(step[0][0]),float(plain[0][0]))
        low["science_step_focus"]=np.array([0.])
        no_step=_physical_qualities(low,np,crouch_allowance_m=.03)
        np.testing.assert_allclose(no_step[0],plain[0])


if __name__=="__main__":
    unittest.main()
