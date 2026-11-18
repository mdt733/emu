"""v5053 GPU-matrix contract: true per-world fixtures and honest score bins."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from dataclasses import asdict
import numpy as np

from scripts.v5053_matrix_exam import (
    assignments,reduce_results,wilson_95,compare_paired_exams,PAIRED_GATES)
from emu.physical_capability_exam import (
    Scenario, score_first_life, summarize, save_evidence)
from emu.terrain_carpet_mjx import (
    TILE_M,exact_exam_lip_profiles,tile_geometry)


class MatrixExamTests(unittest.TestCase):
    def test_ten_capabilities_twenty_levels_ten_trials(self):
        a=assignments(2048)
        self.assertEqual(len(a),2048)
        self.assertEqual(sum(x.kind!="padding" for x in a),2000)
        self.assertEqual(sum(x.kind=="padding" for x in a),48)
        self.assertEqual(len({(x.kind,x.level,x.replicate)
                              for x in a if x.kind!="padding"}),2000)
        self.assertEqual(len({x.kind for x in a if x.kind!="padding"}),10)
        steps=[x for x in a if x.kind=="step_up"]
        self.assertEqual(len(steps),200)
        self.assertAlmostEqual(min(x.exact_step_m for x in steps),.0025)
        self.assertAlmostEqual(max(x.exact_step_m for x in steps),.05)
        # Dedicated exact-height allocation: same world, different challenge.
        a=assignments(2048,mode="stairs")
        self.assertEqual(sum(x.kind=="step_up" for x in a),1000)
        self.assertEqual(sum(x.kind=="step_down" for x in a),1000)
        self.assertEqual(len({x.level for x in a if x.kind=="step_up"}),20)
        self.assertEqual(len({x.level for x in a if x.kind=="step_down"}),20)
        self.assertEqual(min(x.exact_step_m for x in a if x.kind=="step_down"),-.05)
        self.assertEqual(sum(x.kind=="step_up" for x in assignments(2048,mode="stairs_up")),2000)
        self.assertEqual(sum(x.kind=="step_down" for x in assignments(2048,mode="stairs_down")),2000)
        flat=assignments(128,mode="flat")
        self.assertEqual(sum(t.kind=="flat_control" for t in flat),100)
        self.assertTrue(all(t.exact_step_m==0. for t in flat))
        self.assertTrue(all(t.vx==.25 for t in flat if t.kind=="flat_control"))
        paired=assignments(2048,mode="flat",trials_per_level=2000)
        stairs=assignments(2048,mode="stairs")
        self.assertEqual(len([t for t in paired if t.kind=="flat_control"]),2000)
        self.assertEqual([t.world for t in paired],[t.world for t in stairs])
        self.assertTrue(all(t.vx==stairs[i].vx and t.vy==stairs[i].vy
                            and t.yaw==stairs[i].yaw for i,t in enumerate(paired)))
        with self.assertRaises(ValueError):
            assignments(2048,mode="flat",trials_per_level=2001)
        with self.assertRaisesRegex(ValueError,"Need at least"):
            assignments(128)
        with self.assertRaises(ValueError):
            assignments(2048,mode="unknown")
        # Grouped numerators must never average away labels or padding.
        a=assignments(4,mode="stairs_up",levels=1,trials_per_level=3)
        result=dict(
            actual_lip_m=np.array([.05,.05,.05,0.]),
            actual_grade_deg=np.zeros(4),
            survived=np.array([True,True,False,True]),
            lead_foot_top=np.array([True,True,False,False]),
            both_feet_top_five=np.array([True,False,False,False]),
            goal_line_crossed=np.array([True,True,False,False]),
            goal_line_after_both=np.array([True,False,False,False]),
            goal_x_window_seen=np.array([True,True,False,False]),
            goal_x_window_after_both=np.array([True,False,False,False]),
            first_both_foot_frame=np.array([8,-1,-1,-1]),
            first_both_foot_body_x_m=np.array([1.30,0,0,0]),
            traversed_upright_5frame=np.array([True,True,False,False]),
            traversed_in_lane_5frame=np.array([True,False,False,False]),
            goal_window_after_both=np.array([True,False,False,False]),
            goal_window_aligned=np.array([True,False,False,False]),
            goal_window_safe_except_slip=np.array([True,False,False,False]),
            goal_window_all_gates=np.array([True,False,False,False]),
            traversed_5frame=np.array([True,False,False,False]),
            completed_destination=np.array([True,False,False,False]),
            best_forward_progress_m=np.array([.82,.4,.1,0.]),
            speed_mae_mps=np.zeros(4),
            yaw_mae_rad_s=np.zeros(4))
        detail,curve=reduce_results(a,result)
        self.assertEqual(len(detail),3)
        self.assertEqual(len(curve),1)
        row=curve[0]
        self.assertEqual(row["leading_top"],2)
        self.assertEqual(row["each_foot_top_five"],1)
        self.assertEqual(row["completed_destinations"],1)
        self.assertEqual(row["goal_line_crossed"],2)
        self.assertEqual(row["goal_line_after_both"],1)
        self.assertEqual(row["goal_x_window_seen"],2)
        self.assertEqual(row["goal_x_window_after_both"],1)
        self.assertEqual(row["traversed_upright_5frame"],2)
        self.assertEqual(row["traversed_in_lane_5frame"],1)
        self.assertAlmostEqual(row["first_both_foot_x_mean_m"],1.3)
        self.assertEqual(row["traversed_5frame"],1)
        self.assertAlmostEqual(row["traversal_rate"],1/3,places=4)
        self.assertAlmostEqual(row["completion_rate"],1/3,places=4)
        self.assertLess(row["completion_wilson95_lo"],1/3)
        self.assertGreater(row["completion_wilson95_hi"],1/3)
        self.assertEqual(wilson_95(0,0),(0.,0.))
        flat_records=assignments(4,mode="flat",trials_per_level=3)
        flat_result={k:np.array(v,copy=True) for k,v in result.items()}
        flat_result["actual_lip_m"]=np.zeros(4)
        _,flat_curve=reduce_results(flat_records,flat_result)
        self.assertEqual(flat_curve[0]["kind"],"flat_control")
        self.assertEqual(flat_curve[0]["completed_destinations"],1)
        self.assertEqual(flat_curve[0]["traversed_5frame"],1)



    def test_same_index_paired_flat_comparison_rejects_false_pairs(self):
        stairs=assignments(4,mode="stairs",levels=1,trials_per_level=2)
        flats=assignments(4,mode="flat",trials_per_level=4)
        def row(t,ok):
            record=asdict(t)
            record["actual_lip_m"]=t.exact_step_m
            record["speed_mae_mps"]=.02 if t.kind=="flat_control" else .05
            for key in PAIRED_GATES:
                record[key]=str(ok)  # same strings read from CSV
            return record
        sr=[row(t,i%2==0) for i,t in enumerate(stairs)]
        fr=[row(t,True) for t in flats]
        common=dict(schema="h40_true_prism_heterogeneous_matrix_v4",
                    policy_sha256="policy",actor_sha256="weights",
                    robot_model_sha256="model",actor_training_source={"step":120061952},
                    exam_source_sha256="same-exam-source",
                    seed=1701,worlds=4,horizon=350,device="gpu",
                    collision="8 true prisms",first_life_only=True,
                    zero_optimizer_updates=True,assigned_worlds=4)
        up={"manifest":dict(common,mode="stairs")}
        flat={"manifest":dict(common,mode="flat")}
        pairs,curves=compare_paired_exams(up,sr,flat,fr)
        self.assertEqual(len(pairs),4)
        self.assertEqual(len(curves),2)
        for entry in curves:
            self.assertEqual(entry["pairs"],2)
            self.assertEqual(entry["traversed_5frame_stair"],1)
            self.assertEqual(entry["traversed_5frame_flat"],2)
            self.assertEqual(entry["traversed_5frame_lost"],1)
            self.assertEqual(entry["traversed_5frame_gained"],0)
            self.assertEqual(entry["traversed_5frame_difference_pct"],-50.)
            self.assertAlmostEqual(entry["speed_mae_excess_mps"],.03)
        with self.assertRaisesRegex(ValueError,"exam_source_sha256"):
            compare_paired_exams(up,sr,{"manifest":dict(flat["manifest"],exam_source_sha256="different")},fr)
        with self.assertRaisesRegex(ValueError,"seed"):
            compare_paired_exams(up,sr,{"manifest":dict(flat["manifest"],seed=99)},fr)
        with self.assertRaisesRegex(ValueError,"misindexed"):
            compare_paired_exams(up,sr,flat,fr[:-1])
        with self.assertRaisesRegex(ValueError,"unmatched vx"):
            compare_paired_exams(up,sr,flat,[dict(fr[0],vx=.12),*fr[1:]])
        with self.assertRaisesRegex(ValueError,"mismatched physical seam"):
            compare_paired_exams(up,sr,flat,[dict(fr[0],actual_lip_m=.01),*fr[1:]])
        with self.assertRaisesRegex(ValueError,"Unrecognised physical"):
            compare_paired_exams(up,[dict(sr[0],traversed_5frame="maybe"),*sr[1:]],flat,fr)

        # The new v1 first-life scoring checks share the existing physical
        # examiner contract case; no test-per-experiment growth.
        v1 = _PhysicalV1Checks()
        for check in ["verify_zero_command_real_standing","verify_first_fall_excludes_subsequent_reset_lives","verify_walk_away_is_not_standing","verify_requested_trajectory_and_actor_are_distinct","verify_static_survival_cannot_pass_walking","verify_early_fall_cannot_pass_walking","verify_reject_wrong_operator_commands","verify_transition_requires_real_stop","verify_perfect_transition_passes","verify_partial_scenario_cannot_become_complete_summary","verify_zip_does_not_overwrite_prior_exam"]:
            getattr(v1, check)()

    def test_signed_exact_step_and_continuous_grade_geometry(self):
        heights=np.array([.0025,.05,-.0375,0.],np.float32)
        x,y=exact_exam_lip_profiles(np,heights)
        self.assertEqual(x.shape,(4,256))
        np.testing.assert_allclose(y,0.)
        np.testing.assert_allclose(x[:,128],0.)
        np.testing.assert_allclose(x[:,129],heights)
        seed=np.zeros(4,dtype=np.int32)
        zeros=np.zeros(4,dtype=np.float32)
        grades=np.array([0.,0.,0.,12.],np.float32)
        x0=np.zeros(4,dtype=np.int32)
        x1=np.ones(4,dtype=np.int32)
        for tile in (x0,x1):
            centre,vertices=tile_geometry(
                np,tile,x0,seed,x,y,zeros,zeros,zeros,
                exact_grade_x=grades)
            self.assertEqual(vertices.shape,(4,4))
            self.assertTrue(np.isfinite(centre).all())
        c0,h0=tile_geometry(np,x0,x0,seed,x,y,zeros,zeros,zeros,
                            exact_grade_x=grades)
        c1,h1=tile_geometry(np,x1,x0,seed,x,y,zeros,zeros,zeros,
                            exact_grade_x=grades)
        def real_edge(c,h,which):
            return c[:,2]+.5*.5*(h[:,which[0]]+h[:,which[1]])
        measured=real_edge(c1,h1,(0,2))-real_edge(c0,h0,(1,3))
        np.testing.assert_allclose(measured,heights,atol=1e-6)
        grad=(h0[:,1]-h0[:,0])*.5/TILE_M
        self.assertAlmostEqual(float(np.rad2deg(np.arctan(grad[-1]))),
                               12.,places=4)


    def test_true_eight_prism_matrix_compiles_and_measures_exacts(self):
        # On CI: only 4 worlds x 3 frames (no PPO updates), preventing
        # a fake NumPy-only test from certifying a broken batched MJX API.
        from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
        from emu.environment import EmuMJXEnv
        from emu.ppo import make_trainer
        from emu.training_config import (
            PPO_LR,ROLLOUT_STEPS,PPO_ENTROPY_COST,PPO_VALUE_COST)
        import jax
        import jax.numpy as jnp
        env=EmuMJXEnv(
            str(ROBOT_XML),num_envs=4,seed=1701,
            observation_schema=OBS_SCHEMA_TERRAIN,
            unified_capability_focus=True,
            terrain_carpet=True,terrain_carpet_prisms=True,
            terrain_step_max_m=.08,academy_stream=True,
            science_step_finish_on_transfer=True,
            solver_iterations=2,solver_ls_iterations=4,
            exam_fixture_steps_m=np.array([0.,.0025,.02,-.05]),
            exam_fixture_grades_deg=np.array([0.,0.,0.,0.]))
        teacher=dict(specialist_mix=0.,
            regime_frac_steady=1.,regime_frac_transition=0.,
            regime_frac_resilience=0.,
            probe_cmd_vx=0.,probe_cmd_vy=0.,probe_cmd_yaw=0.,
            probe_push_n=0.,probe_tilt_rad=0.,
            probe_linvel_mps=0.,probe_angvel_rad_s=0.,
            probe_joint_rad=0.,academy_episode_probability=0.)
        task=env.params_array(teacher)
        trainer=make_trainer(env,seed=1701,lr=PPO_LR,rollout=ROLLOUT_STEPS,
            epochs=1,minibatches=1,
            entropy_cost=PPO_ENTROPY_COST,value_cost=PPO_VALUE_COST)
        init=trainer["make_initial_state"](teacher)
        # Exact H40 one-metre tile: 22cm upstream and centred in Y.
        spawn_xy=np.asarray(jax.device_get(init.env_state.data.qpos[:,:2]))
        np.testing.assert_allclose(spawn_xy[:,0],.78,atol=1e-6)
        np.testing.assert_allclose(spawn_xy[:,1],.50,atol=1e-6)
        # The new success/reset curriculum must retain the EXACT first
        # signed physical seam at birth, even after a later foot transfer.
        np.testing.assert_allclose(
            np.asarray(jax.device_get(init.env_state.academy_birth_lip_m)),
            [0.,.0025,.02,-.05],atol=.0005)
        # The difficulty reference is captured at the SAME birth;
        # an eventual promotion must never relabel this older trial.
        birth_ref=np.asarray(jax.device_get(
            init.env_state.academy_birth_frontier_m))
        assert birth_ref.shape==(4,)
        assert np.all(birth_ref>0.)
        out=trainer["evaluate_matrix"](
            init.params,task,jnp.zeros((360,),dtype=jnp.float32),
            jnp.ones((360,),dtype=jnp.float32),
            jnp.array([[0.,0.,0.],[.15,0.,0.],[.25,0.,0.],
                       [.25,0.,0.]],dtype=jnp.float32),
            jnp.asarray([False,True,True,True]),
            jax.random.PRNGKey(1701),3)
        actual=np.asarray(jax.device_get(out["actual_lip_m"]))
        np.testing.assert_allclose(
            actual,[0.,.0025,.02,-.05],atol=.0005)
        self.assertEqual(out["lead_foot_top"].shape,(4,))
        self.assertEqual(out["completed_destination"].shape,(4,))
        self.assertEqual(out["traversed_5frame"].shape,(4,))
        self.assertEqual(out["goal_window_safe_except_slip"].shape,(4,))
        self.assertEqual(out["goal_line_after_both"].shape,(4,))
        self.assertEqual(out["goal_x_window_after_both"].shape,(4,))
        self.assertEqual(out["first_both_foot_frame"].shape,(4,))
        self.assertEqual(out["traversed_upright_5frame"].shape,(4,))
        self.assertEqual(out["traversed_in_lane_5frame"].shape,(4,))
        # Integration smoke: exercise the *real* exported 120M actor through
        # the actual JAX trace endpoint, rather than only synthetic NumPy
        # scoring. Three frames are API/physics evidence, NOT certification.
        from emu.policy_export import load_exported_actor_for_exam
        from pathlib import Path
        policy, norm, actor = load_exported_actor_for_exam(
            Path("showcase/champion.emupolicy"))
        self.assertEqual(actor["obs_dim"], 360)
        trace = trainer["evaluate_trace"](
            policy, task, jnp.asarray(norm["mean"]),
            jnp.asarray(norm["std"]), jax.random.PRNGKey(1711),
            3, 0., command_schedule=jnp.zeros((3, 3), dtype=jnp.float32))
        self.assertEqual(trace["done"].shape, (3, 4))
        self.assertEqual(trace["physical_position_xy_after_step"].shape, (3, 4, 2))
        self.assertTrue(np.isfinite(np.asarray(jax.device_get(
            trace["physical_position_xy_after_step"]))).all())


def physical_v1_fixture(seconds=1, worlds=2):
    n = 50 * seconds
    zeros = np.zeros((n, worlds))
    pos = np.zeros((n, worlds, 2))
    return dict(
        done=np.zeros((n, worlds), dtype=bool),
        fell=np.zeros((n, worlds), dtype=bool),
        position_xy=pos.copy(),
        physical_position_xy_after_step=pos.copy(),
        heading_rad=zeros.copy(),
        physical_heading_rad_after_step=zeros.copy(),
        command=np.zeros((n, worlds, 3)),
        terrain_gravity_z=np.ones((n, worlds)),
        terrain_bad_contact=zeros.copy(),
        floor_contact=np.ones((n, worlds, 2), dtype=bool),
        servo_shaft_drive_power_w=zeros.copy(),
    )


class _PhysicalV1Checks(unittest.TestCase):
    def verify_zero_command_real_standing(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        rows = score_first_life(physical_v1_fixture(), case)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(x["bounded_standing_pass"] for x in rows))
        self.assertTrue(all(x["survived_full_horizon"] for x in rows))
        self.assertAlmostEqual(rows[0]["final_drift_m"], 0.)
        self.assertEqual(rows[0]["physical_touchdown_events"], 0)
        self.assertEqual(rows[0]["standing_failure_reasons"], "")

    def verify_first_fall_excludes_subsequent_reset_lives(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        x = physical_v1_fixture()
        x["done"][6, 1] = True
        x["fell"][6, 1] = True
        x["physical_position_xy_after_step"][7:, 1, 0] = 99.
        rows = score_first_life(x, case)
        self.assertEqual(rows[1]["first_life_frames"], 7)
        self.assertAlmostEqual(rows[1]["time_to_first_fall_s"], .14)
        self.assertEqual(rows[1]["terminal_reason"], "fall")
        self.assertIn("terminal", rows[1]["standing_failure_reasons"])
        self.assertFalse(rows[1]["survived_full_horizon"])
        self.assertFalse(rows[1]["bounded_standing_pass"])
        self.assertAlmostEqual(rows[1]["max_radius_m"], 0.)
        self.assertTrue(rows[0]["survived_full_horizon"])

    def verify_walk_away_is_not_standing(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        x = physical_v1_fixture()
        x["physical_position_xy_after_step"][:, 0, 0] = (
            .01 * np.arange(1, 51))
        rows = score_first_life(x, case)
        self.assertTrue(rows[0]["survived_full_horizon"])
        self.assertFalse(rows[0]["bounded_standing_pass"])
        self.assertIn("position_drift", rows[0]["standing_failure_reasons"])
        self.assertAlmostEqual(rows[0]["max_radius_m"], .5)

    def verify_requested_trajectory_and_actor_are_distinct(self):
        case = Scenario("forward", 1, ((1, .25, 0., 0.),),
                        "locomotion")
        x = physical_v1_fixture()
        x["command"][:, :, 0] = .25
        x["physical_position_xy_after_step"][:, :, 0] = (
            .005 * np.arange(1, 51)[:, None])
        rows = score_first_life(x, case)
        self.assertAlmostEqual(rows[0]["vx_mae_mps"], 0., places=5)
        self.assertAlmostEqual(rows[0]["final_trajectory_error_m"],
                               0., places=5)
        self.assertTrue(rows[0]["locomotion_mission_pass"])
        self.assertEqual(rows[0]["locomotion_failure_reasons"], "")

    def verify_static_survival_cannot_pass_walking(self):
        case = Scenario("forward", 1, ((1, .25, 0., 0.),),
                        "locomotion")
        x = physical_v1_fixture()
        x["command"][:, :, 0] = .25
        rows = score_first_life(x, case)
        self.assertTrue(rows[0]["survived_full_horizon"])
        self.assertFalse(rows[0]["locomotion_mission_pass"])
        self.assertIn("vx_tracking", rows[0]["locomotion_failure_reasons"])

    def verify_early_fall_cannot_pass_walking(self):
        case = Scenario("forward", 1, ((1, .25, 0., 0.),),
                        "locomotion")
        x = physical_v1_fixture()
        x["command"][:, :, 0] = .25
        x["physical_position_xy_after_step"][:, :, 0] = (
            .005 * np.arange(1, 51)[:, None])
        x["done"][4, 0] = True
        x["fell"][4, 0] = True
        rows = score_first_life(x, case)
        self.assertFalse(rows[0]["locomotion_mission_pass"])
        self.assertIn("terminal", rows[0]["locomotion_failure_reasons"])
        self.assertTrue(rows[1]["locomotion_mission_pass"])
        self.assertEqual(rows[0]["first_life_frames"], 5)

    def verify_reject_wrong_operator_commands(self):
        case = Scenario("forward", 1, ((1, .25, 0., 0.),),
                        "locomotion")
        x = physical_v1_fixture()
        # A trace delivering no operator command cannot be credited as
        # forward walking merely because its pose happened to move.
        x["physical_position_xy_after_step"][:, :, 0] = (
            .005 * np.arange(1, 51)[:, None])
        with self.assertRaisesRegex(ValueError, "operator commands differ"):
            score_first_life(x, case)

    def verify_transition_requires_real_stop(self):
        case = Scenario("forward_stop_reverse_30s", 30,
                        ((10, .25, 0., 0.), (10, 0., 0., 0.),
                         (10, -.20, 0., 0.)), "transitions")
        x = physical_v1_fixture(seconds=30)
        x["command"][:, :, :] = case.command_schedule()[:, None, :]
        # A policy moving forward through the middle rest period.
        x["physical_position_xy_after_step"][:, :, 0] = (
            .005 * np.arange(1, 1501)[:, None])
        rows = score_first_life(x, case)
        self.assertGreater(rows[0]["stop_interval_mean_speed_mps"], .20)
        self.assertFalse(rows[0]["locomotion_mission_pass"])
        self.assertIn("failed_to_stop", rows[0]["locomotion_failure_reasons"])

    def verify_perfect_transition_passes(self):
        case = Scenario("forward_stop_reverse_30s", 30,
                        ((10, .25, 0., 0.), (10, 0., 0., 0.),
                         (10, -.20, 0., 0.)), "transitions")
        x = physical_v1_fixture(seconds=30)
        schedule = case.command_schedule()
        x["command"][:] = schedule[:, None, :]
        x["physical_position_xy_after_step"][:, :, 0] = (
            np.cumsum(schedule[:, 0])[:, None] * .02)
        rows = score_first_life(x, case)
        self.assertTrue(all(r["locomotion_mission_pass"] for r in rows))
        self.assertAlmostEqual(rows[0]["stop_interval_mean_speed_mps"], 0.)
        summary = summarize(rows, expected_worlds=2)
        self.assertEqual(summary[0]["locomotion_mission_passes"], 2)

    def verify_partial_scenario_cannot_become_complete_summary(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        rows = score_first_life(physical_v1_fixture(), case)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(rows[:-1], expected_worlds=2)

    def verify_zip_does_not_overwrite_prior_exam(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        rows = score_first_life(physical_v1_fixture(), case)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.zip"
            save_evidence(path, {"schema": "test"}, rows,
                          summarize(rows, expected_worlds=2))
            self.assertTrue(path.is_file())
            with self.assertRaises(FileExistsError):
                save_evidence(path, {"schema": "test"}, rows,
                              summarize(rows, expected_worlds=2))



if __name__=="__main__":
    unittest.main()
