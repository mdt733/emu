"""GPU-free measurement semantics shared by curriculum and examiner.

No reward weights, actor inputs or checkpoint formats are involved.
"""
import numpy as np
import pytest

from emu.scoring import terrain_edge_trials, graded_step_score, ExactStepResult


def _events(done):
    done=np.asarray(done,dtype=np.float32)
    return {
        "terrain_edge_mask":np.ones_like(done),
        "terrain_edge_done":done,
        "terrain_edge_crossing":np.zeros_like(done),
        "terrain_edge_contact":np.zeros_like(done),
        "terrain_edge_fell":np.zeros_like(done),
        "terrain_height_birth_m":np.full_like(done,.020),
    }


def test_unfinished_episodes_are_unknown_not_failures():
    row=terrain_edge_trials(_events(np.zeros((32,3))))
    assert row["terrain_edge_exposure"] == 1.
    assert row["terrain_edge_done_count"] == 0.
    assert row["terrain_edge_crossing"] == -1.
    assert row["terrain_edge_contact"] == -1.
    assert row["terrain_edge_fall_rate"] == -1.
    assert row["terrain_height_birth_m"] == pytest.approx(.020)
    assert terrain_edge_trials({}) == {}


def test_completed_edge_trials_have_correct_denominators_and_counts():
    m=_events([[0,0,0],[1,0,1],[0,1,0]])
    m["terrain_edge_crossing"][1,0]=1.
    m["terrain_edge_contact"][1,0]=1.
    m["terrain_edge_contact"][1,2]=1.
    m["terrain_edge_fell"][2,1]=1.
    row=terrain_edge_trials(m)
    assert row["terrain_edge_done_count"] == 3.
    assert row["terrain_edge_completed_fraction"] == pytest.approx(3/9)
    assert row["terrain_edge_crossing"] == pytest.approx(1/3)
    assert row["terrain_edge_contact"] == pytest.approx(2/3)
    assert row["terrain_edge_fall_rate"] == pytest.approx(1/3)


def test_invalid_training_event_shapes_cannot_be_reported_as_success():
    m=_events([[0,1],[1,0]])
    m["terrain_edge_crossing"]=np.ones((4,))
    with pytest.raises(ValueError,match="terrain_edge_crossing"):
        terrain_edge_trials(m)


def _exam(heights=(5,10,20)):
    d={"physical_step_exact_height":True}
    for mm in heights:
        d[f"course_{mm}mm_attempt_count"]=32.
        d[f"course_{mm}mm_contact_rate"]=.5
        d[f"course_{mm}mm_crossing_rate"]=.25 if mm==20 else 0.
    return d


def test_exact_step_grade_measures_completed_trails_not_training_rewards():
    heights=(5,10,20)
    weights=(.2,.2,.6)
    evidence=_exam(heights)
    evidence["reward_mean"]=1000000.
    score,trials=graded_step_score(evidence,heights,weights)
    assert score == pytest.approx(.15)
    assert all(isinstance(t,ExactStepResult) for t in trials)
    assert [t.attempts for t in trials] == [32,32,32]
    evidence["reward_mean"]=-1000000.
    assert graded_step_score(evidence,heights,weights)[0] == score


@pytest.mark.parametrize("change", [
    {"physical_step_exact_height":False},
    {"course_20mm_attempt_count":0.},
    {"course_20mm_attempt_count":7.},
    {"course_20mm_attempt_count":4.5},
    {"course_20mm_contact_rate":.01},
    {"course_20mm_crossing_rate":-1.},
    {"course_20mm_crossing_rate":float("nan")},
])
def test_exam_never_certifies_invalid_or_sparse_physical_evidence(change):
    e=_exam()
    e.update(change)
    with pytest.raises((ValueError,TypeError)):
        graded_step_score(e,(5,10,20),(.2,.2,.6))



def test_success_cannot_appear_before_trial_completes():
    m=_events([[0,0],[0,1]])
    m["terrain_edge_crossing"][0,0]=1.
    with pytest.raises(ValueError,match="without completed trial"):
        terrain_edge_trials(m)



def test_physical_peak_is_per_world_per_frame_not_a_misleading_mean():
    from emu.results import physical_peak_metric
    data={"spring_potential_energy_j":np.full((32,4096),.8)}
    data["spring_potential_energy_j"][11,1337]=2500000.
    peak=physical_peak_metric(data,"spring_potential_energy_j")
    assert peak==2500000.
    assert np.mean(data["spring_potential_energy_j"]) < 20.
    assert physical_peak_metric({},"spring_potential_energy_j")==-1.
    data["spring_potential_energy_j"][11,1337]=float("nan")
    assert np.isnan(physical_peak_metric(data,"spring_potential_energy_j"))


@pytest.mark.parametrize("change", [
    {"progress_m": 0.}, {"commanded_m": 0.}, {"gait": 0.},
    {"upright": .3}, {"support": 0.}, {"fell": True},
    {"observed_steps": 349}, {"vx_mae": .10}, {"progress_m": float("inf")},
])
def test_joint_walking_evidence_rejects_standing_falls_and_incomplete(change):
    from emu.scoring import basic_walking_pass
    values = dict(observed_steps=np.array([350]), horizon=350,
                  fell=np.array([False]), progress_m=.3, commanded_m=.7,
                  gait=.2, upright=.95, support=.7, vx_mae=.05)
    assert basic_walking_pass(np, **values).all()
    assert not basic_walking_pass(np, **(values | change)).any()


@pytest.mark.parametrize("early_fall", [True, False])
def test_compiled_evaluator_excludes_reset_aftermath(early_fall):
    """Exercise the actual JIT evaluator with controlled physical telemetry."""
    import jax
    import jax.numpy as jnp
    from typing import NamedTuple
    from emu.ppo import make_trainer
    from emu.contracts import CONTROL_DT
    from emu.evaluation import EVAL_FIELDS

    class Output(NamedTuple):
        state: object
        obs: object
        metrics: object
        done: object

    class ExamWorld:
        num_envs = 3
        def reset(self, params):
            return jnp.array(0), self.obs(None)
        def _reset_with_keys(self, keys, params):
            return jnp.array(0)
        def obs(self, state):
            return jnp.zeros((3, 220))
        def step(self, state, action, params):
            # World 0 stands, world 1 walks, world 2 falls on frame 1
            # then looks perfect forever. Only world 1 should qualify.
            one = jnp.ones(3)
            zero = jnp.zeros(3)
            vx = jnp.array([0., .10, .10])
            fell = jnp.array([0., 0., 1.]) * (state == 0)
            metrics = dict(vx=vx, vy=zero, yaw_rate=zero, cmd_vx=one*.10,
                cmd_vy=zero, cmd_yaw=zero, gravity_z=one, foot_mid_cone=one,
                step_evidence=jnp.array([0., .3, .3]), support=one,
                body_pitch=zero, cmd_pitch=zero, height_error=zero,
                episode_fell=fell*early_fall, terrain_forward_delta_m=vx*CONTROL_DT)
            return Output(state+1, self.obs(None), metrics, fell)

    trainer = make_trainer(ExamWorld(), seed=733, lr=0.0003, rollout=1, epochs=1, minibatches=1)
    state = trainer['make_initial_state']({})
    result = trainer['evaluate_summary'](state.params, {}, jnp.zeros(220),
        jnp.ones(220), jax.random.PRNGKey(0), 350)
    result = dict(zip(EVAL_FIELDS, np.asarray(result)))
    assert result['walking_pass'] == pytest.approx(1/3)
    assert result['survival'] == pytest.approx(2/3)
    # Only one frame of world 2 counts toward gait/command evidence.
    assert result['step_evidence'] == pytest.approx(.3*351/701, abs=1e-6)


def test_compiled_terrain_exam_revokes_crossing_when_robot_later_falls():
    import jax
    import jax.numpy as jnp
    from typing import NamedTuple
    from emu.ppo import make_trainer
    from emu.environment import T, TASK_FIELDS

    class State(NamedTuple):
        frame: object
        task: object
        prev_action: object
    class Output(NamedTuple):
        state: object
        obs: object
        metrics: object
        done: object
    class World:
        num_envs = 2
        terrain_adaptive = True
        def _reset_with_keys(self, keys, params):
            task = jnp.zeros((2,len(TASK_FIELDS)))
            task = task.at[:,T.terrain_height_m].set(.02)
            task = task.at[:,T.terrain_birth_frontier_m].set(.02)
            return State(jnp.array(0),task,jnp.zeros((2,6)))
        def reset(self, params):
            return self._reset_with_keys(None,params), self.obs(None)
        def obs(self, state):
            return jnp.zeros((2,220))
        def step(self, state, action, params):
            fell = jnp.array([1.,0.]) * (state.frame == 1)
            metrics = dict(terrain_box_foot_contact_any=jnp.ones(2),
                terrain_crossed_this_frame=jnp.ones(2), episode_fell=fell)
            metrics.update(servo_shaft_abs_power_w=jnp.ones(2)*10*(state.frame == 0),
                servo_shaft_drive_power_w=jnp.ones(2)*5*(state.frame == 0),
                servo_over_rated_frac=jnp.zeros(2), servo_peak_torque_limit_frac=jnp.zeros(2))
            return Output(State(state.frame+1,state.task,state.prev_action),self.obs(None),metrics,fell)
    trainer = make_trainer(World(), seed=733, lr=.0003, rollout=1, epochs=1, minibatches=1)
    initial = trainer['make_initial_state']({})
    result = trainer['evaluate_terrain_summary'](initial.params, {},
        jnp.zeros(220),jnp.ones(220),jax.random.PRNGKey(0),4)
    np.testing.assert_allclose(np.asarray(result), [2.,1.,.5])


    effort = trainer['evaluate_efficiency'](initial.params, {},
        jnp.zeros(220),jnp.ones(220),jax.random.PRNGKey(0),4)
    # Six observed first-episode frames, never the two post-fall frames.
    np.testing.assert_allclose(np.asarray(effort)[:2], [20/6,10/6], rtol=1e-6)


@pytest.mark.parametrize("changes", [
    {"terrain_edge_mask": 0.},
    {"terrain_edge_crossing": 1.},
    {"terrain_edge_contact": 1., "terrain_edge_crossing": 1., "terrain_edge_fell": 1.},
    {"terrain_edge_contact": .5},
])
def test_contradictory_or_fractional_terrain_events_are_rejected(changes):
    metrics = _events([1.])
    for key, value in changes.items():
        metrics[key][:] = value
    with pytest.raises(ValueError):
        terrain_edge_trials(metrics)
