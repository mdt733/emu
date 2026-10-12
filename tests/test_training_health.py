from emu.health import TrainingHealth


def test_kl_gate_becomes_chronic_without_permanent_curriculum_freeze():
    h = TrainingHealth()
    for _ in range(h.KL_STRESS_CHRONIC_WINDOW - 1):
        r = h.observe_ppo({'ppo_kl_stopped_frac': 0.2})
        assert r['kl_ok_to_expand'] is False
        assert r['kl_stress_chronic'] == 0.0
        assert r['kl_became_chronic'] is False

    r = h.observe_ppo({'ppo_kl_stopped_frac': 0.2})
    assert r['kl_stress_chronic'] == 1.0
    assert r['kl_became_chronic'] is True
    assert r['kl_ok_to_expand'] is True


def test_collapse_request_is_health_state_not_curriculum_state():
    h = TrainingHealth()
    for _ in range(h.COLLAPSE_STREAK_LIMIT - 1):
        r = h.observe_collapse(0.8)
        assert r['needs_recovery'] is False
    r = h.observe_collapse(0.8)
    assert r['needs_recovery'] is True
    h.note_recovery_performed()
    assert h.stepping_collapse_streak == 0


def test_health_roundtrip_preserves_next_decision_state():
    h = TrainingHealth()
    for _ in range(17):
        h.observe_ppo({'ppo_kl_stopped_frac': 0.2})
    for _ in range(9):
        h.observe_collapse(0.8)
    state = h.state_dict()

    restored = TrainingHealth()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.observe_ppo({'ppo_kl_stopped_frac': 0.0}) == h.observe_ppo({'ppo_kl_stopped_frac': 0.0})
    assert restored.observe_collapse(0.0) == h.observe_collapse(0.0)


def test_health_state_is_exact_not_best_effort():
    h = TrainingHealth()
    state = h.state_dict()
    del state['stepping_collapse_streak']
    try:
        TrainingHealth().load_state_dict(state)
    except ValueError:
        return
    raise AssertionError('partial training-health state was accepted')
