from pathlib import Path

def test_production_lineage_is_monotonic_and_ends_400m():
    from emu.production import PHASES
    targets=[x[1] for x in PHASES]
    assert targets == sorted(targets)
    assert targets[-1] == 400_000_000
    assert PHASES[0][2:] == (0.0,0.0,0.0)
    assert PHASES[-1][2:] == (0.75,0.70,0.75)

def test_operator_pitch_keeps_220_observation_contract():
    from emu.contracts import BASE_OBS_DIM, OBS_DIM, HISTORY_LEN, BASE_OBS_FIELDS, PITCH_CMD_MAX_RAD
    assert 'cmd_pitch' in BASE_OBS_FIELDS
    assert BASE_OBS_DIM == 55 and HISTORY_LEN == 4 and OBS_DIM == 220
    assert 0.12 <= PITCH_CMD_MAX_RAD <= 0.18

def test_viewer_joystick_signs_match_operator_semantics():
    root=Path(__file__).resolve().parents[1]/'viewer'/'src'
    main=(root/'main.js').read_text()
    sim=(root/'sim.js').read_text()
    assert 'const vy = left.x*l.vy_abs;' in main
    assert 'const yaw = -right.x*l.yaw_abs;' in main
    assert 'this.commandTarget.vy = leftX * lim.vy_abs;' in sim
    assert 'this.commandTarget.yaw = -rightX * lim.yaw_abs;' in sim
    assert 'dropOverlay.hidden = true;' in main
