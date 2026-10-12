from pathlib import Path

from emu.tournament import (
    ExperimentSpec, base_specs, mechanism_specs, schedule_specs,
    pick_best, gate, utility,
)


def fake_summary(*, broad=.80, hard=.30, growth=.60, recovery=.50, worst=.20, cal=.30):
    return {
        'broad': broad, 'screen_hard_recovery': hard,
        'screen_growth_headroom': growth, 'recovery': recovery,
        'worst_direction': worst, 'calibration_mae': cal,
    }


def test_tournament_defines_exactly_twelve_screen_hypotheses():
    shadow, doses = base_specs()
    mechs = mechanism_specs(.25)
    schedules = schedule_specs(mechs[0])
    specs = [shadow, *doses, *mechs, *schedules]
    assert len(specs) == 12
    assert len({s.name for s in specs}) == 12
    assert [s.dose for s in doses] == [.10, .25, .40, .65]


def test_mechanism_arms_are_orthogonal():
    b1,b2,b3,b4 = mechanism_specs(.25)
    assert (b1.teach_nodes,b1.teach_edges,b1.teach_recovery,b1.evidence_from_taught) == (True,False,True,True)
    assert (b2.teach_nodes,b2.teach_edges,b2.teach_recovery,b2.evidence_from_taught) == (False,True,False,True)
    assert (b3.teach_nodes,b3.teach_edges,b3.teach_recovery,b3.evidence_from_taught) == (True,True,False,True)
    assert b4.evidence_from_taught is False


def test_schedule_arms_encode_slow_delayed_and_release():
    base = mechanism_specs(.25)[0]
    c1,c2,c3 = schedule_specs(base)
    assert (c1.influence_start,c1.influence_full) == (20_000_000,60_000_000)
    assert (c2.influence_start,c2.influence_full) == (40_000_000,60_000_000)
    assert (c3.influence_start,c3.influence_full,c3.fade_start,c3.zero_at) == (20_000_000,30_000_000,45_000_000,60_000_000)


def test_experiment_spec_promotes_only_intended_training_fields():
    spec = mechanism_specs(.25)[3]
    cfg = spec.config(run_dir=Path('/tmp/x'),steps=50_000_000,seed=733,device='cuda',envs=2048,resume=Path('/tmp/a.pkl'))
    assert cfg.field_max_influence == .25
    assert cfg.field_evidence_from_taught is False
    assert cfg.field_teach_nodes and cfg.field_teach_edges and cfg.field_teach_recovery
    assert cfg.capability_field_mode == 'graph'


def test_promotion_gate_protects_recovery_even_when_broad_is_better():
    shadow=fake_summary(broad=.80,hard=.30)
    specialist=fake_summary(broad=.86,hard=.15)
    ok,reasons=gate(specialist,shadow)
    assert not ok
    assert any('hard recovery' in r for r in reasons)


def test_dose_selection_prefers_lower_dose_when_nearly_tied():
    specs=[ExperimentSpec('lo','lo',dose=.10),ExperimentSpec('hi','hi',dose=.40)]
    shadow=fake_summary()
    # Utilities are deliberately within the 0.015 efficiency tie band.
    summaries={'lo':fake_summary(broad=.800),'hi':fake_summary(broad=.805)}
    selected,_=pick_best(specs,summaries,shadow,prefer_lower_dose_within=.015)
    assert selected.name == 'lo'
    assert utility(summaries['hi']) >= utility(summaries['lo'])


def test_screening_uses_the_v5012_diagnostic_recovery_growth_coordinates():
    from emu.screening import SCREEN_PANELS
    assert set(SCREEN_PANELS) == {'resilience100','robust70','growth140','growthcombo140'}
