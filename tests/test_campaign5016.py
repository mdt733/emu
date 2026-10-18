"""Fast, GPU-free contracts for the bounded v5016 pinnacle campaign."""
import json
from pathlib import Path

import pytest

from emu.campaign5016 import (
    VERSION, ARMS, POLISH_TARGETS, score, phase_config, budget_allows,
    report, main,
)
from emu.contracts import OBS_SCHEMA_SENSOR


def exam(core=.96, hard=.35, contact=.32, watts=7.0):
    out = {'screen_core_survival': core}
    for name in ('resilience100', 'robust70', 'growth140',
                 'growthcombo140', 'contact_slope', 'contact_harder'):
        survival = (hard if name in ('resilience100','robust70') else
                    contact if 'contact' in name else .30)
        out['screen_' + name + '_survival'] = survival
        for metric, value in (
            ('vx_mae', .09), ('vy_mae', .12), ('yaw_mae', .42),
            ('height_shortfall_m', .005), ('pitch_mae', .12),
            ('step_evidence', .55), ('support', .92)):
            out['screen_' + name + '_' + metric] = value
    for name in ('robust', 'contact_slope'):
        for label, value in (
            ('shaft_abs_power_w', watts),
            ('over_rated_frac', .40)):
            out['power_' + name + '_' + label] = value
    return out


def test_contender_scoring_guards_core_hard_and_real_slope():
    original = score(exam())
    assert original['eligible']
    assert score(exam(watts=5.))['score'] > original['score']
    assert not score(exam(core=.50))['eligible']
    assert not score(exam(hard=.02))['eligible']
    assert not score(exam(contact=.01))['eligible']
    assert not score(exam(core=.82), ancestor=original)['eligible']
    with pytest.raises(ValueError, match='missing/nonfinite'):
        score({'screen_core_survival': 1.0})


def test_campaign_pins_232_actor_and_independent_arms(tmp_path):
    assert len(ARMS) == 4
    config = phase_config(tmp_path / 'round1' / 'grace', target=80_000_000,
                          seed=733, envs=2048, device='cuda', arm='grace')
    assert config.observation_schema == OBS_SCHEMA_SENSOR
    assert config.pitch_command_max_deg == 15.
    assert config.motor_guidance_weight == pytest.approx(.035)
    assert config.domain_randomization is False
    inclined = phase_config(tmp_path / 'round1' / 'contact', target=80_000_000,
                            seed=733, envs=2048, device='cuda', arm='contact')
    assert inclined.domain_terrain_mode == 'contact'
    assert inclined.domain_terrain_radius == pytest.approx(.40)
    late = phase_config(tmp_path / 'polish' / '800m', target=800_000_000,
                        seed=733, envs=2048, device='cuda',
                        arm='grace', polish_index=5)
    assert late.domain_terrain_radius == pytest.approx(.9)
    assert late.domain_hardware_radius == pytest.approx(.25)
    assert late.motor_guidance_weight == pytest.approx(.035)
    assert len(POLISH_TARGETS) == 6


def test_soft_budget_preserves_final_choice():
    ledger = {'budget_hours': 12., 'gpu_seconds': 0.,
              'observed_rates': []}
    assert budget_allows(ledger, steps=50_000_000)
    ledger['gpu_seconds'] = 11.8 * 3600
    assert not budget_allows(ledger, steps=100_000_000)
    assert ledger['gpu_seconds'] == pytest.approx(11.8 * 3600)


def test_reports_are_shareable_and_never_include_checkpoint_weights(tmp_path):
    foundation = {'path': '/outside/foundation/checkpoints/final.pkl',
                  'sha256': '1' * 64, 'step': 20_054_016}
    ledger = {'source_commit': 'a' * 40, 'sensor_sha256': 'b' * 64,
              'budget_hours': 12., 'gpu_seconds': 1234.,
              'champion': foundation, 'foundation': foundation,
              'schema': 1, 'finished': True}
    (tmp_path / 'campaign.json').write_text(json.dumps(ledger))
    report(tmp_path, ledger)
    assert (tmp_path / 'results.md').is_file()
    assert (tmp_path / 'results_data.json').is_file()
    assert (tmp_path / 'champion.json').is_file()
    import zipfile
    with zipfile.ZipFile(tmp_path / 'results_bundle.zip') as bundle:
        names = bundle.namelist()
    assert 'results.md' in names and 'champion.json' in names
    assert not any(path.endswith('.pkl') for path in names)


def test_cli_pinnacle_help_is_discoverable(capsys):
    from emu.cli import main as emu_main
    with pytest.raises(SystemExit) as exc:
        emu_main(['campaign5016', '--help'])
    assert exc.value.code == 0
    assert '--run' in capsys.readouterr().out


def test_planning_does_not_mutate_campaign_root(tmp_path, monkeypatch, capsys):
    import emu.campaign5016 as pinnacle
    monkeypatch.setattr(pinnacle, 'base_policy', lambda path: {
        'step': 20_054_016, 'path': '/home/example/final.pkl',
        'sha256': 'a' * 64})
    run_root = tmp_path / 'campaign'
    assert main(['--root', str(run_root), '--plan']) == 0
    assert '1.08B' in capsys.readouterr().out
    assert not run_root.exists()
