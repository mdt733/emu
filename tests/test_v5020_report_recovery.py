"""Null-parent foundation regression: report failure must never stop a valid campaign."""
import json

from emu import results


def test_shared_foundation_with_null_parent_writes_full_report(tmp_path):
    root=tmp_path/"lab"
    foundation=root/"foundation20"
    foundation.mkdir(parents=True)
    (root/"lab_context.json").write_text(
        json.dumps({"seed":733,"envs":4096,"git_commit":"pinned-origin"}))
    (foundation/"lab_spec.json").write_text(json.dumps({
        "stage":"foundation","arm":None,"target":20_000_000,"parent":None}))
    (foundation/"training_config.json").write_text(json.dumps({
        "steps":20_000_000,"domain_terrain_radius":.16,
        "pitch_command_max_deg":8.0,"capability_field_mode":"off"}))
    (foundation/"run_meta.json").write_text(json.dumps({
        "source_provenance":{"git_commit":"pinned-origin"}}))
    # A real run also has metrics, evaluation and a checkpoint; deliberately
    # omit those to ensure the offline report records gaps instead of crashing.
    evidence=results.collect(root)
    assert evidence["runs"]["foundation20"]["parent"]=={}
    outputs=results.write_report(root)
    assert {p.name for p in outputs}=={
        "results.md","results_data.json","results_bundle.zip"}
    report=(root/"results.md").read_text()
    assert "foundation20" in report
    assert "Parent SHA-256: `unavailable`" in report
    assert (root/"results_data.json").is_file()
