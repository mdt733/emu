"""The HTTP viewer, policy export and capability atlas are distinct modules."""
from emu.policy_export import (
    build_bundle, write_bundle, load_exported_actor_for_exam
)
from emu.viewer_server import (
    build_bundle as server_bundle,
    LivePolicyWatcher, read_terrain_training_snapshot, serve,
)
from emu.capability_viewer import field_main, payload


def test_export_owned_by_actor_module_not_http_server():
    assert server_bundle is build_bundle
    assert build_bundle.__module__ == "emu.policy_export"
    assert write_bundle.__module__ == "emu.policy_export"
    assert load_exported_actor_for_exam.__module__ == "emu.policy_export"
    assert LivePolicyWatcher.__module__ == "emu.viewer_server"
    assert read_terrain_training_snapshot.__module__ == "emu.viewer_server"
    assert serve.__module__ == "emu.viewer_server"


def test_capability_field_is_independent_webapp_not_exporter():
    assert field_main.__module__ == "emu.capability_viewer"
    assert payload.__module__ == "emu.capability_viewer"
