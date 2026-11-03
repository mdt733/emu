"""v5043: CUDA smoke report must describe real enabled prism collisions."""
from scripts.smoke_gpu_carpet import physics_report_fields


def test_gpu_smoke_report_prisms_are_not_labeled_boxes():
    info = physics_report_fields(prisms=True)
    assert info == {
        "physics": "eighteen-dynamic-convex-triangular-prisms",
        "active_collision_geoms": 18,
        "terrain_carpet_prisms": True,
    }


def test_gpu_smoke_legacy_box_output_unchanged():
    info = physics_report_fields(prisms=False)
    assert info == {
        "physics": "nine-solid-tilted-boxes",
        "active_collision_geoms": 9,
        "terrain_carpet_prisms": False,
    }
