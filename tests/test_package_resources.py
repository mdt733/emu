from emu.paths import PACKAGE_ROOT, ROBOT_XML, VERSION, VERSION_FILE


def test_package_resources_are_owned_by_emu_package():
    assert VERSION
    assert VERSION_FILE.parent == PACKAGE_ROOT
    assert ROBOT_XML.parent == PACKAGE_ROOT
    assert VERSION_FILE.is_file()
    assert ROBOT_XML.is_file()
    assert '<mujoco' in ROBOT_XML.read_text(encoding='utf-8')
