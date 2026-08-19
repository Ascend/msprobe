import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


def _load_hatch_build_module():
    module_names = (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.builders.hooks.plugin",
        "hatchling.builders.hooks.plugin.interface",
    )
    fake_modules = {name: types.ModuleType(name) for name in module_names}
    fake_modules[module_names[-1]].BuildHookInterface = object
    module_path = Path(__file__).resolve().parents[4] / "hatch_build.py"
    spec = importlib.util.spec_from_file_location("msprobe_hatch_build_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


hatch_build = _load_hatch_build_module()


class TestHatchBuildInfo(TestCase):
    @patch.dict("os.environ", {"MSPROBE_BUILD_COMMIT": "41E14EC123456789"}, clear=True)
    def test_commit_id_uses_build_environment(self):
        self.assertEqual(hatch_build._get_commit_id(Path(".")), "41e14ec")

    @patch.dict("os.environ", {"SOURCE_DATE_EPOCH": "0"}, clear=True)
    def test_build_date_honours_source_date_epoch(self):
        self.assertEqual(hatch_build._get_build_date(), "1970-01-01T00:00:00Z")

    def test_render_build_info_contains_commit(self):
        content = hatch_build._render_build_info("26.1.0", "41e14ec", "2026-07-15T10:32:11Z")

        self.assertIn("VERSION = '26.1.0'", content)
        self.assertIn("COMMIT_ID = '41e14ec'", content)

    @patch.dict(
        "os.environ",
        {"MSPROBE_BUILD_COMMIT": "41e14ec123456789", "SOURCE_DATE_EPOCH": "0"},
        clear=True,
    )
    def test_hook_embeds_and_restores_build_info(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_info_path = Path(temp_dir) / hatch_build.BUILD_INFO_PATH
            build_info_path.parent.mkdir(parents=True)
            build_info_path.write_text("original\n", encoding="utf-8")
            hook = hatch_build.CppWheelTagHook()
            hook.root = temp_dir
            hook.metadata = types.SimpleNamespace(version="26.1.0")

            hook.initialize("standard", {})
            generated = build_info_path.read_text(encoding="utf-8")
            self.assertIn("VERSION = '26.1.0'", generated)
            self.assertNotIn("VERSION = 'standard'", generated)
            self.assertIn("COMMIT_ID = '41e14ec'", generated)
            self.assertIn("BUILD_DATE = '1970-01-01T00:00:00Z'", generated)

            hook.finalize("standard", {}, "artifact.whl")
            self.assertEqual(build_info_path.read_text(encoding="utf-8"), "original\n")
