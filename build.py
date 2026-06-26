# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MOD_LIST_RANGE = {
    "tb_graph_ascend",
    "trend_analyzer",
    "atb_probe",
    "aclgraph_dump",
    "nan_check",
    "xor_checksum",
}

CPP_MODS = {"atb_probe", "aclgraph_dump", "nan_check", "xor_checksum"}

TORCH_NPU_MODS = {"aclgraph_dump", "nan_check", "xor_checksum"}
TORCH_NPU_URL_PATH = "torch_npu_url.json"
TORCH_NPU_DEPS = {"pyyaml", "numpy", "importlib_metadata"}

FRONTEND_MOD_MAP = {
    "tb_graph_ascend": "hierarchy_plugin",
    "trend_analyzer": "monvis_plugin",
}

PACKAGE_NAME_MAP = {
    "tb_graph_ascend": "hierarchy_plugin",
    "trend_analyzer": "trend_analyzer",
}


class BuildManager:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self._parse_args()
        self._parse_extra()
        self._original_pyproject_content = None

    @property
    def _pip_cmd(self):
        return ["uv", "pip"] if sys.prefix != sys.base_prefix else [sys.executable, "-m", "pip"]

    def _parse_args(self):
        ap = argparse.ArgumentParser(description='Build MindStudio-Probe and optionally run tests.')
        ap.add_argument(
            'command',
            nargs='*',
            default=[],
            help='Build action: omit for full build, "local" to skip dependency download, "test" to run unit tests',
        )
        ap.add_argument(
            '-v',
            '--version',
            type=str,
            default=None,
            help='Build version (default: read from pyproject.toml)',
        )
        ap.add_argument(
            '-e',
            '--extra',
            metavar='KEY=VALUE',
            action='append',
            default=[],
            help=(
                'Extra build options in KEY=VALUE format, can be specified multiple times. '
                'Supported keys: include-mod, no-check'
            ),
        )
        self.args = ap.parse_args()

        valid_commands = {'local', 'test'}
        for cmd in self.args.command:
            if cmd not in valid_commands:
                ap.error(f"Unknown command: {cmd}. Valid commands are: local, test")

    def _parse_extra(self):
        self.extra = {}
        for opt in self.args.extra:
            key, sep, val = opt.partition('=')
            if not key or sep != '=':
                raise ValueError(f"Invalid --extra format: {opt}. Expected KEY=VALUE")
            self.extra[key] = val

        self.mod_list = []
        if 'include-mod' in self.extra:
            mods = self.extra['include-mod'].split(',')
            self.mod_list = [m for m in mods if m in MOD_LIST_RANGE]
            invalid_mods = [m for m in mods if m not in MOD_LIST_RANGE]
            if invalid_mods:
                logging.warning("Unknown modules ignored: %s", invalid_mods)

        self.no_check = self.extra.get('no-check', 'false').lower() == 'true'
        self.has_cpp = any(mod in self.mod_list for mod in CPP_MODS)

    def _execute_command(self, cmd, timeout_seconds=3600, cwd=None, env=None):
        logging.info("Running: %s", " ".join(str(c) for c in cmd))
        subprocess.run(cmd, timeout=timeout_seconds, check=True, cwd=cwd, env=env)

    def _get_default_version(self):
        pyproject_path = self.project_root / "pyproject.toml"
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'^version\s*=\s*"([^"]*)"', line.strip())
                if match:
                    return match.group(1)
        return "26.0.0"

    def _need_torch_npu(self):
        return any(mod in self.mod_list for mod in TORCH_NPU_MODS)

    def _prepare_dependencies(self):
        if self._need_torch_npu():
            self._install_torch_npu()
        for mod, plugin_name in FRONTEND_MOD_MAP.items():
            if mod in self.mod_list:
                self._install_frontend_deps(plugin_name)

    def _get_py_tag(self):
        return f"cp{sys.version_info.major}{sys.version_info.minor}-cp{sys.version_info.major}{sys.version_info.minor}"

    def _load_wheels_config(self):
        config_path = self.project_root / TORCH_NPU_URL_PATH
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _verify_sha256(self, file_path, expected):
        if not expected:
            logging.warning("sha256 not configured for %s, skip verification", file_path)
            return
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        actual = h.hexdigest()
        if actual.lower() != expected.lower():
            raise RuntimeError(f"sha256 mismatch for {file_path}: expected {expected}, got {actual}")
        logging.info("sha256 verified for %s", file_path)

    def _install_torch_npu(self):
        result = subprocess.run(
            self._pip_cmd + ["show", "torch_npu"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            logging.info("torch_npu already installed, skip download")
            return

        config = self._load_wheels_config()
        arch = platform.machine()
        py_tag = self._get_py_tag()
        sha_key = f"{py_tag}-{arch}"

        wheels = []
        for name in ("torch", "torch_npu"):
            pkg = config[name]
            url = pkg["url_template"].format(py_tag=py_tag, arch=arch)
            sha256 = pkg.get("sha256", {}).get(sha_key, "")
            wheels.append((name, url, sha256))

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_files = []
            for name, url, sha256 in wheels:
                filename = unquote(urlparse(url).path.split("/")[-1])
                dst = os.path.join(tmp_dir, filename)
                logging.info("Downloading %s from %s", name, url)
                self._execute_command(["wget", "-q", "-O", dst, url])
                self._verify_sha256(dst, sha256)
                local_files.append(dst)
            self._execute_command(self._pip_cmd + ["install"] + local_files)

        self._install_torch_npu_deps()

    def _install_torch_npu_deps(self):
        missing = []
        for dep in TORCH_NPU_DEPS:
            result = subprocess.run(
                self._pip_cmd + ["show", dep],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                missing.append(dep)
        if missing:
            self._execute_command(
                self._pip_cmd + ["install"] + missing,
            )

    def _install_frontend_deps(self, plugin_name):
        fe_path = self.project_root / "plugins" / "tb_graph_ascend" / plugin_name / "front"
        if not fe_path.exists():
            raise RuntimeError(f"Frontend path '{fe_path}' does not exist")

        original_cwd = os.getcwd()
        try:
            os.chdir(fe_path)
            if not os.path.exists("package.json"):
                raise RuntimeError(f"package.json not found in {fe_path}")
            self._execute_command(["npm", "ci"])
        finally:
            os.chdir(original_cwd)

    def _build_frontend(self, plugin_name):
        fe_path = self.project_root / "plugins" / "tb_graph_ascend" / plugin_name / "front"
        if not fe_path.exists():
            raise RuntimeError(f"Frontend path '{fe_path}' does not exist")

        original_cwd = os.getcwd()
        try:
            os.chdir(fe_path)
            self._execute_command(["npm", "run", "build"])
        finally:
            os.chdir(original_cwd)

    def _build_cpp_modules(self):
        if not self.has_cpp:
            return

        release_dir = self.project_root / "output" / "release"
        if release_dir.exists():
            logging.info("Removing previous build output: %s", release_dir)
            shutil.rmtree(release_dir)

        arch = platform.machine()
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        build_cmd = [
            "bash",
            str(self.project_root / "build.sh"),
            "-j",
            "16",
            "-a",
            arch,
            "-v",
            py_version,
            "-m",
            str(self.mod_list).replace(' ', ''),
        ]

        if 'local' in self.args.command:
            build_cmd.append("--local")

        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        if self.no_check:
            env["INSTALL_WITHOUT_CHECK"] = "1"

        self._execute_command(build_cmd, cwd=self.project_root, env=env)

    def _prepare_package_artifacts(self):
        src_scripts = self.project_root / "scripts"
        dst_scripts = self.project_root / "python" / "msprobe" / "scripts"
        if src_scripts.exists():
            if dst_scripts.exists():
                shutil.rmtree(dst_scripts)
            shutil.copytree(src_scripts, dst_scripts)

        vendor_dir = self.project_root / "python" / "msprobe" / "vendors"
        if vendor_dir.exists():
            for root, dirs, files in os.walk(vendor_dir):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o750)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o750)
            os.chmod(vendor_dir, 0o750)
            shutil.rmtree(vendor_dir)

        for mod, plugin_name in FRONTEND_MOD_MAP.items():
            if mod in self.mod_list:
                src = self.project_root / "plugins" / "tb_graph_ascend" / plugin_name
                dst = self.project_root / "python" / PACKAGE_NAME_MAP[mod]
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(
                    src,
                    dst,
                    ignore=shutil.ignore_patterns('front', 'node_modules', '__pycache__'),
                )

    def _modify_pyproject(self):
        pyproject_path = self.project_root / "pyproject.toml"
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            self._original_pyproject_content = f.read()

        content = self._original_pyproject_content

        version = self.args.version or self._get_default_version()
        if self.args.version:
            content = re.sub(
                r'^version\s*=\s*"[^"]*"',
                f'version = "{version}"',
                content,
                flags=re.MULTILINE,
            )

        content = re.sub(
            r'\n\[project\.entry-points\.[^\]]*\].*?(?=\n\[|\Z)',
            '',
            content,
            flags=re.DOTALL,
        )

        entry_lines = []
        if "tb_graph_ascend" in self.mod_list:
            entry_lines.append('graph_ascend = "hierarchy_plugin.server.plugin:GraphsPlugin"')
        if "trend_analyzer" in self.mod_list:
            entry_lines.append('TrendVis = "trend_analyzer.server.app:TrendVis"')

        if entry_lines:
            content = content.rstrip('\n') + "\n\n[project.entry-points.tensorboard_plugins]\n"
            for line in entry_lines:
                content += f"{line}\n"

        plugin_packages = []
        if "tb_graph_ascend" in self.mod_list:
            plugin_packages.append('"python/hierarchy_plugin"')
        if "trend_analyzer" in self.mod_list:
            plugin_packages.append('"python/trend_analyzer"')

        if plugin_packages:
            packages_line = f'packages = ["python/msprobe", {", ".join(plugin_packages)}]'
        else:
            packages_line = 'packages = ["python/msprobe"]'

        content = re.sub(
            r'^packages\s*=\s*\[.*?\]',
            packages_line,
            content,
            flags=re.MULTILINE,
        )

        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _restore_pyproject(self):
        if self._original_pyproject_content is not None:
            pyproject_path = self.project_root / "pyproject.toml"
            with open(pyproject_path, 'w', encoding='utf-8') as f:
                f.write(self._original_pyproject_content)
            self._original_pyproject_content = None

    def _build_wheel(self):
        self._modify_pyproject()
        if self.has_cpp:
            os.environ["MSPROBE_CPP_BUILD"] = "1"
        try:
            artifacts_dir = self.project_root / "artifacts"
            self._execute_command(
                ["uv", "build", "--wheel", "--out-dir", str(artifacts_dir)],
                cwd=self.project_root,
            )
            logging.info("Build artifacts saved to: %s", artifacts_dir)
        finally:
            os.environ.pop("MSPROBE_CPP_BUILD", None)
            self._restore_pyproject()

    def _run_tests(self):
        test_dir = self.project_root / "test" / "msprobe_test"
        self._execute_command(["bash", "run_test.sh"], cwd=test_dir)

    def _build_all(self):
        for mod, plugin_name in FRONTEND_MOD_MAP.items():
            if mod in self.mod_list:
                self._build_frontend(plugin_name)

        self._build_cpp_modules()
        self._prepare_package_artifacts()
        self._build_wheel()

    def run(self):
        os.chdir(self.project_root)

        logging.info("--version: %s", self.args.version or self._get_default_version())
        for opt in self.args.extra:
            key, _, val = opt.partition('=')
            logging.info("--extra: %s = %s", key, val)

        if 'local' not in self.args.command:
            self._prepare_dependencies()

        if 'test' in self.args.command:
            self._run_tests()
        else:
            self._build_all()


if __name__ == "__main__":
    try:
        BuildManager().run()
    except Exception:
        logging.error("Unexpected error: %s", traceback.format_exc())
        sys.exit(1)
