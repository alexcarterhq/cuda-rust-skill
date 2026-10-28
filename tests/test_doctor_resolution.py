# SPDX-License-Identifier: Apache-2.0
"""Filesystem and CLI regressions; no CUDA installation or GPU is needed."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_tools import ROOT, doctor


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.host = patch.object(doctor.platform, "system", return_value="Linux")
        self.arch = patch.object(doctor.platform, "machine", return_value="x86_64")
        self.host.start()
        self.arch.start()
        self.addCleanup(self.host.stop)
        self.addCleanup(self.arch.stop)

    def toolkit(self, name, version=13030, assembler=True, include="include"):
        root = self.root / name
        (root / include).mkdir(parents=True)
        (root / include / "cuda.h").write_text(f"#define CUDA_VERSION {version}\n")
        (root / "bin").mkdir(exist_ok=True)
        if assembler:
            (root / "bin/tileiras").touch()
        return root

    def test_discovery_skips_incomplete_and_old_toolkits(self):
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        old = self.toolkit("old", 12090)
        usable = self.toolkit("usable")
        with patch.object(doctor, "STANDARD_TOOLKITS", (incomplete, old, usable)):
            self.assertEqual(doctor.select_toolkit({}), usable)

    def test_explicit_invalid_or_empty_toolkit_does_not_fall_back(self):
        usable = self.toolkit("usable")
        for value in ("", str(self.root / "absent"), "~/cuda"):
            with self.subTest(value=value), patch.object(doctor, "STANDARD_TOOLKITS", (usable,)):
                with self.assertRaisesRegex(ValueError, "CUDA_TOOLKIT_PATH"):
                    doctor.select_toolkit({"CUDA_TOOLKIT_PATH": value, "CUDA_HOME": str(usable)})

    def test_toolkit_environment_precedence(self):
        first = self.toolkit("first")
        second = self.toolkit("second")
        self.assertEqual(doctor.select_toolkit({"CUDA_HOME": str(second)}), second)
        self.assertEqual(doctor.select_toolkit({"CUDA_TOOLKIT_PATH": str(first), "CUDA_HOME": str(second)}), first)

    def test_headers_are_selected_for_the_native_architecture(self):
        wrong_arch = self.toolkit("wrong", include="targets/sbsa-linux/include")
        native = self.toolkit("native", include="targets/x86_64-linux/include")
        with patch.object(doctor, "STANDARD_TOOLKITS", (wrong_arch, native)):
            self.assertEqual(doctor.select_toolkit({}), native)

    def test_arm_linux_supports_server_and_tegra_header_layouts(self):
        tegra = self.toolkit("tegra", include="targets/aarch64-linux/include")
        server = self.toolkit("server", include="targets/sbsa-linux/include")
        with patch.object(doctor.platform, "machine", return_value="aarch64"):
            for root in (tegra, server):
                with self.subTest(root=root):
                    self.assertEqual(doctor.select_toolkit({"CUDA_HOME": str(root)}), root)

    def test_malformed_or_old_first_header_does_not_fall_through(self):
        root = self.toolkit("cuda", include="targets/x86_64-linux/include")
        (root / "include").mkdir()
        for contents in ("no version", "#define CUDA_VERSION 12090\n", "#define CUDA_VERSION 9999999999\n"):
            with self.subTest(contents=contents):
                (root / "include/cuda.h").write_text(contents)
                with self.assertRaises(ValueError):
                    doctor.select_toolkit({"CUDA_HOME": str(root)})

    def test_header_target_override_is_exclusive(self):
        root = self.toolkit("cuda")
        env = {"CUDA_HOME": str(root), "CUDA_TOOLKIT_TARGET_DIR": "sbsa-linux"}
        with self.assertRaises(ValueError):
            doctor.select_toolkit(env)
        (root / "targets/sbsa-linux/include").mkdir(parents=True)
        (root / "targets/sbsa-linux/include/cuda.h").write_text("#define CUDA_VERSION 13030\n")
        self.assertEqual(doctor.select_toolkit(env), root)

    def test_assembler_discovery_is_independent_of_build_toolkit(self):
        build = self.toolkit("build", 13000, assembler=False)
        too_old = self.toolkit("old", 13010)
        jit = self.toolkit("jit")
        with patch.object(doctor, "TILEIRAS_TOOLKITS", (too_old, jit)):
            self.assertEqual(doctor.select_tileiras({"CUDA_HOME": str(build)}), str(jit / "bin/tileiras"))

    def test_explicit_assembler_never_falls_back_or_expands_tilde(self):
        for value in (str(self.root / "absent"), "~/bin/tileiras"):
            with self.subTest(value=value):
                self.assertEqual(doctor.select_tileiras({"CUTILE_TILEIRAS_PATH": value}), value)

    def test_explicit_toolkit_assembler_precedes_defaults(self):
        explicit = self.toolkit("explicit", 13000)
        default = self.toolkit("default")
        with patch.object(doctor, "TILEIRAS_TOOLKITS", (default,)):
            self.assertEqual(doctor.select_tileiras({"CUDA_HOME": str(explicit)}), str(explicit / "bin/tileiras"))

    def test_missing_first_toolkit_assembler_skips_second_variable(self):
        first = self.toolkit("first", assembler=False)
        second = self.toolkit("second")
        default = self.toolkit("default")
        with patch.object(doctor, "TILEIRAS_TOOLKITS", (default,)):
            env = {"CUDA_TOOLKIT_PATH": str(first), "CUDA_HOME": str(second)}
            self.assertEqual(doctor.select_tileiras(env), str(default / "bin/tileiras"))
            env["CUDA_TOOLKIT_PATH"] = ""
            self.assertEqual(doctor.select_tileiras(env), str(second / "bin/tileiras"))

    def test_assembler_falls_back_to_path(self):
        with patch.object(doctor, "TILEIRAS_TOOLKITS", ()), \
             patch.object(doctor.shutil, "which", return_value="/path/tileiras"):
            self.assertEqual(doctor.select_tileiras({}), "/path/tileiras")

    def test_invalid_toolkit_is_a_required_gap_even_with_all_tools_available(self):
        available = {"status": "available", "output": "test"}
        with patch.dict(os.environ, {"CUDA_TOOLKIT_PATH": ""}, clear=True), \
             patch.object(doctor, "probe", return_value=available):
            report = doctor.inventory("simt")
        self.assertIn("toolkit", report["required_probe_gaps"])
        self.assertEqual(report["checks"]["toolkit"]["status"], "error")
        self.assertIn("CUDA_TOOLKIT_PATH", report["checks"]["toolkit"]["output"])

    def test_inventory_probes_the_resolved_binaries(self):
        build = self.toolkit("build", 13000, assembler=False)
        jit = self.toolkit("jit")
        with patch.dict(os.environ, {"CUDA_HOME": str(build)}, clear=True), \
             patch.object(doctor, "TILEIRAS_TOOLKITS", (jit,)), \
             patch.object(doctor, "probe", return_value={"status": "available"}) as probe:
            report = doctor.inventory("tile")
        commands = [call.args[0] for call in probe.call_args_list]
        self.assertIn([str(build / "bin/nvcc"), "--version"], commands)
        self.assertIn([str(jit / "bin/tileiras"), "--version"], commands)
        self.assertEqual(report["selected_toolkit"], str(build))
        self.assertEqual(report["required_probe_gaps"], [])

    def test_strict_status_distinguishes_host_mode_and_probe_failure(self):
        root = self.toolkit("cuda")
        for host, mode, failed, expected in (
            ("Linux", "simt", None, 0),
            ("Linux", "tile", None, 0),
            ("Linux", "tile", "tileiras", 2),
            ("Linux", "simt", "nvidia-smi", 2),
            ("Darwin", "simt", None, 2),
            ("Darwin", "inspect", "nvidia-smi", 0),
        ):
            def fake_probe(argv, timeout):
                return {"status": "missing" if Path(argv[0]).name == failed else "available"}

            with self.subTest(host=host, mode=mode, failed=failed), \
                 patch.dict(os.environ, {"CUDA_HOME": str(root)}, clear=True), \
                 patch.object(doctor.platform, "system", return_value=host), \
                 patch.object(doctor, "probe", side_effect=fake_probe), \
                 patch.object(sys, "argv", ["doctor.py", "--mode", mode, "--strict", "--json"]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(doctor.main(), expected)
                self.assertEqual(json.loads(output.getvalue())["mode"], mode)


class InstalledDoctorTests(unittest.TestCase):
    def test_installed_skill_runs_from_unrelated_directory_with_no_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            installed = root / "installed/cuda-rust"
            shutil.copytree(ROOT / "cuda-rust", installed, ignore=shutil.ignore_patterns("__pycache__", "._*"))
            cwd = root / "project"
            cwd.mkdir()
            env = {"PATH": str(root / "empty-path"), "CUDA_TOOLKIT_PATH": str(root / "missing")}
            command = [sys.executable, str(installed / "scripts/doctor.py"), "--json", "--strict"]
            for mode, status in (("inspect", 0), ("simt", 2), ("tile", 2)):
                with self.subTest(mode=mode):
                    result = subprocess.run(command + ["--mode", mode], cwd=cwd, env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, status, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["gpu_validation"], "not run")
                    self.assertEqual(report["checks"]["rustc"]["status"], "missing")

    def test_cli_rejects_unbounded_timeouts(self):
        for value in ("0", "-1", "31", "nan", "inf"):
            with self.subTest(value=value):
                result = subprocess.run([sys.executable, str(ROOT / "cuda-rust/scripts/doctor.py"), "--timeout", value], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--timeout must be", result.stderr)


if __name__ == "__main__":
    unittest.main()
