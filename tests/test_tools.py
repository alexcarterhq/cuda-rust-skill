# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


doctor = load_module("doctor", ROOT / "cuda-rust/scripts/doctor.py")
validator = load_module("validator", ROOT / "tools/validate.py")


class DoctorTests(unittest.TestCase):
    def test_missing_binary_is_reported(self):
        with patch.object(doctor.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(doctor.probe(["absent-tool"])["status"], "missing")

    def test_timeout_is_bounded_and_reported(self):
        with patch.object(doctor.subprocess, "run", side_effect=subprocess.TimeoutExpired(["tool"], 0.1)):
            self.assertEqual(doctor.probe(["tool"], 0.1)["status"], "timeout")

    def test_probe_disables_installation_and_uses_no_shell(self):
        result = subprocess.CompletedProcess(["rustc", "-Vv"], 0, "rustc test\n", "")
        with patch.object(doctor.subprocess, "run", return_value=result) as run:
            self.assertEqual(doctor.probe(["rustc", "-Vv"])["status"], "available")
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["env"]["RUSTUP_AUTO_INSTALL"], "0")
            self.assertFalse(kwargs.get("shell", False))
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)

    def test_nonzero_probe_is_error_and_output_is_bounded(self):
        result = subprocess.CompletedProcess(["tool"], 1, "x" * 9000, "failure")
        with patch.object(doctor.subprocess, "run", return_value=result):
            check = doctor.probe(["tool"])
            self.assertEqual(check["status"], "error")
            self.assertEqual(len(check["output"]), 8192)
            self.assertTrue(check["truncated"])

    def test_inventory_has_no_secrets_or_implicit_builds(self):
        calls = []

        def fake_probe(argv, timeout):
            calls.append(argv)
            return {"status": "missing", "command": argv}

        with patch.dict(os.environ, {"TOP_SECRET_TOKEN": "do-not-print", "CUDA_RUST_DEBUG": "line"}), \
             patch.object(doctor, "probe", side_effect=fake_probe):
            report = doctor.inventory("tile")
        self.assertNotIn("do-not-print", json.dumps(report))
        self.assertEqual(report["environment"]["CUDA_RUST_DEBUG"], "line")
        self.assertIn("tileiras", report["required_probe_gaps"])
        self.assertEqual(report["gpu_validation"], "not run")
        self.assertFalse(any("oxide" in argv or "install" in argv or "build" in argv for argv in calls))

    def test_unavailable_checkout_is_unknown(self):
        with patch.object(doctor, "probe", return_value={"status": "missing"}):
            report = doctor.inventory(upstream="/missing/checkout")
        self.assertIsNone(report["upstream"]["matches_snapshot"])


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "package"
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns(".git", "__pycache__", "._*"))

    def test_clean_package(self):
        self.assertEqual(validator.validate(self.root), [])

    def test_broken_reference_is_rejected(self):
        entry = self.root / "cuda-rust/SKILL.md"
        entry.write_text(entry.read_text() + "\n[missing](references/missing.md)\n")
        self.assertTrue(any("Broken local link" in error for error in validator.validate(self.root)))

    def test_mixed_cuda_revisions_are_rejected(self):
        manifest = self.root / "cuda-rust/assets/simt-vector-add/Cargo.toml"
        manifest.write_text(manifest.read_text().replace("a4a939a9b5f01f3bf712e92386922e1bc7c434a9", "0" * 40, 1))
        self.assertTrue(any("same pinned upstream revision" in error for error in validator.validate(self.root)))

    def test_missing_starter_dependencies_are_rejected(self):
        for mode in ("simt", "tile"):
            with self.subTest(mode=mode):
                manifest = self.root / f"cuda-rust/assets/{mode}-vector-add/Cargo.toml"
                original = manifest.read_text()
                manifest.write_text(original.split("[dependencies]")[0] + "[dependencies]\n")
                self.assertTrue(any("Missing starter dependencies" in error for error in validator.validate(self.root)))
                manifest.write_text(original)

    def test_missing_simt_compiler_components_are_rejected(self):
        toolchain = self.root / "cuda-rust/assets/simt-vector-add/rust-toolchain.toml"
        toolchain.write_text(toolchain.read_text().replace('"rustc-dev", ', ''))
        self.assertTrue(any("compiler components" in error for error in validator.validate(self.root)))

    def test_skill_link_must_survive_subdirectory_installation(self):
        entry = self.root / "cuda-rust/SKILL.md"
        entry.write_text(entry.read_text() + "\n[readme](../README.md)\n")
        self.assertTrue(any("Link escapes installed skill" in error for error in validator.validate(self.root)))

    def test_skill_link_cannot_escape_through_a_symlink(self):
        (self.root / "cuda-rust/references/outside.md").symlink_to(self.root / "README.md")
        entry = self.root / "cuda-rust/SKILL.md"
        entry.write_text(entry.read_text() + "\n[outside](references/outside.md)\n")
        self.assertTrue(any("Link escapes installed skill" in error for error in validator.validate(self.root)))

    def test_external_source_path_is_rejected(self):
        filename = self.root / "cuda-rust/references/upstream.json"
        data = json.loads(filename.read_text())
        data["sources"][0]["path"] = "../outside"
        filename.write_text(json.dumps(data))
        self.assertTrue(any("Unsafe upstream source path" in error for error in validator.validate(self.root)))

    def test_dirty_source_hash_is_rejected_even_with_matching_commit(self):
        checkout = Path(self.temp.name) / "upstream"
        checkout.mkdir()
        filename = self.root / "cuda-rust/references/upstream.json"
        data = json.loads(filename.read_text())
        source = data["sources"][0]
        data["sources"] = [source]
        filename.write_text(json.dumps(data))
        (checkout / source["path"]).write_text("modified source")
        revision = subprocess.CompletedProcess(["git"], 0, data["commit"] + "\n", "")
        with patch.object(validator.subprocess, "run", return_value=revision):
            errors = validator.validate(self.root, checkout)
        self.assertTrue(any("Upstream source drift" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
