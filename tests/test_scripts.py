"""Regression checks for benchmark orchestration and profiler reports; no GPU needed."""
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ProfileReportTests(unittest.TestCase):
    def test_fused_kernel_has_own_bytes_and_speedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                "resize": [("resize", "hip", 120)],
                "crop_mirror_normalize": [("crop_mirror_normalize", "hip", 150)],
                "resize_cmn": [("resize_cmn", "hip_fused", 600)],
            }
            for pipeline, rows in cases.items():
                session = root / "b1" / pipeline
                session.mkdir(parents=True)
                with (session / "results.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["pipeline", "impl", "bytes_read", "bytes_written", "device"])
                    writer.writerows((p, impl, size, 0, "fixture") for p, impl, size in rows)
                kernels = {
                    "resize": [("resize_kernel", 2000)],
                    "crop_mirror_normalize": [("cmn_kernel", 1000)],
                    "resize_cmn": [("resize_kernel", 2000), ("cmn_kernel", 1000),
                                   ("fused_rcmn_kernel", 1500)],
                }[pipeline]
                with (session / "fixture_kernel_stats.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Name", "Calls", "AverageNs"])
                    writer.writerows((f"void hipaug::(anonymous namespace)::{k}()", 10, ns)
                                     for k, ns in kernels)
            md, report = root / "report.md", root / "report.csv"
            subprocess.run([sys.executable, str(ROOT / "scripts/profile_report.py"), str(root),
                            "--md", str(md), "--csv", str(report)], check=True,
                           capture_output=True, text=True)
            with report.open(newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["pipeline"] == "resize_cmn"]
            self.assertEqual(len(rows), 3)
            by_kernel = {r["kernel"]: r for r in rows}
            self.assertEqual(int(by_kernel["fused_rcmn_kernel"]["bytes"]), 600)
            self.assertEqual(int(by_kernel["cmn_kernel"]["bytes"]), 150)
            self.assertIn("| 1 | 3.0 | 1.5 | 2.00x |", md.read_text())


class BenchShellTests(unittest.TestCase):
    def run_bench(self, *args):
        # Prefer Git Bash on Windows: the system bash.exe may be a WSL launcher.
        git = shutil.which("git")
        bash = (Path(git).resolve().parents[1] / "bin/bash.exe") if git else Path("missing")
        bash = str(bash) if bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/bench.sh", root / "scripts/bench.sh")
            (root / "bin with spaces").mkdir()
            log = root / "calls.log"
            for path, label in [(root / "bin with spaces/hipaug", "hipaug"),
                                (root / "python-stub", "python")]:
                path.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "{label}" "$@" END >> "$CALL_LOG"\n',
                                encoding="utf-8", newline="\n")
                path.chmod(0o755)
            env = dict(os.environ, BUILD_DIR=(root / "bin with spaces").as_posix(),
                       PYTHON=(root / "python-stub").as_posix(), CALL_LOG=log.as_posix(),
                       OUT="environment.csv")
            subprocess.run([bash, (root / "scripts/bench.sh").as_posix(), *args],
                           env=env, check=True, capture_output=True, text=True)
            return [call.splitlines() for call in log.read_text().split("END\n") if call]

    def test_custom_options_and_output_reach_baselines(self):
        options = ["--batches", "2,3", "--size", "32x24", "--src-min", "40", "--src-max", "60",
                   "--peak-gbps", "100", "--pipelines", "resize_cmn", "--warmup", "0",
                   "--cpu-runs", "3", "--runs", "7", "--no-gpu", "--out", "custom results.csv"]
        calls = self.run_bench(*options)
        self.assertEqual(calls[0], ["hipaug", "bench", "--out", "custom results.csv", *options])
        self.assertIn(["python", "bench/baselines.py", "--append", "custom results.csv",
                       *options[:14], "--runs", "3"], calls)
        self.assertIn(["python", "docs/make_charts.py", "custom results.csv"], calls)

    def test_no_cpu_skips_baselines(self):
        calls = self.run_bench("--no-cpu")
        self.assertFalse(any("bench/baselines.py" in c for c in calls))

    def test_default_output_is_shared(self):
        calls = self.run_bench()
        self.assertIn(["python", "bench/baselines.py", "--append", "environment.csv"], calls)


class BaselineTests(unittest.TestCase):
    def test_pipeline_selection_filters_rows(self):
        # Exercise argument parsing, selection and CSV output without requiring media libraries.
        numpy = types.SimpleNamespace(array=lambda values, **kw: values, float32="float32")
        cv2 = types.SimpleNamespace(__version__="fixture", getNumThreads=lambda: 1)
        with patch.dict(sys.modules, {"numpy": numpy, "cv2": cv2, "torchvision": None}):
            spec = importlib.util.spec_from_file_location("baselines", ROOT / "bench/baselines.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results.csv"
                argv = ["baselines", "--append", str(output), "--batches", "2",
                        "--size", "32x24", "--src-min", "40", "--src-max", "60",
                        "--runs", "3", "--pipelines", "resize_cmn"]
                with patch.object(sys, "argv", argv), \
                     patch.object(module, "make_images", return_value=(
                         [types.SimpleNamespace(shape=(40, 60, 3))] * 2, [False, True])), \
                     patch.object(module, "opencv_pipelines", return_value={
                         "resize": lambda: None, "resize_cmn": lambda: None}), \
                     patch.object(module, "time_fn", return_value=[1, 2, 3]):
                    module.main()
                with output.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["pipeline"], "resize_cmn")
                self.assertEqual(rows[0]["out_w"], "32")
                self.assertEqual(rows[0]["out_h"], "24")
                self.assertEqual(rows[0]["batch"], "2")
                self.assertEqual(rows[0]["runs"], "3")
                with patch.object(sys, "argv", argv[:-1] + ["gaussian_blur_3x3"]), \
                     patch.object(module, "make_images") as make_images:
                    module.main()
                    make_images.assert_not_called()


if __name__ == "__main__":
    unittest.main()
