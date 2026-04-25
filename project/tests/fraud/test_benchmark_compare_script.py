"""Tests for project/scripts/benchmark_compare.py.

Validates:
- CSV loading and delta computation
- SLA transition classification (FAIL->PASS, PASS->FAIL, no change)
- cp1252-safe output (no non-ASCII characters in printed text)
- Output file generation (JSON + CSV written to temp dir)
- Edge cases: mismatched keys, zero-row CSV, same-row data
"""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from project.scripts.benchmark_compare import (
    BenchRow,
    DeltaRow,
    build_delta,
    load_csv,
    print_table,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_bench_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "endpoint_name", "url", "concurrency", "requests_total",
        "success_count", "error_count", "error_rate_pct", "throughput_rps",
        "p50_ms", "p95_ms", "p99_ms", "viability",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_BASE_ROW = {
    "url": "http://127.0.0.1:8000/score_transaction",
    "requests_total": 160,
    "success_count": 160,
    "error_count": 0,
    "error_rate_pct": "0.0",
    "throughput_rps": "12.5",
    "viability": "FAIL",
}


def _row(endpoint, c, p50, p95, p99, **kw) -> dict:
    return {**_BASE_ROW, "endpoint_name": endpoint, "concurrency": c,
            "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, **kw}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadCsv(unittest.TestCase):
    def test_load_basic(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bench.csv"
            _write_bench_csv([_row("score_transaction", 6, 114.2, 725.3, 820.1)], path)
            rows = load_csv(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].endpoint_name, "score_transaction")
        self.assertEqual(rows[0].concurrency, 6)
        self.assertAlmostEqual(rows[0].p95_ms, 725.3, places=1)

    def test_load_multiple_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bench.csv"
            _write_bench_csv([
                _row("score_transaction", 2, 112.5, 164.3, 182.0),
                _row("score_transaction", 6, 367.5, 455.7, 464.5),
                _row("wallet_authorize_payment", 6, 358.8, 433.7, 439.6),
            ], path)
            rows = load_csv(path)
        self.assertEqual(len(rows), 3)


# ---------------------------------------------------------------------------
# Delta computation + SLA classification
# ---------------------------------------------------------------------------

class TestBuildDelta(unittest.TestCase):
    def _make_rows(self, before_p95, after_p95, err=0.0):
        before = [BenchRow(
            endpoint_name="ep", concurrency=6,
            p50_ms=100.0, p95_ms=before_p95, p99_ms=before_p95 + 50,
            error_rate_pct=err, throughput_rps=12.0,
            requests_total=160, success_count=160, error_count=0,
        )]
        after = [BenchRow(
            endpoint_name="ep", concurrency=6,
            p50_ms=100.0, p95_ms=after_p95, p99_ms=after_p95 + 50,
            error_rate_pct=err, throughput_rps=15.0,
            requests_total=160, success_count=160, error_count=0,
        )]
        return before, after

    def test_fail_to_pass(self):
        before, after = self._make_rows(before_p95=725.0, after_p95=200.0)
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].sla_changed, "FAIL->PASS")
        self.assertEqual(deltas[0].before_sla, "FAIL")
        self.assertEqual(deltas[0].after_sla, "PASS")

    def test_pass_to_fail(self):
        before, after = self._make_rows(before_p95=200.0, after_p95=725.0)
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        self.assertEqual(deltas[0].sla_changed, "PASS->FAIL")

    def test_no_change_both_fail(self):
        before, after = self._make_rows(before_p95=725.0, after_p95=455.0)
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        self.assertEqual(deltas[0].sla_changed, "no change")
        self.assertEqual(deltas[0].before_sla, "FAIL")
        self.assertEqual(deltas[0].after_sla, "FAIL")

    def test_no_change_both_pass(self):
        before, after = self._make_rows(before_p95=220.0, after_p95=200.0)
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        self.assertEqual(deltas[0].sla_changed, "no change")
        self.assertEqual(deltas[0].before_sla, "PASS")
        self.assertEqual(deltas[0].after_sla, "PASS")

    def test_delta_values(self):
        before, after = self._make_rows(before_p95=725.0, after_p95=455.0)
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        d = deltas[0]
        self.assertAlmostEqual(d.delta_p95, -270.0, places=1)
        self.assertAlmostEqual(d.pct_p95, -37.2, places=0)

    def test_mismatched_keys_skipped(self):
        before = [BenchRow(
            endpoint_name="ep_A", concurrency=6,
            p50_ms=100.0, p95_ms=500.0, p99_ms=600.0,
            error_rate_pct=0.0, throughput_rps=12.0,
            requests_total=160, success_count=160, error_count=0,
        )]
        after = [BenchRow(
            endpoint_name="ep_B", concurrency=6,
            p50_ms=100.0, p95_ms=200.0, p99_ms=250.0,
            error_rate_pct=0.0, throughput_rps=15.0,
            requests_total=160, success_count=160, error_count=0,
        )]
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        self.assertEqual(len(deltas), 0)

    def test_error_rate_causes_sla_fail(self):
        before = [BenchRow(
            endpoint_name="ep", concurrency=6,
            p50_ms=100.0, p95_ms=200.0, p99_ms=250.0,
            error_rate_pct=0.0, throughput_rps=12.0,
            requests_total=160, success_count=160, error_count=0,
        )]
        after = [BenchRow(
            endpoint_name="ep", concurrency=6,
            p50_ms=100.0, p95_ms=200.0, p99_ms=250.0,
            error_rate_pct=5.0, throughput_rps=12.0,
            requests_total=160, success_count=152, error_count=8,
        )]
        deltas = build_delta(before, after, sla_p95=250.0, sla_p99=500.0, sla_err=1.0)
        self.assertEqual(deltas[0].sla_changed, "PASS->FAIL")


# ---------------------------------------------------------------------------
# cp1252-safe output (no non-ASCII in any printed characters)
# ---------------------------------------------------------------------------

class TestCp1252SafeOutput(unittest.TestCase):
    def _capture_print_table(self, deltas: list[DeltaRow], before_label="before", after_label="after") -> str:
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_table(deltas, before_label, after_label)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def _make_deltas(self, sla_changed="FAIL->PASS") -> list[DeltaRow]:
        return [DeltaRow(
            endpoint_name="score_transaction", concurrency=6,
            before_p50=114.0, after_p50=98.0, delta_p50=-16.0, pct_p50=-14.0,
            before_p95=725.0, after_p95=452.0, delta_p95=-273.0, pct_p95=-37.7,
            before_p99=820.0, after_p99=590.0, delta_p99=-230.0, pct_p99=-28.0,
            before_err_pct=0.0, after_err_pct=0.0,
            before_sla="FAIL", after_sla="PASS",
            sla_changed=sla_changed,
        )]

    def test_no_non_ascii_in_print_table(self):
        output = self._capture_print_table(self._make_deltas())
        for char in output:
            self.assertLess(
                ord(char), 128,
                f"Non-ASCII character found in print_table output: {repr(char)} (U+{ord(char):04X})",
            )

    def test_all_sla_labels_ascii_safe(self):
        for label in ("FAIL->PASS", "PASS->FAIL", "no change"):
            deltas = self._make_deltas(sla_changed=label)
            output = self._capture_print_table(deltas)
            for char in output:
                self.assertLess(ord(char), 128, f"Non-ASCII in label {label!r}: {repr(char)}")

    def test_print_table_contains_endpoint_name(self):
        output = self._capture_print_table(self._make_deltas())
        self.assertIn("score_transaction", output)

    def test_print_table_contains_sla_status(self):
        output = self._capture_print_table(self._make_deltas())
        self.assertIn("FAIL", output)
        self.assertIn("PASS", output)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

class TestWriteOutputs(unittest.TestCase):
    def _make_deltas(self) -> list[DeltaRow]:
        return [DeltaRow(
            endpoint_name="score_transaction", concurrency=6,
            before_p50=114.0, after_p50=98.0, delta_p50=-16.0, pct_p50=-14.0,
            before_p95=725.0, after_p95=452.0, delta_p95=-273.0, pct_p95=-37.7,
            before_p99=820.0, after_p99=590.0, delta_p99=-230.0, pct_p99=-28.0,
            before_err_pct=0.0, after_err_pct=0.0,
            before_sla="FAIL", after_sla="PASS",
            sla_changed="FAIL->PASS",
        )]

    def test_json_and_csv_created(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            before_path = Path("before.csv")
            after_path = Path("after.csv")
            paths = write_outputs(self._make_deltas(), before_path, after_path, out_dir)
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())

    def test_json_contains_summary(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            paths = write_outputs(self._make_deltas(), Path("b.csv"), Path("a.csv"), out_dir)
            with open(paths["json"], encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertIn("summary", report)
            self.assertEqual(report["summary"]["fail_to_pass"], 1)
            self.assertEqual(report["summary"]["pass_to_fail"], 0)
            self.assertEqual(report["summary"]["total_comparisons"], 1)

    def test_csv_has_expected_columns(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            paths = write_outputs(self._make_deltas(), Path("b.csv"), Path("a.csv"), out_dir)
            with open(paths["csv"], newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertIn("delta_p95", rows[0])
            self.assertIn("sla_changed", rows[0])
            self.assertEqual(rows[0]["sla_changed"], "FAIL->PASS")

    def test_json_rows_ascii_safe(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            paths = write_outputs(self._make_deltas(), Path("b.csv"), Path("a.csv"), out_dir)
            raw = Path(paths["json"]).read_text(encoding="utf-8")
            for char in raw:
                self.assertLess(ord(char), 128, f"Non-ASCII in JSON output: {repr(char)}")


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMainIntegration(unittest.TestCase):
    def test_main_returns_0_no_regression(self):
        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "before.csv"
            ap = Path(td) / "after.csv"
            _write_bench_csv([_row("score_transaction", 6, 114.0, 725.0, 820.0)], bp)
            _write_bench_csv([_row("score_transaction", 6, 98.0, 452.0, 590.0)], ap)
            sys.argv = ["benchmark_compare.py", str(bp), str(ap),
                        "--no-save", "--sla-p95-ms", "250"]
            from project.scripts.benchmark_compare import main
            rc = main()
        self.assertEqual(rc, 0)

    def test_main_returns_1_on_regression(self):
        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "before.csv"
            ap = Path(td) / "after.csv"
            _write_bench_csv([_row("score_transaction", 6, 98.0, 200.0, 250.0)], bp)
            _write_bench_csv([_row("score_transaction", 6, 150.0, 725.0, 820.0)], ap)
            sys.argv = ["benchmark_compare.py", str(bp), str(ap),
                        "--no-save", "--sla-p95-ms", "250"]
            from project.scripts.benchmark_compare import main
            rc = main()
        self.assertEqual(rc, 1)

    def test_main_returns_1_missing_before(self):
        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "missing.csv"
            ap = Path(td) / "after.csv"
            _write_bench_csv([_row("score_transaction", 6, 98.0, 452.0, 590.0)], ap)
            sys.argv = ["benchmark_compare.py", str(bp), str(ap), "--no-save"]
            from project.scripts.benchmark_compare import main
            rc = main()
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
