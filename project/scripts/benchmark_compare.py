#!/usr/bin/env python3
"""Compare two benchmark CSV files and output a delta table with SLA pass/fail.

Usage:
    python benchmark_compare.py before.csv after.csv
    python benchmark_compare.py before.csv after.csv --sla-p95-ms 250 --sla-p99-ms 500
    python benchmark_compare.py before.csv after.csv --out project/outputs/benchmark/

Outputs:
    - Console table (stdout)
    - comparison_<stamp>.json under --out dir
    - comparison_<stamp>.csv under --out dir
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "benchmark"

SLA_P95_MS_DEFAULT = 250.0
SLA_P99_MS_DEFAULT = 500.0
SLA_ERROR_RATE_PCT_DEFAULT = 1.0


@dataclass
class BenchRow:
    endpoint_name: str
    concurrency: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate_pct: float
    throughput_rps: float
    requests_total: int
    success_count: int
    error_count: int


def _pct_delta(before: float, after: float) -> Optional[float]:
    if before == 0:
        return None
    return round((after - before) / before * 100, 1)


def _abs_delta(before: float, after: float) -> float:
    return round(after - before, 2)


def _sla_pass(p95: float, p99: float, error_rate: float, sla_p95: float, sla_p99: float, sla_err: float) -> str:
    return "PASS" if p95 <= sla_p95 and p99 <= sla_p99 and error_rate <= sla_err else "FAIL"


def load_csv(path: Path) -> list[BenchRow]:
    rows: list[BenchRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(BenchRow(
                endpoint_name=r["endpoint_name"].strip(),
                concurrency=int(r["concurrency"]),
                p50_ms=float(r["p50_ms"]),
                p95_ms=float(r["p95_ms"]),
                p99_ms=float(r["p99_ms"]),
                error_rate_pct=float(r["error_rate_pct"]),
                throughput_rps=float(r["throughput_rps"]),
                requests_total=int(r["requests_total"]),
                success_count=int(r["success_count"]),
                error_count=int(r["error_count"]),
            ))
    return rows


@dataclass
class DeltaRow:
    endpoint_name: str
    concurrency: int
    before_p50: float
    after_p50: float
    delta_p50: float
    pct_p50: Optional[float]
    before_p95: float
    after_p95: float
    delta_p95: float
    pct_p95: Optional[float]
    before_p99: float
    after_p99: float
    delta_p99: float
    pct_p99: Optional[float]
    before_err_pct: float
    after_err_pct: float
    before_sla: str
    after_sla: str
    sla_changed: str


def build_delta(
    before_rows: list[BenchRow],
    after_rows: list[BenchRow],
    sla_p95: float,
    sla_p99: float,
    sla_err: float,
) -> list[DeltaRow]:
    before_map = {(r.endpoint_name, r.concurrency): r for r in before_rows}
    after_map = {(r.endpoint_name, r.concurrency): r for r in after_rows}
    all_keys = sorted(set(before_map) | set(after_map))

    deltas: list[DeltaRow] = []
    for key in all_keys:
        b = before_map.get(key)
        a = after_map.get(key)
        if b is None or a is None:
            continue
        b_sla = _sla_pass(b.p95_ms, b.p99_ms, b.error_rate_pct, sla_p95, sla_p99, sla_err)
        a_sla = _sla_pass(a.p95_ms, a.p99_ms, a.error_rate_pct, sla_p95, sla_p99, sla_err)
        if b_sla == "FAIL" and a_sla == "PASS":
            sla_changed = "FAIL->PASS"
        elif b_sla == "PASS" and a_sla == "FAIL":
            sla_changed = "PASS->FAIL"
        else:
            sla_changed = "no change"
        deltas.append(DeltaRow(
            endpoint_name=key[0],
            concurrency=key[1],
            before_p50=b.p50_ms,
            after_p50=a.p50_ms,
            delta_p50=_abs_delta(b.p50_ms, a.p50_ms),
            pct_p50=_pct_delta(b.p50_ms, a.p50_ms),
            before_p95=b.p95_ms,
            after_p95=a.p95_ms,
            delta_p95=_abs_delta(b.p95_ms, a.p95_ms),
            pct_p95=_pct_delta(b.p95_ms, a.p95_ms),
            before_p99=b.p99_ms,
            after_p99=a.p99_ms,
            delta_p99=_abs_delta(b.p99_ms, a.p99_ms),
            pct_p99=_pct_delta(b.p99_ms, a.p99_ms),
            before_err_pct=b.error_rate_pct,
            after_err_pct=a.error_rate_pct,
            before_sla=b_sla,
            after_sla=a_sla,
            sla_changed=sla_changed,
        ))
    return deltas


def _fmt_delta(val: float, pct: Optional[float]) -> str:
    sign = "+" if val > 0 else ""
    pct_str = f" ({'+' if (pct or 0) > 0 else ''}{pct:.1f}%)" if pct is not None else ""
    return f"{sign}{val:.1f}ms{pct_str}"


def print_table(deltas: list[DeltaRow], before_label: str, after_label: str) -> None:
    col_w = [24, 5, 10, 10, 22, 10, 22, 10, 22, 8, 8, 12]
    headers = [
        "endpoint", "c", f"p50 {before_label[:4]}", f"p50 {after_label[:4]}", "dp50",
        f"p95 {before_label[:4]}", "dp95", f"p99 {before_label[:4]}", "dp99",
        "before", "after", "SLA change",
    ]
    sep = "  ".join("-" * w for w in col_w)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print()
    print(f"  Benchmark Comparison  |  before={before_label}  after={after_label}")
    print(sep)
    print(header_line)
    print(sep)
    for d in deltas:
        row = [
            d.endpoint_name[:col_w[0]],
            str(d.concurrency),
            f"{d.before_p50:.1f}",
            f"{d.after_p50:.1f}",
            _fmt_delta(d.delta_p50, d.pct_p50),
            f"{d.before_p95:.1f}",
            _fmt_delta(d.delta_p95, d.pct_p95),
            f"{d.before_p99:.1f}",
            _fmt_delta(d.delta_p99, d.pct_p99),
            d.before_sla,
            d.after_sla,
            d.sla_changed,
        ]
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))
    print(sep)
    print()


def write_outputs(deltas: list[DeltaRow], before_path: Path, after_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"comparison_{stamp}.json"
    csv_path = out_dir / f"comparison_{stamp}.csv"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "before_file": str(before_path),
        "after_file": str(after_path),
        "rows": [d.__dict__ for d in deltas],
        "summary": {
            "total_comparisons": len(deltas),
            "fail_to_pass": sum(1 for d in deltas if d.sla_changed == "FAIL->PASS"),
            "pass_to_fail": sum(1 for d in deltas if d.sla_changed == "PASS->FAIL"),
            "avg_p95_delta_ms": round(sum(d.delta_p95 for d in deltas) / len(deltas), 2) if deltas else 0,
            "avg_p99_delta_ms": round(sum(d.delta_p99 for d in deltas) / len(deltas), 2) if deltas else 0,
        },
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    fieldnames = [
        "endpoint_name", "concurrency",
        "before_p50", "after_p50", "delta_p50", "pct_p50",
        "before_p95", "after_p95", "delta_p95", "pct_p95",
        "before_p99", "after_p99", "delta_p99", "pct_p99",
        "before_err_pct", "after_err_pct",
        "before_sla", "after_sla", "sla_changed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for d in deltas:
            writer.writerow(d.__dict__)

    return {"json": str(json_path), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark CSVs and produce a delta report.")
    parser.add_argument("before", type=Path, help="Baseline benchmark CSV (before)")
    parser.add_argument("after", type=Path, help="New benchmark CSV (after)")
    parser.add_argument("--sla-p95-ms", type=float, default=SLA_P95_MS_DEFAULT)
    parser.add_argument("--sla-p99-ms", type=float, default=SLA_P99_MS_DEFAULT)
    parser.add_argument("--sla-error-rate-pct", type=float, default=SLA_ERROR_RATE_PCT_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true", help="Print only, do not save files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before_path: Path = args.before
    after_path: Path = args.after

    if not before_path.exists():
        print(f"ERROR: before file not found: {before_path}", file=sys.stderr)
        return 1
    if not after_path.exists():
        print(f"ERROR: after file not found: {after_path}", file=sys.stderr)
        return 1

    before_rows = load_csv(before_path)
    after_rows = load_csv(after_path)

    if not before_rows:
        print("ERROR: before CSV has no data rows", file=sys.stderr)
        return 1
    if not after_rows:
        print("ERROR: after CSV has no data rows", file=sys.stderr)
        return 1

    deltas = build_delta(before_rows, after_rows, args.sla_p95_ms, args.sla_p99_ms, args.sla_error_rate_pct)

    if not deltas:
        print("WARNING: no matching (endpoint, concurrency) pairs found between the two files.", file=sys.stderr)
        return 1

    print_table(deltas, before_path.stem, after_path.stem)

    summary = {
        "total_comparisons": len(deltas),
        "fail_to_pass": sum(1 for d in deltas if d.sla_changed == "FAIL->PASS"),
        "pass_to_fail": sum(1 for d in deltas if d.sla_changed == "PASS->FAIL"),
        "avg_p95_delta_ms": round(sum(d.delta_p95 for d in deltas) / len(deltas), 2),
        "avg_p99_delta_ms": round(sum(d.delta_p99 for d in deltas) / len(deltas), 2),
    }
    print(f"  Summary: {summary['total_comparisons']} comparisons | "
          f"FAIL->PASS: {summary['fail_to_pass']} | PASS->FAIL: {summary['pass_to_fail']} | "
          f"avg dp95: {summary['avg_p95_delta_ms']:+.1f}ms | avg dp99: {summary['avg_p99_delta_ms']:+.1f}ms")
    print()

    if not args.no_save:
        paths = write_outputs(deltas, before_path, after_path, args.out)
        print(f"  Saved: {paths['json']}")
        print(f"         {paths['csv']}")
        print()

    any_regression = any(d.sla_changed == "PASS->FAIL" for d in deltas)
    return 1 if any_regression else 0


if __name__ == "__main__":
    sys.exit(main())
