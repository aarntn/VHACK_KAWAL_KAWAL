import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_LOG = REPO_ROOT / "project" / "outputs" / "audit" / "fraud_audit_log.jsonl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "latency_stage_analysis.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "latency_stage_analysis.csv"
STAGE_ALIASES = {
    "feature_preparation_ms": "preprocessing_ms",
    "model_inference_ms": "model_predict_ms",
    "context_scoring_ms": "context_adjustment_ms",
    "audit_log_write_ms": "audit_logging_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stage_timings_ms from fraud audit logs for latency bottlenecks."
    )
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-records", type=int, default=50000)
    return parser.parse_args()


def normalize_stage_dict(stages: Dict[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for stage_name, raw_value in stages.items():
        canonical_name = STAGE_ALIASES.get(stage_name, stage_name)
        try:
            normalized[canonical_name] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return normalized


def load_stage_rows(path: Path, max_records: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    dominant_stage_rows: List[Dict[str, str | float]] = []
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(), pd.DataFrame()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stages = record.get("stage_timings_ms") or record.get("timing_spans_ms")
            if isinstance(stages, dict) and stages:
                normalized = normalize_stage_dict(stages)
                if not normalized:
                    continue
                rows.append(normalized)

                candidate_stages = {
                    name: value
                    for name, value in normalized.items()
                    if name != "total_pipeline_ms"
                }
                if candidate_stages:
                    dominant_stage = max(candidate_stages, key=candidate_stages.get)
                    dominant_stage_rows.append(
                        {
                            "dominant_stage": dominant_stage,
                            "dominant_stage_ms": candidate_stages[dominant_stage],
                        }
                    )

    if max_records > 0 and len(rows) > max_records:
        rows = rows[-max_records:]
        dominant_stage_rows = dominant_stage_rows[-max_records:]
    return pd.DataFrame(rows), pd.DataFrame(dominant_stage_rows)


def summarize_stages(df: pd.DataFrame) -> pd.DataFrame:
    out: List[dict] = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        stage_summary = {
            "stage": col,
            "count": int(s.count()),
            "mean_ms": float(s.mean()),
            "p50_ms": float(s.quantile(0.50)),
            "p95_ms": float(s.quantile(0.95)),
            "p99_ms": float(s.quantile(0.99)),
        }

        if "total_pipeline_ms" in df.columns and col != "total_pipeline_ms":
            total_s = pd.to_numeric(df["total_pipeline_ms"], errors="coerce")
            valid_totals = total_s[(total_s > 0) & total_s.notna()]
            aligned_stage = s.reindex(valid_totals.index).dropna()
            if not aligned_stage.empty:
                share = (aligned_stage / valid_totals.reindex(aligned_stage.index)).clip(lower=0)
                if not share.empty:
                    stage_summary["mean_share_of_total_pct"] = float(share.mean() * 100.0)

        out.append(stage_summary)
    return pd.DataFrame(out).sort_values(by="p95_ms", ascending=False)


def main() -> None:
    args = parse_args()
    df, dominant_df = load_stage_rows(args.audit_log, args.max_records)

    summary_df = summarize_stages(df) if not df.empty else pd.DataFrame()
    dominant = summary_df.iloc[0].to_dict() if not summary_df.empty else None

    dominant_frequency = []
    if not dominant_df.empty and not df.empty:
        freq = (
            dominant_df["dominant_stage"]
            .value_counts(dropna=True)
            .rename_axis("stage")
            .reset_index(name="records_where_dominant")
        )
        freq["share_pct"] = (freq["records_where_dominant"] / float(len(dominant_df))) * 100.0
        dominant_frequency = freq.to_dict(orient="records")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(args.output_csv, index=False)
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_log": str(args.audit_log),
        "records_analyzed": int(len(df)),
        "has_stage_data": not df.empty,
        "dominant_stage_by_p95": dominant,
        "dominant_stage_frequency": dominant_frequency,
        "stages": summary_df.to_dict(orient="records"),
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(
        json.dumps(
            {
                "records_analyzed": len(df),
                "has_stage_data": not df.empty,
                "dominant_stage": (dominant or {}).get("stage"),
            },
            indent=2,
        )
    )
    print(f"Saved stage analysis JSON: {args.output_json}")
    print(f"Saved stage analysis CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
