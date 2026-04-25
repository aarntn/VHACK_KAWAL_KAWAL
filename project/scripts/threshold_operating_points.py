import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "ieee_cis_preprocessing_threshold_comparison.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "ieee_cis_operating_points.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "ieee_cis_operating_points.json"


REQUIRED_COLUMNS = {
    "preprocessing_setting",
    "threshold",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select conservative/balanced/aggressive operating points from a threshold sweep under FPR/recall constraints."
    )
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    parser.add_argument("--setting", default="onehot_robust", help="preprocessing_setting value to filter")
    parser.add_argument("--fpr-cap", type=float, default=0.05)
    parser.add_argument("--min-recall", type=float, default=0.30)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def _pick_operating_points(feasible: pd.DataFrame) -> pd.DataFrame:
    # Conservative: strongest FPR control (higher threshold tie-break).
    conservative = feasible.sort_values(
        ["false_positive_rate", "threshold", "precision"],
        ascending=[True, False, False],
    ).iloc[0]

    # Aggressive: highest recall among feasible candidates (lower threshold tie-break).
    aggressive = feasible.sort_values(
        ["recall", "threshold", "f1"],
        ascending=[False, True, False],
    ).iloc[0]

    # Balanced: maximize F1, but avoid reusing conservative/aggressive threshold when alternatives exist.
    balanced_pool = feasible[~feasible["threshold"].isin({conservative["threshold"], aggressive["threshold"]})]
    if balanced_pool.empty:
        balanced_pool = feasible[feasible["threshold"] != conservative["threshold"]]
    if balanced_pool.empty:
        balanced_pool = feasible
    balanced = balanced_pool.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    rows = []
    for profile, row in [("conservative", conservative), ("balanced", balanced), ("aggressive", aggressive)]:
        row_df = row.to_frame().T.copy()
        row_df.insert(0, "profile", profile)
        rows.append(row_df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.sweep_csv)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Threshold sweep is missing columns: {missing}")

    setting_df = df[df["preprocessing_setting"] == args.setting].copy()
    if setting_df.empty:
        raise ValueError(f"No rows found for preprocessing_setting={args.setting!r} in {args.sweep_csv}")

    feasible = setting_df[
        (setting_df["false_positive_rate"] <= float(args.fpr_cap))
        & (setting_df["recall"] >= float(args.min_recall))
    ].copy()
    if feasible.empty:
        raise ValueError(
            "No thresholds satisfy constraints. Try relaxing --fpr-cap or --min-recall."
        )

    operating = _pick_operating_points(feasible)
    operating["fpr_cap"] = float(args.fpr_cap)
    operating["min_recall"] = float(args.min_recall)
    operating = operating[
        [
            "profile",
            "preprocessing_setting",
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "pr_auc",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "fpr_cap",
            "min_recall",
        ]
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    operating.to_csv(args.output_csv, index=False)
    payload = {
        "source_sweep_csv": str(args.sweep_csv),
        "setting": args.setting,
        "constraints": {
            "fpr_cap": float(args.fpr_cap),
            "min_recall": float(args.min_recall),
        },
        "candidate_count_in_setting": int(len(setting_df)),
        "candidate_count_feasible": int(len(feasible)),
        "profiles": operating.to_dict(orient="records"),
        "inference_policy": {
            "default_profile": "balanced",
            "override_condition": "switch to conservative if business requires stricter FPR than balanced profile",
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Threshold operating point selection complete")
    print(f"- Feasible points: {len(feasible)} / {len(setting_df)}")
    print(f"- Output CSV: {args.output_csv}")
    print(f"- Output JSON: {args.output_json}")


if __name__ == "__main__":
    main()
