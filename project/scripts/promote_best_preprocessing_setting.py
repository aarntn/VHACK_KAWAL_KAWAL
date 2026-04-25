import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MONITORING_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"
DEFAULT_MODELS_DIR = REPO_ROOT / "project" / "models"
DEFAULT_PROMOTION_REPORT = REPO_ROOT / "project" / "outputs" / "monitoring" / "preprocessing_promotion_report.json"


class SelectionPolicyError(ValueError):
    def __init__(self, message: str, diagnostics: dict, remediation: list[str]):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.remediation = remediation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote best preprocessing setting from evaluation outputs and optionally retrain final model."
    )
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--best-csv", type=Path, help="Path to *_preprocessing_threshold_best.csv")
    parser.add_argument("--comparison-csv", type=Path, help="Path to *_preprocessing_threshold_comparison.csv")
    parser.add_argument(
        "--selection-scope",
        choices=["best", "full"],
        default="best",
        help="Use pre-aggregated best-per-setting rows or full threshold comparison rows for selection.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="f1",
        choices=["f1", "recall", "precision"],
        help="Metric column used for selecting winner.",
    )
    parser.add_argument("--max-fpr", type=float, default=0.75, help="False-positive-rate upper bound (IEEE-friendly default: 0.75).")
    parser.add_argument("--min-precision", type=float, default=0.02, help="Precision lower bound (IEEE-friendly default: 0.02).")
    parser.add_argument("--min-f1", type=float, default=0.0, help="Minimum acceptable F1 for promotion quality gate.")
    parser.add_argument("--min-pr-auc", type=float, default=0.0, help="Minimum acceptable PR-AUC for promotion quality gate.")
    parser.add_argument("--min-recall", type=float, default=0.0, help="Minimum acceptable recall for promotion quality gate.")
    parser.add_argument(
        "--allow-policy-fallback",
        action="store_true",
        help="When policy filters remove all rows, fall back to unconstrained best row and record policy violations in report.",
    )
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODELS_DIR / "final_xgboost_model_promoted_preproc.pkl")
    parser.add_argument("--features-output", type=Path, default=DEFAULT_MODELS_DIR / "feature_columns_promoted_preproc.pkl")
    parser.add_argument("--thresholds-output", type=Path, default=DEFAULT_MODELS_DIR / "decision_thresholds_promoted_preproc.pkl")
    parser.add_argument("--preprocessing-artifact-output", type=Path, default=DEFAULT_MODELS_DIR / "preprocessing_artifact_promoted.pkl")
    parser.add_argument("--promotion-report", type=Path, default=DEFAULT_PROMOTION_REPORT)
    parser.add_argument(
        "--validation-robustness-report",
        type=Path,
        default=DEFAULT_MONITORING_DIR / "ieee_cis_validation_robustness_report.json",
        help="Robustness report JSON from tune_model_candidates.py (defaults to ieee_cis_validation_robustness_report.json).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected setting and training command without executing")
    parser.add_argument("--entity-smoothing-method", choices=["none", "mean", "ema", "blend"], default="none")
    parser.add_argument("--entity-smoothing-min-history", type=int, default=2)
    parser.add_argument("--entity-smoothing-ema-alpha", type=float, default=0.3)
    parser.add_argument("--entity-smoothing-blend-alpha", type=float, default=0.5)
    parser.add_argument("--entity-smoothing-blend-cap", type=float, default=0.25)
    return parser.parse_args()


def parse_setting_name(setting_name: str) -> tuple[str, str]:
    parts = setting_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid preprocessing setting format: {setting_name}")
    encoding, scaler = parts[0], parts[1]
    if encoding not in {"onehot", "frequency"}:
        raise ValueError(f"Unsupported encoding in setting: {setting_name}")
    if scaler not in {"standard", "robust"}:
        raise ValueError(f"Unsupported scaler in setting: {setting_name}")
    return encoding, scaler


def resolve_selection_csv(args: argparse.Namespace) -> Path:
    if args.selection_scope == "full":
        if args.comparison_csv:
            return args.comparison_csv.expanduser().resolve()
        return (DEFAULT_MONITORING_DIR / f"{args.dataset_source}_preprocessing_threshold_comparison.csv").resolve()

    if args.best_csv:
        return args.best_csv.expanduser().resolve()
    return (DEFAULT_MONITORING_DIR / f"{args.dataset_source}_preprocessing_threshold_best.csv").resolve()


def _apply_policy_filters(df: pd.DataFrame, max_fpr: float | None, min_precision: float | None) -> pd.DataFrame:
    filtered = df.copy()

    if max_fpr is not None:
        filtered = filtered[filtered["false_positive_rate"] <= float(max_fpr)]
    if min_precision is not None:
        filtered = filtered[filtered["precision"] >= float(min_precision)]

    return filtered


def _validate_policy_columns(df: pd.DataFrame, max_fpr: float | None, min_precision: float | None) -> None:
    missing: list[str] = []
    if max_fpr is not None and "false_positive_rate" not in df.columns:
        missing.append("false_positive_rate")
    if min_precision is not None and "precision" not in df.columns:
        missing.append("precision")

    if missing:
        raise ValueError(
            "Selection policy requires missing columns: "
            f"{missing}. Available columns: {df.columns.tolist()}"
        )


def build_policy_diagnostics(df: pd.DataFrame, max_fpr: float | None, min_precision: float | None) -> dict:
    diagnostics: dict = {
        "candidate_count_before_policy": int(len(df)),
    }
    if "false_positive_rate" in df.columns:
        diagnostics["false_positive_rate_range"] = {
            "min": float(df["false_positive_rate"].min()),
            "max": float(df["false_positive_rate"].max()),
        }
    if "precision" in df.columns:
        diagnostics["precision_range"] = {
            "min": float(df["precision"].min()),
            "max": float(df["precision"].max()),
        }
    diagnostics["requested_policy"] = {
        "max_fpr": max_fpr,
        "min_precision": min_precision,
    }
    return diagnostics


def build_policy_remediation(diagnostics: dict) -> list[str]:
    remediation: list[str] = [
        "Run evaluate_preprocessing_settings.py with a wider threshold sweep, then retry promotion.",
        "Use --selection-scope full to evaluate all threshold rows.",
    ]

    requested = diagnostics.get("requested_policy") or {}
    precision_range = diagnostics.get("precision_range") or {}
    fpr_range = diagnostics.get("false_positive_rate_range") or {}

    if requested.get("min_precision") is not None and precision_range:
        remediation.append(
            f"Requested min_precision={requested['min_precision']} exceeds observed max precision={precision_range.get('max')}. Lower --min-precision."
        )
    if requested.get("max_fpr") is not None and fpr_range:
        remediation.append(
            f"Requested max_fpr={requested['max_fpr']} is stricter than needed; observed min/max FPR={fpr_range.get('min')}..{fpr_range.get('max')}."
        )

    remediation.append("If you must proceed despite policy infeasibility, use --allow-policy-fallback together with strict quality gates.")
    return remediation


def rank_rows(df: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    tie_break_columns = [c for c in [selection_metric, "f1", "recall", "precision"] if c in df.columns]
    return df.sort_values(tie_break_columns, ascending=[False] * len(tie_break_columns))


def policy_violations_for_row(row: pd.Series, max_fpr: float | None, min_precision: float | None) -> list[str]:
    violations: list[str] = []
    if max_fpr is not None and "false_positive_rate" in row and float(row["false_positive_rate"]) > max_fpr:
        violations.append(f"false_positive_rate={float(row['false_positive_rate']):.6f} > max_fpr={max_fpr}")
    if min_precision is not None and "precision" in row and float(row["precision"]) < min_precision:
        violations.append(f"precision={float(row['precision']):.6f} < min_precision={min_precision}")
    return violations


def evaluate_quality_gate(
    row: pd.Series,
    min_f1: float,
    min_pr_auc: float,
    min_recall: float,
    max_fpr: float | None,
) -> dict:
    checks = {
        "f1": {
            "required": min_f1,
            "actual": float(row.get("f1", float("nan"))),
        },
        "pr_auc": {
            "required": min_pr_auc,
            "actual": float(row.get("pr_auc", float("nan"))),
        },
        "recall": {
            "required": min_recall,
            "actual": float(row.get("recall", float("nan"))),
        },
    }
    if max_fpr is not None:
        checks["false_positive_rate"] = {
            "required_max": float(max_fpr),
            "actual": float(row.get("false_positive_rate", float("nan"))),
        }

    checks["f1"]["passed"] = bool(pd.notna(checks["f1"]["actual"])) and checks["f1"]["actual"] >= min_f1
    checks["pr_auc"]["passed"] = bool(pd.notna(checks["pr_auc"]["actual"])) and checks["pr_auc"]["actual"] >= min_pr_auc
    checks["recall"]["passed"] = bool(pd.notna(checks["recall"]["actual"])) and checks["recall"]["actual"] >= min_recall
    if "false_positive_rate" in checks:
        checks["false_positive_rate"]["passed"] = bool(pd.notna(checks["false_positive_rate"]["actual"])) and checks[
            "false_positive_rate"
        ]["actual"] <= max_fpr

    failed = [name for name, payload in checks.items() if not payload.get("passed", False)]
    return {
        "passed": len(failed) == 0,
        "failed_metrics": failed,
        "checks": checks,
    }


def build_quality_gate_next_actions(quality_gate: dict) -> list[str]:
    failed = quality_gate.get("failed_metrics") or []
    if not failed:
        return ["Quality gate passed; proceed with promotion training or keep --dry-run for inspection."]

    actions: list[str] = [
        "Relax quality thresholds (e.g., --min-f1/--min-pr-auc/--min-recall/--max-fpr) only if business risk tolerance allows.",
        "Improve candidate quality before promotion: revisit feature engineering, adversarial filtering threshold, and label policy consistency.",
        "Re-run tune_model_candidates.py and evaluate_preprocessing_settings.py after changes, then retry promotion with the refreshed artifacts.",
    ]

    if "false_positive_rate" in failed:
        actions.append("Lower false positive rate by increasing decision threshold or tightening preprocessing setting selection policy.")
    if "recall" in failed:
        actions.append("Improve recall via richer behavior/entity aggregate features and by relaxing overly aggressive feature filtering.")
    if "pr_auc" in failed:
        actions.append("Boost PR-AUC through class-imbalance strategy tuning and candidate model/ensemble exploration.")

    return actions


def select_best_row(
    selection_csv: Path,
    selection_metric: str,
    selection_scope: str = "best",
    max_fpr: float | None = None,
    min_precision: float | None = None,
    allow_policy_fallback: bool = False,
) -> tuple[pd.Series, dict]:
    if not selection_csv.exists() or not selection_csv.is_file():
        raise FileNotFoundError(f"Selection CSV not found: {selection_csv}")

    df = pd.read_csv(selection_csv)
    if df.empty:
        raise ValueError(f"Selection CSV is empty: {selection_csv}")
    if selection_metric not in df.columns:
        raise ValueError(f"Selection metric '{selection_metric}' not found in columns: {df.columns.tolist()}")

    _validate_policy_columns(df, max_fpr=max_fpr, min_precision=min_precision)

    diagnostics = build_policy_diagnostics(df, max_fpr=max_fpr, min_precision=min_precision)
    filtered = _apply_policy_filters(df, max_fpr=max_fpr, min_precision=min_precision)
    diagnostics["candidate_count_after_policy"] = int(len(filtered))
    diagnostics["policy_fallback_used"] = False

    if filtered.empty:
        if not allow_policy_fallback:
            remediation = build_policy_remediation(diagnostics)
            raise SelectionPolicyError(
                "No candidate rows remain after applying selection policies: "
                f"max_fpr={max_fpr}, min_precision={min_precision}. "
                f"Diagnostics: {diagnostics}",
                diagnostics=diagnostics,
                remediation=remediation,
            )
        diagnostics["policy_fallback_used"] = True
        fallback_row = rank_rows(df, selection_metric=selection_metric).iloc[0]
        diagnostics["fallback_policy_violations"] = policy_violations_for_row(
            fallback_row,
            max_fpr=max_fpr,
            min_precision=min_precision,
        )
        return fallback_row, diagnostics

    ranked = rank_rows(filtered, selection_metric=selection_metric)
    return ranked.iloc[0], diagnostics


def build_training_command(args: argparse.Namespace, encoding: str, scaler: str) -> list[str]:
    cmd = [
        sys.executable,
        str((REPO_ROOT / "project" / "models" / "final_xgboost_model.py").resolve()),
        "--dataset-source",
        args.dataset_source,
        "--use-preprocessing",
        "--preprocessing-categorical-encoding",
        encoding,
        "--preprocessing-scaler",
        scaler,
        "--model-output",
        str(args.model_output.expanduser().resolve()),
        "--features-output",
        str(args.features_output.expanduser().resolve()),
        "--thresholds-output",
        str(args.thresholds_output.expanduser().resolve()),
        "--preprocessing-artifact-output",
        str(args.preprocessing_artifact_output.expanduser().resolve()),
    ]

    if args.dataset_source == "creditcard":
        cmd.extend(["--dataset-path", str(args.dataset_path.expanduser().resolve())])
    else:
        if not args.ieee_transaction_path or not args.ieee_identity_path:
            raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
        cmd.extend(
            [
                "--ieee-transaction-path",
                str(args.ieee_transaction_path.expanduser().resolve()),
                "--ieee-identity-path",
                str(args.ieee_identity_path.expanduser().resolve()),
            ]
        )

    return cmd


def resolve_robustness_report(args: argparse.Namespace) -> Path:
    if args.validation_robustness_report:
        path = args.validation_robustness_report.expanduser().resolve()
    else:
        path = (DEFAULT_MONITORING_DIR / f"{args.dataset_source}_validation_robustness_report.json").resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Validation robustness report is required for promotion and was not found: {path}"
        )
    return path


def load_and_validate_robustness_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("robustness_gate") or {}
    if not bool(gate.get("passed", False)):
        raise ValueError(
            "Promotion blocked: robustness_gate.passed is not true in validation robustness report. "
            f"report={path}"
        )
    return payload


def resolve_preprocessing_metadata_path(selection_csv: Path) -> Path:
    name = selection_csv.name
    if name.endswith("_preprocessing_threshold_best.csv"):
        meta_name = name.replace("_preprocessing_threshold_best.csv", "_preprocessing_threshold_metadata.json")
    elif name.endswith("_preprocessing_threshold_comparison.csv"):
        meta_name = name.replace("_preprocessing_threshold_comparison.csv", "_preprocessing_threshold_metadata.json")
    else:
        meta_name = selection_csv.stem + "_metadata.json"
    return selection_csv.with_name(meta_name)


def validate_label_policy_alignment(
    expected_label_policy: str,
    robustness_report: dict,
    preprocessing_metadata: dict | None,
) -> dict:
    robustness_policy = robustness_report.get("label_policy")
    preprocessing_policy = (preprocessing_metadata or {}).get("label_policy")

    checks = {
        "expected": expected_label_policy,
        "robustness_report": robustness_policy,
        "preprocessing_metadata": preprocessing_policy,
    }

    mismatches: list[str] = []
    if robustness_policy is not None and str(robustness_policy) != expected_label_policy:
        mismatches.append(f"robustness_report={robustness_policy} != expected={expected_label_policy}")
    if preprocessing_policy is not None and str(preprocessing_policy) != expected_label_policy:
        mismatches.append(f"preprocessing_metadata={preprocessing_policy} != expected={expected_label_policy}")
    if robustness_policy is not None and preprocessing_policy is not None and str(robustness_policy) != str(preprocessing_policy):
        mismatches.append(
            f"robustness_report={robustness_policy} != preprocessing_metadata={preprocessing_policy}"
        )

    return {
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
        "policies": checks,
    }


def main() -> None:
    args = parse_args()
    robustness_report_path = resolve_robustness_report(args)
    robustness_report = load_and_validate_robustness_report(robustness_report_path)
    selection_csv = resolve_selection_csv(args)
    preprocessing_metadata_path = resolve_preprocessing_metadata_path(selection_csv)
    preprocessing_metadata = None
    if preprocessing_metadata_path.exists() and preprocessing_metadata_path.is_file():
        preprocessing_metadata = json.loads(preprocessing_metadata_path.read_text(encoding="utf-8"))

    label_policy_check = validate_label_policy_alignment(
        expected_label_policy=args.label_policy,
        robustness_report=robustness_report,
        preprocessing_metadata=preprocessing_metadata,
    )
    if not label_policy_check["passed"]:
        raise ValueError(
            "Promotion blocked due to mixed/inconsistent label policies: "
            + "; ".join(label_policy_check["mismatches"])
            + f". robustness_report={robustness_report_path}; preprocessing_metadata={preprocessing_metadata_path if preprocessing_metadata else 'not_found'}; observed={label_policy_check['policies']}"
        )
    try:
        row, policy_diagnostics = select_best_row(
            selection_csv,
            args.selection_metric,
            selection_scope=args.selection_scope,
            max_fpr=args.max_fpr,
            min_precision=args.min_precision,
            allow_policy_fallback=args.allow_policy_fallback,
        )
    except SelectionPolicyError as exc:
        failure_report = {
            "dataset_source": args.dataset_source,
            "selection_scope": args.selection_scope,
            "selection_csv": str(selection_csv),
            "selection_metric": args.selection_metric,
            "selection_policy": {
                "max_fpr": args.max_fpr,
                "min_precision": args.min_precision,
            },
            "status": "blocked_selection_policy",
            "policy_diagnostics": exc.diagnostics,
            "remediation": exc.remediation,
        }
        report_path = args.promotion_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        raise ValueError(f"{exc} See promotion report: {report_path}")

    setting_name = str(row["preprocessing_setting"])
    encoding, scaler = parse_setting_name(setting_name)
    command = build_training_command(args, encoding=encoding, scaler=scaler)
    quality_gate = evaluate_quality_gate(
        row=row,
        min_f1=args.min_f1,
        min_pr_auc=args.min_pr_auc,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
    )

    if policy_diagnostics.get("policy_fallback_used") and not quality_gate["checks"]["f1"]["passed"]:
        quality_gate["passed"] = False
        quality_gate["hard_floor_failure_reason"] = (
            "Fallback candidate failed hard floor: f1 below --min-f1 requirement. Promotion refused."
        )

    report = {
        "dataset_source": args.dataset_source,
        "selection_scope": args.selection_scope,
        "label_policy": args.label_policy,
        "validation_robustness_report": str(robustness_report_path),
        "validation_robustness_gate": robustness_report.get("robustness_gate"),
        "preprocessing_metadata": str(preprocessing_metadata_path) if preprocessing_metadata else None,
        "label_policy_check": label_policy_check,
        "selection_csv": str(selection_csv),
        "selection_metric": args.selection_metric,
        "selection_policy": {
            "max_fpr": args.max_fpr,
            "min_precision": args.min_precision,
        },
        "selection_constraints_applied": {
            "max_fpr": args.max_fpr is not None,
            "min_precision": args.min_precision is not None,
        },
        "quality_gate": quality_gate,
        "policy_diagnostics": policy_diagnostics,
        "selected_setting": setting_name,
        "selected_threshold": float(row["threshold"]) if "threshold" in row else None,
        "selected_objective": args.selection_metric,
        "selected_row": row.to_dict(),
        "selected_row_metadata": {
            "preprocessing_setting": setting_name,
            "threshold": float(row["threshold"]) if "threshold" in row else None,
            "selection_metric": args.selection_metric,
            "selection_metric_value": float(row[args.selection_metric]) if args.selection_metric in row else None,
        },
        "next_actions": build_quality_gate_next_actions(quality_gate),
        "resolved_encoding": encoding,
        "resolved_scaler": scaler,
        "training_command": command,
        "entity_smoothing": {
            "method": args.entity_smoothing_method,
            "min_history": args.entity_smoothing_min_history,
            "ema_alpha": args.entity_smoothing_ema_alpha,
            "blend_alpha": args.entity_smoothing_blend_alpha,
            "blend_cap": args.entity_smoothing_blend_cap,
            "selected_row_smoothed_metrics": {
                "f1": row.get("f1_smoothed"),
                "pr_auc": row.get("pr_auc_smoothed"),
                "recall": row.get("recall_smoothed"),
            },
        },
        "dry_run": args.dry_run,
    }

    report_path = args.promotion_report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not quality_gate["passed"]:
        reason = quality_gate.get("hard_floor_failure_reason") or (
            "Promotion blocked by quality gate; failed metrics: " + ", ".join(quality_gate.get("failed_metrics", []))
        )
        raise ValueError(f"{reason}. See promotion report: {report_path}")

    print("Best preprocessing setting selected")
    print(f"- Setting: {setting_name}")
    print(f"- Threshold: {report['selected_threshold']}")
    print(f"- Metric ({args.selection_metric}): {row[args.selection_metric]}")
    print(f"- Report: {report_path}")

    if args.dry_run:
        print("Dry-run enabled; training not executed.")
        print("Command:")
        print(" ".join(command))
        return

    subprocess.run(command, check=True)
    print("Promotion training completed.")


if __name__ == "__main__":
    main()
