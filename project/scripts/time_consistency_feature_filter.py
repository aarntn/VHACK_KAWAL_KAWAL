import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.dataset_loader import load_creditcard, load_ieee_cis

DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "monitoring"


@dataclass
class FeatureConsistencyRow:
    candidate: str
    candidate_type: str
    feature_count: int
    train_auc: float
    validation_auc: float
    auc_delta: float
    inversion_flag: bool
    keep: bool
    drop_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate time consistency of features/feature blocks on early-train vs late-validation windows."
    )
    parser.add_argument("--dataset-source", choices=["creditcard", "ieee_cis"], default="ieee_cis")
    parser.add_argument("--label-policy", choices=["transaction", "account_propagated"], default="transaction")
    parser.add_argument("--dataset-path", type=Path, help="Legacy credit-card CSV path (used when --dataset-source creditcard)")
    parser.add_argument("--ieee-transaction-path", type=Path)
    parser.add_argument("--ieee-identity-path", type=Path)
    parser.add_argument("--feature-groups-file", type=Path, help="Optional JSON mapping {group_name: [feature,..]}")
    parser.add_argument("--candidate-features-file", type=Path, help="Optional newline file to limit single-feature candidates")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-auc", type=float, default=0.60)
    parser.add_argument("--min-validation-auc", type=float, default=0.53)
    parser.add_argument("--max-auc-drop", type=float, default=0.10)
    parser.add_argument("--inversion-validation-auc-max", type=float, default=0.50)
    parser.add_argument(
        "--adversarial-report-json",
        type=Path,
        help="Optional adversarial_validation_report.json used to exclude split-predictive features.",
    )
    parser.add_argument(
        "--adversarial-max-importance",
        type=float,
        help="Maximum allowed adversarial combined importance; features above are excluded.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, dict]:
    if args.dataset_source == "creditcard":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for creditcard source (legacy assets now in project/legacy_creditcard)")
        x, y, metadata = load_creditcard(args.dataset_path, label_policy=args.label_policy)
        return x, y, metadata

    if not args.ieee_transaction_path or not args.ieee_identity_path:
        raise ValueError("--ieee-transaction-path and --ieee-identity-path are required for ieee_cis source")
    x, y, metadata = load_ieee_cis(
        args.ieee_transaction_path,
        args.ieee_identity_path,
        label_policy=args.label_policy,
    )
    return x, y, metadata


def resolve_event_time_column(features: pd.DataFrame, dataset_source: str) -> pd.Series:
    col = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    if col in features.columns:
        return pd.to_numeric(features[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(np.arange(len(features), dtype=float), index=features.index)


def build_time_windows(event_time: pd.Series, train_fraction: float, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0 < train_fraction < 1.0) or not (0 < validation_fraction < 1.0):
        raise ValueError("train_fraction and validation_fraction must be in (0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.0 to preserve a temporal gap")

    order = np.argsort(event_time.to_numpy(), kind="mergesort")
    n = len(order)
    train_end = max(1, min(int(n * train_fraction), n - 2))
    val_start = max(train_end + 1, int(n * (1.0 - validation_fraction)))
    val_start = min(val_start, n - 1)

    train_idx = order[:train_end]
    val_idx = order[val_start:]
    if len(val_idx) == 0:
        val_idx = order[-1:]
    return train_idx, val_idx


def _coerce_numeric_block(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].fillna("<NA>").astype(str)
            out[col] = pd.factorize(out[col])[0]
        elif str(out[col].dtype).startswith("category"):
            out[col] = out[col].cat.codes
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_candidate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    candidate_name: str,
    candidate_type: str,
    min_train_auc: float,
    min_validation_auc: float,
    max_auc_drop: float,
    inversion_validation_auc_max: float,
    seed: int,
) -> FeatureConsistencyRow:
    from xgboost import XGBClassifier

    x_train_num = _coerce_numeric_block(x_train)
    x_val_num = _coerce_numeric_block(x_val)

    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
    )
    model.fit(x_train_num, y_train)

    train_score = model.predict_proba(x_train_num)[:, 1]
    val_score = model.predict_proba(x_val_num)[:, 1]

    train_auc = _safe_auc(y_train.to_numpy(), train_score)
    validation_auc = _safe_auc(y_val.to_numpy(), val_score)
    auc_delta = validation_auc - train_auc if np.isfinite(train_auc) and np.isfinite(validation_auc) else float("nan")
    inversion_flag = bool(
        np.isfinite(train_auc)
        and np.isfinite(validation_auc)
        and train_auc >= min_train_auc
        and validation_auc <= inversion_validation_auc_max
    )

    keep = True
    drop_reason = ""
    if not np.isfinite(validation_auc):
        keep = False
        drop_reason = "invalid_validation_auc"
    elif validation_auc < min_validation_auc:
        keep = False
        drop_reason = "validation_auc_below_threshold"
    elif np.isfinite(auc_delta) and auc_delta < (-max_auc_drop):
        keep = False
        drop_reason = "auc_delta_exceeds_max_drop"
    elif inversion_flag:
        keep = False
        drop_reason = "inversion_flag"

    return FeatureConsistencyRow(
        candidate=candidate_name,
        candidate_type=candidate_type,
        feature_count=int(x_train.shape[1]),
        train_auc=float(train_auc) if np.isfinite(train_auc) else float("nan"),
        validation_auc=float(validation_auc) if np.isfinite(validation_auc) else float("nan"),
        auc_delta=float(auc_delta) if np.isfinite(auc_delta) else float("nan"),
        inversion_flag=inversion_flag,
        keep=keep,
        drop_reason=drop_reason,
    )


def load_candidate_features(path: Path | None, available: list[str]) -> list[str]:
    if path is None:
        return available
    lines = [line.strip() for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines()]
    requested = [line for line in lines if line and not line.startswith("#")]
    return [f for f in requested if f in set(available)]


def load_feature_groups(path: Path | None, available: set[str]) -> dict[str, list[str]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("feature-groups-file must be a JSON object mapping name -> list[str]")

    groups: dict[str, list[str]] = {}
    for name, cols in payload.items():
        if not isinstance(cols, list):
            continue
        filtered = [c for c in cols if c in available]
        if filtered:
            groups[str(name)] = filtered
    return groups


def load_adversarial_drop_features(
    report_json: Path | None,
    max_importance: float | None,
    available: set[str],
) -> list[str]:
    if report_json is None or max_importance is None:
        return []

    payload = json.loads(report_json.expanduser().resolve().read_text(encoding="utf-8"))
    ranked = payload.get("ranked_features") if isinstance(payload, dict) else None
    if not isinstance(ranked, list):
        raise ValueError("adversarial-report-json missing ranked_features list")

    dropped: list[str] = []
    for row in ranked:
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature", ""))
        score = row.get("importance_combined")
        if name in available and score is not None and float(score) > float(max_importance):
            dropped.append(name)
    return sorted(set(dropped))


def main() -> None:
    args = parse_args()
    x, y, source_metadata = load_source(args)
    event_time = resolve_event_time_column(x, args.dataset_source)
    train_idx, val_idx = build_time_windows(event_time, args.train_fraction, args.validation_fraction)

    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]
    x_val = x.iloc[val_idx]
    y_val = y.iloc[val_idx]

    single_features = load_candidate_features(args.candidate_features_file, list(x.columns))
    groups = load_feature_groups(args.feature_groups_file, set(x.columns))
    adversarial_drops = load_adversarial_drop_features(
        args.adversarial_report_json,
        args.adversarial_max_importance,
        set(x.columns),
    )
    if adversarial_drops:
        drop_set = set(adversarial_drops)
        single_features = [f for f in single_features if f not in drop_set]
        groups = {
            name: [col for col in cols if col not in drop_set]
            for name, cols in groups.items()
            if any(col not in drop_set for col in cols)
        }

    rows: list[FeatureConsistencyRow] = []
    for feature in single_features:
        rows.append(
            evaluate_candidate(
                x_train=x_train[[feature]],
                y_train=y_train,
                x_val=x_val[[feature]],
                y_val=y_val,
                candidate_name=feature,
                candidate_type="single_feature",
                min_train_auc=args.min_train_auc,
                min_validation_auc=args.min_validation_auc,
                max_auc_drop=args.max_auc_drop,
                inversion_validation_auc_max=args.inversion_validation_auc_max,
                seed=args.seed,
            )
        )

    for group_name, columns in groups.items():
        rows.append(
            evaluate_candidate(
                x_train=x_train[columns],
                y_train=y_train,
                x_val=x_val[columns],
                y_val=y_val,
                candidate_name=group_name,
                candidate_type="feature_group",
                min_train_auc=args.min_train_auc,
                min_validation_auc=args.min_validation_auc,
                max_auc_drop=args.max_auc_drop,
                inversion_validation_auc_max=args.inversion_validation_auc_max,
                seed=args.seed,
            )
        )

    results_df = pd.DataFrame([asdict(r) for r in rows]).sort_values(
        ["keep", "validation_auc", "auc_delta", "train_auc"],
        ascending=[False, False, False, False],
    )

    keep_features = [r.candidate for r in rows if r.keep and r.candidate_type == "single_feature"]
    keep_groups = [r.candidate for r in rows if r.keep and r.candidate_type == "feature_group"]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "time_consistency_feature_scores.csv"
    json_path = output_dir / "time_consistency_feature_decisions.json"
    results_df["label_policy"] = args.label_policy
    results_df.to_csv(csv_path, index=False)

    decisions: dict[str, Any] = {
        "dataset_source": args.dataset_source,
        "label_policy": args.label_policy,
        "label_policy_diagnostics": source_metadata.get("label_policy_diagnostics"),
        "seed": args.seed,
        "thresholds": {
            "min_train_auc": args.min_train_auc,
            "min_validation_auc": args.min_validation_auc,
            "max_auc_drop": args.max_auc_drop,
            "inversion_validation_auc_max": args.inversion_validation_auc_max,
        },
        "windowing": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
        },
        "summary": {
            "candidate_count": int(len(rows)),
            "kept_count": int(sum(1 for r in rows if r.keep)),
            "dropped_count": int(sum(1 for r in rows if not r.keep)),
            "adversarial_dropped_features": adversarial_drops,
            "kept_single_features": keep_features,
            "kept_feature_groups": keep_groups,
        },
        "outputs": {
            "scores_csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    print("Time consistency feature filter complete")
    print(f"- Scores CSV: {csv_path}")
    print(f"- Decisions JSON: {json_path}")
    print(f"- Kept single features: {len(keep_features)} / {len(single_features)}")


if __name__ == "__main__":
    main()
