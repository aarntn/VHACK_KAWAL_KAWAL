import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "outputs" / "figures" / "tables"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare class-imbalance handling strategies for fraud detection."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def _fpr(y_true: pd.Series, y_pred: pd.Series) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    return {
        "strategy": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "false_positive_rate": _fpr(y_test, y_pred),
    }


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {dataset_path}. Use --dataset-path to point to creditcard.csv"
        )

    df = pd.read_csv(dataset_path)
    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    experiments = []

    xgb_scale_pos_weight = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )
    experiments.append(
        evaluate_model(
            f"xgboost_scale_pos_weight({scale_pos_weight:.2f})",
            xgb_scale_pos_weight,
            x_train,
            y_train,
            x_test,
            y_test,
        )
    )

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE experiment requires imbalanced-learn. Install with `pip install imbalanced-learn`."
        ) from exc

    smote = SMOTE(random_state=args.seed)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    smote_baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=args.seed)),
        ]
    )
    experiments.append(
        evaluate_model(
            "smote_plus_logistic_regression",
            smote_baseline,
            x_train_smote,
            y_train_smote,
            x_test,
            y_test,
        )
    )

    class_weighted_baseline = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    experiments.append(
        evaluate_model(
            "class_weighted_random_forest",
            class_weighted_baseline,
            x_train,
            y_train,
            x_test,
            y_test,
        )
    )

    results_df = pd.DataFrame(experiments).sort_values(by="f1", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "class_imbalance_experiment_metrics.csv"
    metrics_md = output_dir / "class_imbalance_experiment_metrics.md"

    results_df.to_csv(metrics_csv, index=False)
    metrics_md.write_text(results_df.to_markdown(index=False) + "\n", encoding="utf-8")

    print("Class imbalance experiments complete.")
    print(f"Split seed: {args.seed}")
    print(f"Train rows: {len(x_train):,}, Test rows: {len(x_test):,}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved metrics Markdown: {metrics_md}")
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
