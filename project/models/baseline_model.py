import argparse
import json
import os
import pickle
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_MODEL_OUTPUT = REPO_ROOT / "project" / "models" / "baseline_logreg_model.pkl"
DEFAULT_SCALER_OUTPUT = REPO_ROOT / "project" / "models" / "baseline_scaler.pkl"
DEFAULT_METRICS_OUTPUT = REPO_ROOT / "project" / "outputs" / "baseline_metrics.json"


def _resolve_path(arg_value: str | None, env_var: str, default_path: Path) -> Path:
    chosen = arg_value or os.getenv(env_var)
    return Path(chosen).expanduser().resolve() if chosen else default_path


def _validate_dataset(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Set --dataset-path or DATASET_PATH to a valid CSV file."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline logistic regression model.")
    parser.add_argument("--dataset-path", type=str, help="Path to creditcard.csv")
    parser.add_argument("--model-output", type=str, help="Output path for baseline model pickle")
    parser.add_argument("--scaler-output", type=str, help="Output path for scaler pickle")
    parser.add_argument("--metrics-output", type=str, help="Output path for metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_path(args.dataset_path, "DATASET_PATH", DEFAULT_DATASET_PATH)
    model_output = _resolve_path(args.model_output, "BASELINE_MODEL_OUTPUT", DEFAULT_MODEL_OUTPUT)
    scaler_output = _resolve_path(args.scaler_output, "BASELINE_SCALER_OUTPUT", DEFAULT_SCALER_OUTPUT)
    metrics_output = _resolve_path(args.metrics_output, "BASELINE_METRICS_OUTPUT", DEFAULT_METRICS_OUTPUT)

    _validate_dataset(dataset_path)

    # Delayed third-party imports so `--help` works even if ML deps are not installed.
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("BASELINE MODEL - LOGISTIC REGRESSION")
    print("=" * 60)
    print(f"Dataset loaded from: {dataset_path}")

    df = pd.read_csv(dataset_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nTrain class distribution:")
    print(y_train.value_counts())
    print("\nTest class distribution:")
    print(y_test.value_counts())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    scaler_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    with model_output.open("wb") as f:
        pickle.dump(model, f)

    with scaler_output.open("wb") as f:
        pickle.dump(scaler, f)

    metrics_payload = {
        "dataset_path": str(dataset_path),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "model_output": str(model_output),
        "scaler_output": str(scaler_output),
    }
    with metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved files:")
    print(f"- {model_output}")
    print(f"- {scaler_output}")
    print(f"- {metrics_output}")


if __name__ == "__main__":
    main()