import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "project" / "legacy_creditcard" / "creditcard.csv"
DEFAULT_RESULTS_OUTPUT = REPO_ROOT / "project" / "outputs" / "threshold_results.csv"
DEFAULT_METRICS_PLOT_OUTPUT = REPO_ROOT / "project" / "outputs" / "threshold_metrics.png"
DEFAULT_ERRORS_PLOT_OUTPUT = REPO_ROOT / "project" / "outputs" / "threshold_errors.png"
DEFAULT_BEST_OUTPUT = REPO_ROOT / "project" / "outputs" / "best_threshold.json"


def _resolve_path(arg_value: str | None, env_var: str, default_path: Path) -> Path:
    chosen = arg_value or os.getenv(env_var)
    return Path(chosen).expanduser().resolve() if chosen else default_path


def _validate_dataset(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Set --dataset-path or DATASET_PATH to a valid CSV file."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune thresholds for logistic regression fraud model.")
    parser.add_argument("--dataset-path", type=str, help="Path to creditcard.csv")
    parser.add_argument("--results-output", type=str, help="CSV output path for threshold comparison")
    parser.add_argument("--metrics-plot-output", type=str, help="PNG output path for precision/recall/F1 plot")
    parser.add_argument("--errors-plot-output", type=str, help="PNG output path for FP/FN plot")
    parser.add_argument("--best-threshold-output", type=str, help="JSON output path for best threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_path(args.dataset_path, "DATASET_PATH", DEFAULT_DATASET_PATH)
    results_output = _resolve_path(args.results_output, "THRESHOLD_RESULTS_OUTPUT", DEFAULT_RESULTS_OUTPUT)
    metrics_plot_output = _resolve_path(args.metrics_plot_output, "THRESHOLD_METRICS_PLOT_OUTPUT", DEFAULT_METRICS_PLOT_OUTPUT)
    errors_plot_output = _resolve_path(args.errors_plot_output, "THRESHOLD_ERRORS_PLOT_OUTPUT", DEFAULT_ERRORS_PLOT_OUTPUT)
    best_threshold_output = _resolve_path(args.best_threshold_output, "BEST_THRESHOLD_OUTPUT", DEFAULT_BEST_OUTPUT)

    _validate_dataset(dataset_path)

    # Delayed third-party imports so `--help` works even if ML deps are not installed.
    import matplotlib.pyplot as plt
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
    print("THRESHOLD TUNING - LOGISTIC REGRESSION")
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "true_negatives": tn,
            }
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])
    best_pred = (y_proba >= best_threshold).astype(int)

    print("\nBest threshold by F1:")
    print(best_row)
    print("\nClassification Report at Best Threshold:")
    print(classification_report(y_test, best_pred, digits=4))

    for output_path in [results_output, metrics_plot_output, errors_plot_output, best_threshold_output]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(results_output, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["threshold"], results_df["precision"], marker="o", label="Precision")
    plt.plot(results_df["threshold"], results_df["recall"], marker="o", label="Recall")
    plt.plot(results_df["threshold"], results_df["f1"], marker="o", label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(metrics_plot_output)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["threshold"], results_df["false_positives"], marker="o", label="False Positives")
    plt.plot(results_df["threshold"], results_df["false_negatives"], marker="o", label="False Negatives")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.title("False Positives / False Negatives vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(errors_plot_output)
    plt.close()

    with best_threshold_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_threshold": best_threshold,
                "roc_auc": roc_auc,
                "dataset_path": str(dataset_path),
                "results_output": str(results_output),
                "metrics_plot_output": str(metrics_plot_output),
                "errors_plot_output": str(errors_plot_output),
            },
            f,
            indent=2,
        )

    print("\nSaved files:")
    print(f"- {results_output}")
    print(f"- {metrics_plot_output}")
    print(f"- {errors_plot_output}")
    print(f"- {best_threshold_output}")


if __name__ == "__main__":
    main()