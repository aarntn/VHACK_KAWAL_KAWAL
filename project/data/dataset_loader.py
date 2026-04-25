from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from project.data.entity_identity import build_entity_id
from project.data.behavior_features import build_behavior_features
from project.data.feature_registry import map_to_canonical_features


DatasetPayload = tuple[pd.DataFrame, pd.Series, dict[str, Any]]

HIGH_CARDINALITY_THRESHOLD = 1000
MIN_TARGET_PREVALENCE = 0.001
SUPPORTED_DATASET_SOURCES = {"creditcard", "ieee_cis"}


def build_ieee_eda_gate(tx_df: pd.DataFrame, id_df: pd.DataFrame, merged_df: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    join_row_ok = len(merged_df) == len(tx_df)

    key_missing_tx = float(tx_df["TransactionID"].isna().mean())
    key_missing_id = float(id_df["TransactionID"].isna().mean())
    key_missing_merged = float(merged_df["TransactionID"].isna().mean())
    key_missing_ok = max(key_missing_tx, key_missing_id, key_missing_merged) == 0.0

    labels_num = pd.to_numeric(labels, errors="coerce")
    labeled = labels_num.dropna()
    target_prevalence = float((labeled == 1).mean()) if len(labeled) else 0.0
    target_prevalence_ok = 0.0 < target_prevalence < (1.0 - MIN_TARGET_PREVALENCE)

    high_cardinality_fields: list[str] = []
    for col in merged_df.columns:
        series = merged_df[col]
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            if int(series.nunique(dropna=True)) > HIGH_CARDINALITY_THRESHOLD:
                high_cardinality_fields.append(col)
    high_cardinality_ok = True

    tx_time = pd.to_numeric(tx_df.get("TransactionDT"), errors="coerce") if "TransactionDT" in tx_df.columns else pd.Series(dtype=float)
    ts_min = float(tx_time.min()) if len(tx_time.dropna()) else None
    ts_max = float(tx_time.max()) if len(tx_time.dropna()) else None
    timestamp_range_ok = ts_min is not None and ts_max is not None and ts_min >= 0 and ts_max > ts_min

    checks = {
        "join_row_counts": {
            "passed": bool(join_row_ok),
            "transaction_rows": int(len(tx_df)),
            "identity_rows": int(len(id_df)),
            "merged_rows": int(len(merged_df)),
        },
        "key_missingness": {
            "passed": bool(key_missing_ok),
            "transaction_missing_ratio": key_missing_tx,
            "identity_missing_ratio": key_missing_id,
            "merged_missing_ratio": key_missing_merged,
        },
        "target_prevalence": {
            "passed": bool(target_prevalence_ok),
            "positive_rate": target_prevalence,
            "minimum_expected_positive_rate": MIN_TARGET_PREVALENCE,
        },
        "high_cardinality_fields": {
            "passed": bool(high_cardinality_ok),
            "count": int(len(high_cardinality_fields)),
            "sample_fields": sorted(high_cardinality_fields)[:25],
            "threshold": HIGH_CARDINALITY_THRESHOLD,
        },
        "timestamp_range_sanity": {
            "passed": bool(timestamp_range_ok),
            "timestamp_column": "TransactionDT",
            "min": ts_min,
            "max": ts_max,
        },
    }
    failures = [name for name, payload in checks.items() if not bool(payload.get("passed"))]
    return {
        "enabled": True,
        "passed": len(failures) == 0,
        "failed_checks": failures,
        "checks": checks,
    }


def enforce_eda_gate(metadata: dict[str, Any], *, allow_failures: bool, context: str) -> None:
    gate = metadata.get("eda_gate")
    if not isinstance(gate, dict):
        return
    if gate.get("passed", True):
        return
    failed = gate.get("failed_checks", [])
    failed_label = ", ".join(str(item) for item in failed) if failed else "unknown"
    if allow_failures:
        print(f"WARNING: EDA gate failed in {context}: {failed_label}")
        return
    raise RuntimeError(
        f"EDA gate failed in {context}: {failed_label}. "
        "Use --allow-eda-failures to continue at your own risk."
    )


def _compute_metadata(df: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    missingness = df.isna().mean().to_dict()
    class_counts = labels.value_counts(dropna=False).to_dict()
    total = int(len(labels))
    class_balance = {
        str(label): {
            "count": int(count),
            "rate": float(count / total) if total else 0.0,
        }
        for label, count in class_counts.items()
    }
    return {
        "row_count": int(len(df)),
        "missingness": missingness,
        "class_balance": class_balance,
    }


def _resolve_event_time(features: pd.DataFrame, dataset_source: str) -> pd.Series:
    if dataset_source not in SUPPORTED_DATASET_SOURCES:
        raise ValueError(f"Unsupported dataset_source: {dataset_source}")
    col = "TransactionDT" if dataset_source == "ieee_cis" else "Time"
    if col in features.columns:
        return pd.to_numeric(features[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(range(len(features)), index=features.index, dtype=float)


def apply_label_policy(
    features: pd.DataFrame,
    labels: pd.Series,
    dataset_source: str,
    label_policy: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if dataset_source not in SUPPORTED_DATASET_SOURCES:
        raise ValueError(f"Unsupported dataset_source: {dataset_source}")
    policy = str(label_policy).strip().lower()
    if policy not in {"transaction", "account_propagated"}:
        raise ValueError(f"Unsupported label_policy: {label_policy}")

    labels_num = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int).clip(lower=0, upper=1)
    if policy == "transaction":
        return labels_num, {
            "label_policy": policy,
            "rows_changed": 0,
            "positive_before": int(labels_num.sum()),
            "positive_after": int(labels_num.sum()),
        }

    event_time = _resolve_event_time(features, dataset_source)
    entity_id, entity_diag = build_entity_id(features, dataset_source=dataset_source)

    work = pd.DataFrame(
        {
            "label": labels_num.to_numpy(dtype=int),
            "entity_id": entity_id.to_numpy(dtype=str),
            "event_time": event_time.to_numpy(dtype=float),
        },
        index=features.index,
    ).sort_values(["entity_id", "event_time"], kind="mergesort")

    propagated = work.groupby("entity_id", sort=False)["label"].cummax().astype(int)
    work["label_propagated"] = propagated
    out = work.sort_index()["label_propagated"].astype(int)

    changed = int((out != labels_num).sum())
    return out, {
        "label_policy": policy,
        "rows_changed": changed,
        "positive_before": int(labels_num.sum()),
        "positive_after": int(out.sum()),
        "entity_diagnostics": entity_diag,
    }


def _format_missing_file_error(path: Path, label: str) -> str:
    cwd = Path.cwd().resolve()
    return (
        f"{label} file not found: {path}\n"
        f"- Current working directory: {cwd}\n"
        "- Verify the path is correct and points to an existing file."
    )


def _require_existing_file(path: str | Path, label: str) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(_format_missing_file_error(resolved_path, label))
    return resolved_path




def _augment_with_behavior_features(features: pd.DataFrame, dataset_source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    behavior_df, behavior_diag = build_behavior_features(features, dataset_source=dataset_source)
    overlap = [col for col in behavior_df.columns if col in features.columns]
    if overlap:
        raise ValueError(f"Behavior feature collision with existing columns: {overlap}")
    augmented = pd.concat([features, behavior_df], axis=1)
    return augmented, behavior_diag

def load_creditcard(path: str | Path, label_policy: str = "transaction") -> DatasetPayload:
    dataset_path = _require_existing_file(path, "Dataset")

    df = pd.read_csv(dataset_path)
    if "Class" not in df.columns:
        raise ValueError("creditcard dataset must contain a 'Class' column.")

    labels_raw = df["Class"].copy()
    features_raw = df.drop(columns=["Class"])
    labels, policy_meta = apply_label_policy(features_raw, labels_raw, dataset_source="creditcard", label_policy=label_policy)
    features, behavior_diag = _augment_with_behavior_features(features_raw, dataset_source="creditcard")
    metadata = _compute_metadata(features, labels)
    canonical_df, passthrough_df = map_to_canonical_features(features, "creditcard")
    metadata.update(
        {
            "dataset_source": "creditcard",
            "dataset_path": str(dataset_path),
            "label_policy": policy_meta["label_policy"],
            "label_policy_diagnostics": policy_meta,
            "canonical_feature_columns": canonical_df.columns.tolist(),
            "passthrough_columns": passthrough_df.columns.tolist(),
            "canonical_feature_preview": canonical_df.head(5).to_dict(orient="records"),
            "behavior_feature_columns": behavior_diag.get("feature_columns", []),
            "behavior_feature_diagnostics": behavior_diag,
        }
    )

    return features, labels, metadata


def load_ieee_cis(
    transaction_path: str | Path,
    identity_path: str | Path,
    label_col: str = "isFraud",
    label_policy: str = "transaction",
) -> DatasetPayload:
    tx_path = _require_existing_file(transaction_path, "Transaction")
    id_path = _require_existing_file(identity_path, "Identity")

    tx_df = pd.read_csv(tx_path)
    id_df = pd.read_csv(id_path)

    if "TransactionID" not in tx_df.columns or "TransactionID" not in id_df.columns:
        raise ValueError("IEEE-CIS transaction and identity datasets must include 'TransactionID'.")
    if label_col not in tx_df.columns:
        raise ValueError(f"IEEE-CIS transaction dataset must include '{label_col}'.")

    merged = tx_df.merge(id_df, how="left", on="TransactionID", suffixes=("", "_identity"))

    labels_raw = merged[label_col].copy()
    features_raw = merged.drop(columns=[label_col])
    labels, policy_meta = apply_label_policy(features_raw, labels_raw, dataset_source="ieee_cis", label_policy=label_policy)
    features, behavior_diag = _augment_with_behavior_features(features_raw, dataset_source="ieee_cis")
    eda_gate = build_ieee_eda_gate(tx_df, id_df, merged, labels_raw)

    metadata = _compute_metadata(features, labels)
    canonical_df, passthrough_df = map_to_canonical_features(features, "ieee_cis")
    metadata.update(
        {
            "dataset_source": "ieee_cis",
            "transaction_path": str(tx_path),
            "identity_path": str(id_path),
            "transaction_row_count": int(len(tx_df)),
            "identity_row_count": int(len(id_df)),
            "merged_row_count": int(len(merged)),
            "join_type": "left",
            "join_key": "TransactionID",
            "label_col": label_col,
            "label_policy": policy_meta["label_policy"],
            "label_policy_diagnostics": policy_meta,
            "canonical_feature_columns": canonical_df.columns.tolist(),
            "passthrough_columns": passthrough_df.columns.tolist(),
            "canonical_feature_preview": canonical_df.head(5).to_dict(orient="records"),
            "behavior_feature_columns": behavior_diag.get("feature_columns", []),
            "behavior_feature_diagnostics": behavior_diag,
            "eda_gate": eda_gate,
        }
    )

    return features, labels, metadata


def load_combined(*_: Any, **__: Any) -> DatasetPayload:
    raise NotImplementedError(
        "Dataset source 'combined' is a placeholder for future integration and is not implemented yet."
    )
