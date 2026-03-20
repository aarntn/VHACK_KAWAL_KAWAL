from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from project.data.behavior_features import BEHAVIOR_FEATURE_COLUMNS, build_behavior_features
from project.data.feature_registry import map_to_canonical_features


BASE_NUMERIC_CANONICAL_COLUMNS = [
    "amount_raw",
    "amount_log",
    "event_time_raw",
    "hour_of_day",
    "day_of_week",
    "device_signal_present",
    "location_signal_present",
]
NUMERIC_CANONICAL_COLUMNS = BASE_NUMERIC_CANONICAL_COLUMNS + BEHAVIOR_FEATURE_COLUMNS


def _drop_duplicate_columns(df: pd.DataFrame, keep: Literal["first", "last"] = "last") -> tuple[pd.DataFrame, list[str]]:
    duplicated_mask = df.columns.duplicated(keep=keep)
    duplicated_names = [str(col) for col in df.columns[duplicated_mask].tolist()]
    if not duplicated_names:
        return df, []
    return df.loc[:, ~duplicated_mask].copy(), duplicated_names


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency-encode each categorical column into [0,1] frequency values."""

    def __init__(self) -> None:
        self.frequency_maps_: dict[str, dict[str, float]] = {}
        self.columns_: list[object] = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        # Keep raw column keys (may be ints from upstream transformers like SimpleImputer).
        self.columns_ = list(df.columns)
        self.frequency_maps_ = {}
        for col in self.columns_:
            series = df[col].astype("string").fillna("<NA>")
            counts = series.value_counts(normalize=True)
            self.frequency_maps_[str(col)] = {str(k): float(v) for k, v in counts.items()}
        return self

    def transform(self, X):
        if not self.columns_:
            return np.empty((len(X), 0))
        df = pd.DataFrame(X).copy()
        transformed = []
        for col in self.columns_:
            series = df[col].astype("string").fillna("<NA>")
            mapping = self.frequency_maps_.get(str(col), {})
            transformed.append(series.map(mapping).fillna(0.0).astype(float).to_numpy())
        return np.column_stack(transformed) if transformed else np.empty((len(df), 0))

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{str(col)}__freq" for col in self.columns_], dtype=object)


@dataclass
class PreprocessingBundle:
    dataset_source: str
    include_passthrough: bool
    scaler: str
    categorical_encoding: str
    preprocessor: ColumnTransformer
    canonical_columns: list[str]
    passthrough_categorical_columns: list[str]
    feature_names_out: list[str]
    behavior_feature_diagnostics: dict[str, Any]
    bundle_version: str = "1.0"


def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out[NUMERIC_CANONICAL_COLUMNS].copy()


def _select_passthrough_categorical_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            cols.append(col)
    return cols


def prepare_preprocessing_inputs(
    features_df: pd.DataFrame,
    dataset_source: str,
    usage_context: Literal["training", "runtime_serving"] = "training",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    canonical_df, passthrough_df = map_to_canonical_features(
        features_df,
        dataset_source,
        usage_context=usage_context,
    )
    behavior_df, behavior_diagnostics = build_behavior_features(
        features_df,
        dataset_source,
        usage_context=usage_context,
    )
    canonical_plus_behavior = pd.concat([canonical_df, behavior_df], axis=1)
    canonical_plus_behavior, duplicate_columns = _drop_duplicate_columns(canonical_plus_behavior, keep="last")
    canonical_plus_behavior = _ensure_canonical_columns(canonical_plus_behavior)
    behavior_diagnostics = {
        **behavior_diagnostics,
        "duplicate_canonical_behavior_columns_dropped": duplicate_columns,
    }
    return canonical_plus_behavior, passthrough_df, behavior_diagnostics


def _build_categorical_pipeline(encoding: Literal["onehot", "frequency"]) -> Pipeline:
    if encoding == "frequency":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", FrequencyEncoder()),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(
    canonical_df: pd.DataFrame,
    passthrough_df: pd.DataFrame,
    include_passthrough: bool = True,
    scaler: Literal["standard", "robust"] = "standard",
    categorical_encoding: Literal["onehot", "frequency"] = "onehot",
) -> tuple[ColumnTransformer, list[str], list[str]]:
    scaler_step = StandardScaler() if scaler == "standard" else RobustScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_step),
        ]
    )

    passthrough_df = passthrough_df.loc[:, ~passthrough_df.columns.isin(canonical_df.columns)].copy()
    passthrough_cats = _select_passthrough_categorical_columns(passthrough_df) if include_passthrough else []

    transformers = [("numeric_canonical", numeric_pipeline, NUMERIC_CANONICAL_COLUMNS)]
    if passthrough_cats:
        cat_pipeline = _build_categorical_pipeline(categorical_encoding)
        transformers.append(("categorical_passthrough", cat_pipeline, passthrough_cats))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor, NUMERIC_CANONICAL_COLUMNS.copy(), passthrough_cats


def fit_preprocessing_bundle(
    canonical_df: pd.DataFrame,
    passthrough_df: pd.DataFrame,
    dataset_source: str,
    include_passthrough: bool = True,
    scaler: Literal["standard", "robust"] = "standard",
    categorical_encoding: Literal["onehot", "frequency"] = "onehot",
    behavior_feature_diagnostics: dict[str, Any] | None = None,
) -> tuple[PreprocessingBundle, np.ndarray]:
    canonical_df = _ensure_canonical_columns(canonical_df)
    passthrough_df = passthrough_df.loc[:, ~passthrough_df.columns.isin(canonical_df.columns)].copy()
    combined = pd.concat([canonical_df, passthrough_df], axis=1)

    preprocessor, canonical_columns, passthrough_categorical_columns = build_preprocessor(
        canonical_df=canonical_df,
        passthrough_df=passthrough_df,
        include_passthrough=include_passthrough,
        scaler=scaler,
        categorical_encoding=categorical_encoding,
    )

    transformed = preprocessor.fit_transform(combined)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    bundle = PreprocessingBundle(
        dataset_source=dataset_source,
        include_passthrough=include_passthrough,
        scaler=scaler,
        categorical_encoding=categorical_encoding,
        preprocessor=preprocessor,
        canonical_columns=canonical_columns,
        passthrough_categorical_columns=passthrough_categorical_columns,
        feature_names_out=feature_names,
        behavior_feature_diagnostics=behavior_feature_diagnostics or {},
    )
    return bundle, transformed


def transform_with_bundle(
    bundle: PreprocessingBundle,
    canonical_df: pd.DataFrame,
    passthrough_df: pd.DataFrame,
) -> np.ndarray:
    canonical_df = _ensure_canonical_columns(canonical_df)
    passthrough_df = passthrough_df.loc[:, ~passthrough_df.columns.isin(canonical_df.columns)].copy()
    combined = pd.concat([canonical_df, passthrough_df], axis=1)
    return bundle.preprocessor.transform(combined)


def transform_runtime_record_with_bundle(
    bundle: PreprocessingBundle,
    record: dict[str, Any],
) -> np.ndarray:
    """Runtime helper for a single serving record to keep row-level transforms shared/consistent."""
    row_df = pd.DataFrame([record])
    canonical_df, passthrough_df, _ = prepare_preprocessing_inputs(
        row_df,
        bundle.dataset_source,
        usage_context="runtime_serving",
    )
    return transform_with_bundle(bundle, canonical_df, passthrough_df)


def save_preprocessing_bundle(bundle: PreprocessingBundle, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_preprocessing_bundle(path: str | Path) -> PreprocessingBundle:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, PreprocessingBundle):
        raise TypeError(f"Invalid preprocessing bundle type: {type(obj)!r}")
    return obj
