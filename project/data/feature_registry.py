from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "project" / "data" / "feature_registry_config.json"
SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR
DAYS_IN_WEEK = 7


@lru_cache(maxsize=8)
def _load_registry_config_cached(resolved_path: str) -> dict[str, Any]:
    with Path(resolved_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Feature registry config not found: {resolved}")
    return _load_registry_config_cached(str(resolved))


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index, dtype=float)
    series = pd.to_numeric(df[column], errors="coerce").fillna(default)
    return series.astype(float)


def _presence_signal(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    present = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
    for col in existing:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            non_empty = series.notna()
        else:
            as_text = series.astype("string")
            non_empty = as_text.notna() & (as_text.str.strip() != "")
        present = present | non_empty

    return present.astype(int)


def _derive_time_features(event_time_raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    safe_seconds = event_time_raw.fillna(0.0).astype(float)
    hour_of_day = ((safe_seconds % SECONDS_IN_DAY) // SECONDS_IN_HOUR).astype(int)
    day_of_week = ((safe_seconds // SECONDS_IN_DAY) % DAYS_IN_WEEK).astype(int)
    return hour_of_day, day_of_week


def _build_canonical_for_source(
    df: pd.DataFrame,
    source_cfg: dict[str, Any],
    defaults: dict[str, Any],
    canonical_keys: list[str],
) -> pd.DataFrame:
    amount_col = source_cfg.get("amount_column", "")
    time_col = source_cfg.get("event_time_column", "")

    amount_raw = _numeric_series(df, amount_col, default=float(defaults.get("amount_raw", 0.0)))
    amount_non_negative = amount_raw.clip(lower=0.0)
    amount_log = np.log1p(amount_non_negative)

    event_time_raw = _numeric_series(df, time_col, default=float(defaults.get("event_time_raw", 0.0)))
    hour_of_day, day_of_week = _derive_time_features(event_time_raw)

    device_signal_present = _presence_signal(df, source_cfg.get("device_columns", []))
    location_signal_present = _presence_signal(df, source_cfg.get("location_columns", []))

    canonical = pd.DataFrame(
        {
            "amount_raw": amount_raw.fillna(float(defaults.get("amount_raw", 0.0))).astype(float),
            "amount_log": pd.Series(amount_log, index=df.index).fillna(float(defaults.get("amount_log", 0.0))).astype(float),
            "event_time_raw": event_time_raw.fillna(float(defaults.get("event_time_raw", 0.0))).astype(float),
            "hour_of_day": hour_of_day.fillna(int(defaults.get("hour_of_day", 0))).astype(int),
            "day_of_week": day_of_week.fillna(int(defaults.get("day_of_week", 0))).astype(int),
            "device_signal_present": device_signal_present.fillna(int(defaults.get("device_signal_present", 0))).astype(int),
            "location_signal_present": location_signal_present.fillna(int(defaults.get("location_signal_present", 0))).astype(int),
        },
        index=df.index,
    )

    for key in canonical_keys:
        if key in canonical.columns:
            continue
        canonical[key] = _numeric_series(df, key, default=float(defaults.get(key, 0.0))).astype(float)

    return canonical


def _validate_required_columns(df: pd.DataFrame, source_cfg: dict[str, Any], dataset_source: str) -> None:
    required_columns = [str(col) for col in source_cfg.get("required_columns", [])]
    if not required_columns:
        return
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset source '{dataset_source}' requires columns {required_columns}, "
            f"but received missing columns {missing}."
        )


def _build_passthrough(df: pd.DataFrame, source_cfg: dict[str, Any]) -> pd.DataFrame:
    exclude = set(source_cfg.get("passthrough_exclude", []))
    passthrough_columns = [col for col in df.columns if col not in exclude]
    return df[passthrough_columns].copy()


def map_to_canonical_features(
    df: pd.DataFrame,
    dataset_source: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    usage_context: Literal["training", "runtime_serving"] = "training",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_registry_config(config_path)
    source_map = config.get("sources", {})
    if dataset_source not in source_map:
        raise ValueError(f"Unsupported dataset source for feature registry: {dataset_source}")

    source_cfg = source_map[dataset_source]
    if usage_context == "runtime_serving" and not bool(source_cfg.get("runtime_serving_enabled", True)):
        raise ValueError(
            f"Dataset source '{dataset_source}' is deprecated for runtime serving in feature registry config."
        )
    _validate_required_columns(df, source_cfg, dataset_source)
    defaults = config.get("defaults", {})
    canonical_keys = config.get("canonical_keys", [])

    canonical_df = _build_canonical_for_source(df, source_cfg, defaults, canonical_keys=canonical_keys)

    for key in canonical_keys:
        if key not in canonical_df.columns:
            canonical_df[key] = defaults.get(key, 0)

    canonical_df = canonical_df[canonical_keys].copy() if canonical_keys else canonical_df
    passthrough_df = _build_passthrough(df, source_cfg)

    return canonical_df, passthrough_df
