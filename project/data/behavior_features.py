from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from project.data.entity_aggregation import build_uid, compute_entity_rolling_aggregates

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "project" / "data" / "behavior_features_config.json"
BEHAVIOR_FEATURE_COLUMNS = [
    "tx_count_1h",
    "tx_count_24h",
    "avg_amount_24h",
    "amount_over_user_avg",
    "time_since_last_tx",
    "amount_std_24h",
    "uid_tx_count_7d",
    "uid_avg_amount_7d",
    "uid_amount_std_7d",
    "uid_time_since_last_tx",
]


@lru_cache(maxsize=8)
def _load_behavior_config_cached(resolved_path: str) -> dict[str, Any]:
    with Path(resolved_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_behavior_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Behavior feature config not found: {resolved}")
    return _load_behavior_config_cached(str(resolved))


def get_uid_candidates_for_source(
    dataset_source: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[str]:
    config = load_behavior_config(config_path)
    source_cfg = config.get("sources", {}).get(dataset_source)
    if source_cfg is None:
        raise ValueError(f"Unsupported dataset source for behavior features: {dataset_source}")
    candidates = source_cfg.get("uid_candidates", source_cfg.get("entity_candidates", []))
    return [str(col) for col in candidates]


def _safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0, clip_min: float | None = None) -> pd.Series:
    if col not in df.columns:
        out = pd.Series(np.full(len(df), default), index=df.index, dtype=float)
    else:
        out = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    if clip_min is not None:
        out = out.clip(lower=clip_min)
    return out


def _resolve_entity_id(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return entity id and boolean flag for rows using fallback/global entity."""
    if not candidates:
        entity = pd.Series(np.full(len(df), "GLOBAL", dtype=object), index=df.index)
        fallback = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
        return entity, fallback

    chosen = pd.Series(np.full(len(df), "", dtype=object), index=df.index)
    has_value = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col].astype("string")
        valid = series.notna() & (series.str.strip() != "")
        fill_mask = (~has_value) & valid
        chosen.loc[fill_mask] = series.loc[fill_mask].astype(str)
        has_value.loc[fill_mask] = True

    entity = pd.Series(np.where(has_value, chosen.astype(str), "GLOBAL"), index=df.index).astype(str)
    fallback = ~has_value
    return entity, fallback


def _rolling_window_stats(times: np.ndarray, amounts: np.ndarray, window_seconds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly prior-window stats: count, mean, std (ddof=0)."""
    n = len(times)
    counts = np.zeros(n, dtype=float)
    means = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)

    if n == 0:
        return counts, means, stds

    prefix_sum = np.zeros(n + 1, dtype=float)
    prefix_sq = np.zeros(n + 1, dtype=float)
    prefix_sum[1:] = np.cumsum(amounts)
    prefix_sq[1:] = np.cumsum(amounts ** 2)

    for i in range(n):
        t = times[i]
        left = np.searchsorted(times, t - window_seconds, side="left")
        right = i  # prior rows only -> exclude i itself
        c = max(0, right - left)
        counts[i] = float(c)
        if c > 0:
            window_sum = prefix_sum[right] - prefix_sum[left]
            window_sq = prefix_sq[right] - prefix_sq[left]
            mean = window_sum / c
            var = max((window_sq / c) - (mean ** 2), 0.0)
            means[i] = mean
            stds[i] = np.sqrt(var)

    return counts, means, stds


def build_behavior_features(
    features_df: pd.DataFrame,
    dataset_source: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    usage_context: Literal["training", "runtime_serving"] = "training",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_behavior_config(config_path)
    source_cfg = config.get("sources", {}).get(dataset_source)
    if source_cfg is None:
        raise ValueError(f"Unsupported dataset source for behavior features: {dataset_source}")
    if usage_context == "runtime_serving" and not bool(source_cfg.get("runtime_serving_enabled", True)):
        raise ValueError(
            f"Dataset source '{dataset_source}' is deprecated for runtime serving in behavior feature config."
        )

    defaults = config.get("defaults", {})
    windows = config.get("windows", {})
    include_std = bool(config.get("include_amount_std_24h", True))

    time_col = source_cfg.get("time_column", "")
    amount_col = source_cfg.get("amount_column", "")
    entity_candidates = source_cfg.get("entity_candidates", [])
    uid_candidates = source_cfg.get("uid_candidates", entity_candidates)
    uid_time_bucket_seconds = int(source_cfg.get("uid_time_bucket_seconds", 86400))

    event_time = _safe_numeric_series(features_df, time_col, default=0.0, clip_min=0.0)
    amount = _safe_numeric_series(features_df, amount_col, default=0.0, clip_min=0.0)
    entity_id, fallback_entity = _resolve_entity_id(features_df, entity_candidates)

    work = pd.DataFrame(
        {
            "__idx": np.arange(len(features_df)),
            "__time": event_time,
            "__amount": amount,
            "__entity": entity_id,
            "__fallback": fallback_entity,
        },
        index=features_df.index,
    )

    # Strict temporal ordering for no-leakage feature computation.
    work_sorted = work.sort_values(by=["__time", "__idx"], kind="mergesort").copy()

    out = pd.DataFrame(index=work_sorted.index)
    out["tx_count_1h"] = float(defaults.get("tx_count_1h", 0.0))
    out["tx_count_24h"] = float(defaults.get("tx_count_24h", 0.0))
    out["avg_amount_24h"] = float(defaults.get("avg_amount_24h", 0.0))
    out["amount_over_user_avg"] = float(defaults.get("amount_over_user_avg", 1.0))
    out["time_since_last_tx"] = float(defaults.get("time_since_last_tx", 86400.0))
    out["amount_std_24h"] = float(defaults.get("amount_std_24h", 0.0))
    out["uid_tx_count_7d"] = float(defaults.get("uid_tx_count_7d", 0.0))
    out["uid_avg_amount_7d"] = float(defaults.get("uid_avg_amount_7d", 0.0))
    out["uid_amount_std_7d"] = float(defaults.get("uid_amount_std_7d", 0.0))
    out["uid_time_since_last_tx"] = float(defaults.get("uid_time_since_last_tx", 86400.0))

    c1h_window = int(windows.get("count_1h_seconds", 3600))
    c24h_window = int(windows.get("count_24h_seconds", 86400))
    avg24h_window = int(windows.get("avg_24h_seconds", 86400))
    std24h_window = int(windows.get("std_24h_seconds", 86400))
    uid7d_window = int(windows.get("uid_7d_seconds", 7 * 86400))

    for _, grp in work_sorted.groupby("__entity", sort=False):
        times = grp["__time"].to_numpy(dtype=float)
        amounts = grp["__amount"].to_numpy(dtype=float)
        idx = grp.index

        count_1h, _, _ = _rolling_window_stats(times, amounts, c1h_window)
        count_24h, mean_24h_for_count, _ = _rolling_window_stats(times, amounts, c24h_window)
        _, mean_24h, _ = _rolling_window_stats(times, amounts, avg24h_window)
        _, _, std_24h = _rolling_window_stats(times, amounts, std24h_window)

        prev_times = np.concatenate(([np.nan], times[:-1]))
        time_since_last = times - prev_times
        default_delta = float(defaults.get("time_since_last_tx", 86400.0))
        time_since_last = np.where(np.isnan(time_since_last) | (time_since_last < 0), default_delta, time_since_last)

        expanding_prior_avg = pd.Series(amounts).expanding(min_periods=1).mean().shift(1).to_numpy()
        default_avg = float(defaults.get("avg_amount_24h", 0.0))
        avg_for_ratio = np.where(np.isnan(expanding_prior_avg) | (expanding_prior_avg <= 0.0), default_avg, expanding_prior_avg)
        default_ratio = float(defaults.get("amount_over_user_avg", 1.0))
        ratio = np.full(len(amounts), default_ratio, dtype=float)
        valid_ratio = avg_for_ratio > 0.0
        ratio[valid_ratio] = np.divide(amounts[valid_ratio], avg_for_ratio[valid_ratio])

        mean_24h = np.where(count_24h > 0, mean_24h_for_count, mean_24h)
        mean_24h = np.where(np.isnan(mean_24h), default_avg, mean_24h)

        out.loc[idx, "tx_count_1h"] = count_1h
        out.loc[idx, "tx_count_24h"] = count_24h
        out.loc[idx, "avg_amount_24h"] = mean_24h
        out.loc[idx, "amount_over_user_avg"] = np.where(np.isfinite(ratio), ratio, default_ratio)
        out.loc[idx, "time_since_last_tx"] = time_since_last
        out.loc[idx, "amount_std_24h"] = np.where(np.isfinite(std_24h), std_24h, float(defaults.get("amount_std_24h", 0.0)))

    uid_series = build_uid(
        features_df,
        time_col=time_col,
        uid_candidates=[str(c) for c in uid_candidates],
        time_bucket_seconds=uid_time_bucket_seconds,
    )
    uid_roll = compute_entity_rolling_aggregates(
        event_time=event_time,
        amount=amount,
        entity_id=uid_series,
        window_seconds=uid7d_window,
        default_recency=float(defaults.get("uid_time_since_last_tx", defaults.get("time_since_last_tx", 86400.0))),
    )
    out["uid_tx_count_7d"] = uid_roll["count"].to_numpy(dtype=float)
    out["uid_avg_amount_7d"] = uid_roll["mean"].to_numpy(dtype=float)
    out["uid_amount_std_7d"] = uid_roll["std"].to_numpy(dtype=float)
    out["uid_time_since_last_tx"] = uid_roll["recency"].to_numpy(dtype=float)

    # Restore original row order/index
    out = out.reindex(features_df.index)

    if not include_std:
        out = out.drop(columns=["amount_std_24h"])

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(float(defaults.get(col, 0.0)))

    diagnostics = {
        "dataset_source": dataset_source,
        "feature_columns": out.columns.tolist(),
        "null_rates": out.isna().mean().to_dict(),
        "distribution": {
            col: {
                "min": float(out[col].min()),
                "p50": float(out[col].quantile(0.5)),
                "p95": float(out[col].quantile(0.95)),
                "max": float(out[col].max()),
            }
            for col in out.columns
        },
        "source_coverage": {
            "rows": int(len(features_df)),
            "entity_fallback_rows": int(work["__fallback"].sum()),
            "entity_fallback_rate": float(work["__fallback"].mean()) if len(work) else 0.0,
            "entity_unique_count": int(work["__entity"].nunique(dropna=False)),
            "uid_unique_count": int(uid_series.nunique(dropna=False)),
        },
        "time_column": time_col,
        "amount_column": amount_col,
        "entity_candidates": entity_candidates,
        "uid_candidates": uid_candidates,
        "uid_time_bucket_seconds": uid_time_bucket_seconds,
    }

    return out, diagnostics
