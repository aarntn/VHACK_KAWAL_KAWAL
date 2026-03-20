from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EntitySmoothingConfig:
    method: str = "none"
    min_history: int = 2
    ema_alpha: float = 0.3
    blend_alpha: float = 0.5
    blend_cap: float = 0.25
    fallback_for_unseen: str = "raw"


def validate_smoothing_method(method: str) -> str:
    value = str(method).strip().lower()
    allowed = {"none", "mean", "ema", "blend"}
    if value not in allowed:
        raise ValueError(f"Unsupported smoothing method: {method}. Expected one of {sorted(allowed)}")
    return value


def smooth_mean(df: pd.DataFrame, entity_col: str = "entity_id", raw_col: str = "raw_score") -> np.ndarray:
    return df.groupby(entity_col)[raw_col].transform("mean").to_numpy(dtype=float)


def smooth_ema(df: pd.DataFrame, alpha: float, entity_col: str = "entity_id", raw_col: str = "raw_score") -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby(entity_col, sort=False).groups.items():
        series = df.loc[idx, raw_col]
        out_idx = df.index.get_indexer(idx)
        out[out_idx] = series.ewm(alpha=float(alpha), adjust=False).mean().to_numpy(dtype=float)
    return out


def smooth_capped_blend(raw: np.ndarray, aggregate: np.ndarray, alpha: float, cap: float) -> np.ndarray:
    blended = raw + float(alpha) * (aggregate - raw)
    if cap > 0:
        delta = np.clip(blended - raw, -float(cap), float(cap))
        blended = raw + delta
    return np.clip(blended, 0.0, 1.0)


def apply_entity_smoothing_batch(
    df: pd.DataFrame,
    method: str,
    *,
    entity_col: str = "entity_id",
    raw_col: str = "raw_score",
    min_history: int = 2,
    ema_alpha: float = 0.3,
    blend_alpha: float = 0.5,
    blend_cap: float = 0.25,
) -> np.ndarray:
    resolved = validate_smoothing_method(method)
    raw = df[raw_col].to_numpy(dtype=float)
    if resolved == "none":
        return raw

    counts = df.groupby(entity_col)[raw_col].transform("count").to_numpy(dtype=int)
    if resolved == "mean":
        aggregated = smooth_mean(df, entity_col=entity_col, raw_col=raw_col)
    elif resolved == "ema":
        aggregated = smooth_ema(df, alpha=ema_alpha, entity_col=entity_col, raw_col=raw_col)
    else:
        mean_scores = smooth_mean(df, entity_col=entity_col, raw_col=raw_col)
        aggregated = smooth_capped_blend(raw, mean_scores, alpha=blend_alpha, cap=blend_cap)

    return np.where(counts >= int(min_history), aggregated, raw).astype(float)


class EntitySmoothingState:
    def __init__(self, config: EntitySmoothingConfig):
        self.config = config
        self._state: dict[str, dict[str, float]] = {}

    def _unseen_value(self, raw_score: float) -> float:
        if self.config.fallback_for_unseen == "zero":
            return 0.0
        return float(raw_score)

    def smooth(self, entity_id: str, raw_score: float) -> tuple[float, dict[str, Any]]:
        method = validate_smoothing_method(self.config.method)
        raw = float(np.clip(raw_score, 0.0, 1.0))
        key = str(entity_id)
        state = self._state.get(key)

        used_fallback = state is None
        if state is None:
            smoothed = self._unseen_value(raw)
            prior_count = 0
        else:
            prior_count = int(state.get("count", 0))
            if method == "none":
                smoothed = raw
            elif prior_count < int(self.config.min_history):
                smoothed = raw
            elif method == "mean":
                smoothed = float(state.get("mean", raw))
            elif method == "ema":
                smoothed = float(state.get("ema", raw))
            else:
                mean_val = float(state.get("mean", raw))
                smoothed = float(smooth_capped_blend(np.array([raw]), np.array([mean_val]), self.config.blend_alpha, self.config.blend_cap)[0])

        if state is None:
            self._state[key] = {"count": 1.0, "mean": raw, "ema": raw}
        else:
            count = float(state.get("count", 0.0)) + 1.0
            prev_mean = float(state.get("mean", raw))
            prev_ema = float(state.get("ema", raw))
            state["count"] = count
            state["mean"] = prev_mean + (raw - prev_mean) / max(count, 1.0)
            state["ema"] = float(self.config.ema_alpha) * raw + (1.0 - float(self.config.ema_alpha)) * prev_ema

        smoothed = float(np.clip(smoothed, 0.0, 1.0))
        return smoothed, {
            "method": method,
            "entity_id": key,
            "prior_history_count": prior_count,
            "min_history": int(self.config.min_history),
            "fallback_used": bool(used_fallback or (prior_count < int(self.config.min_history) and method != "none")),
        }

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._state.items()}



def build_uid(
    df: pd.DataFrame,
    *,
    time_col: str,
    uid_candidates: list[str] | None = None,
    time_bucket_seconds: int = 86400,
) -> pd.Series:
    """Construct stable UID from candidate identity columns + coarse time bucket.

    UID format: `<identity_part>|t<bucket>` where identity_part concatenates available
    candidate columns (or GLOBAL if none available).
    """
    candidates = uid_candidates or []
    identity_parts: list[pd.Series] = []
    for col in candidates:
        if col in df.columns:
            part = df[col].astype("string").fillna("<NA>").str.strip()
            part = part.replace("", "<NA>")
            identity_parts.append(part)

    if identity_parts:
        identity = identity_parts[0].astype(str)
        for part in identity_parts[1:]:
            identity = identity + "|" + part.astype(str)
    else:
        identity = pd.Series("GLOBAL", index=df.index, dtype="string")

    if time_col in df.columns:
        event_time = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        event_time = pd.Series(np.arange(len(df), dtype=float), index=df.index)

    bucket_seconds = max(int(time_bucket_seconds), 1)
    bucket = (event_time // bucket_seconds).astype(int).astype(str)
    uid = identity.astype(str) + "|t" + bucket
    return uid.astype(str)


def compute_entity_rolling_aggregates(
    event_time: pd.Series,
    amount: pd.Series,
    entity_id: pd.Series,
    *,
    window_seconds: int,
    default_recency: float,
) -> pd.DataFrame:
    """Strictly prior rolling aggregates by entity with no future leakage."""
    work = pd.DataFrame(
        {
            "__idx": np.arange(len(event_time)),
            "__time": pd.to_numeric(event_time, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            "__amount": pd.to_numeric(amount, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            "__entity": entity_id.astype(str).to_numpy(dtype=str),
        },
        index=event_time.index,
    ).sort_values(["__entity", "__time", "__idx"], kind="mergesort")

    result_count = np.zeros(len(work), dtype=float)
    result_mean = np.zeros(len(work), dtype=float)
    result_std = np.zeros(len(work), dtype=float)
    result_recency = np.full(len(work), float(default_recency), dtype=float)

    for _, grp in work.groupby("__entity", sort=False):
        times = grp["__time"].to_numpy(dtype=float)
        amounts = grp["__amount"].to_numpy(dtype=float)
        row_positions = grp["__idx"].to_numpy(dtype=int)

        n = len(times)
        counts = np.zeros(n, dtype=float)
        means = np.zeros(n, dtype=float)
        stds = np.zeros(n, dtype=float)

        prefix_sum = np.zeros(n + 1, dtype=float)
        prefix_sq = np.zeros(n + 1, dtype=float)
        prefix_sum[1:] = np.cumsum(amounts)
        prefix_sq[1:] = np.cumsum(amounts ** 2)

        for i in range(n):
            left = np.searchsorted(times, times[i] - int(window_seconds), side="left")
            right = i
            c = max(0, right - left)
            counts[i] = float(c)
            if c > 0:
                s = prefix_sum[right] - prefix_sum[left]
                ss = prefix_sq[right] - prefix_sq[left]
                m = s / c
                v = max((ss / c) - (m ** 2), 0.0)
                means[i] = m
                stds[i] = np.sqrt(v)

        prev_times = np.concatenate(([np.nan], times[:-1]))
        recency = times - prev_times
        recency = np.where(np.isnan(recency) | (recency < 0), float(default_recency), recency)

        result_count[row_positions] = counts
        result_mean[row_positions] = means
        result_std[row_positions] = stds
        result_recency[row_positions] = recency

    return pd.DataFrame(
        {
            "count": result_count,
            "mean": result_mean,
            "std": result_std,
            "recency": result_recency,
        },
        index=event_time.index,
    )
