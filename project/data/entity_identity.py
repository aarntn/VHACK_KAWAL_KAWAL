from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EntityTier:
    name: str
    columns: tuple[str, ...]


CREDITCARD_TIERS: tuple[EntityTier, ...] = (
    EntityTier("credit_user_device", ("card1", "card2", "card3", "card5", "DeviceType", "DeviceInfo", "P_emaildomain")),
    EntityTier("credit_user_proxy", ("Amount", "Time", "V1", "V2", "V3")),
)

IEEE_TIERS: tuple[EntityTier, ...] = (
    EntityTier("ieee_card_addr_email_device", ("card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo")),
    EntityTier("ieee_card_addr_email", ("card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain")),
    EntityTier("ieee_card_only", ("card1", "card2", "card3", "card5")),
)


def _normalize_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).strip().lower()


def _hash_key(parts: list[str]) -> str:
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _tier_series(features: pd.DataFrame, tier: EntityTier) -> pd.Series:
    available_cols = [col for col in tier.columns if col in features.columns]
    if not available_cols:
        return pd.Series([""] * len(features), index=features.index)

    rows: list[str] = []
    view = features[available_cols]
    for _, row in view.iterrows():
        parts: list[str] = []
        for col in available_cols:
            normalized = _normalize_value(row[col])
            if normalized:
                parts.append(f"{col}={normalized}")
        rows.append(_hash_key(parts) if parts else "")

    return pd.Series(rows, index=features.index)


def build_entity_id(features: pd.DataFrame, dataset_source: str) -> tuple[pd.Series, dict[str, Any]]:
    tiers = CREDITCARD_TIERS if dataset_source == "creditcard" else IEEE_TIERS

    entity_series = pd.Series([""] * len(features), index=features.index, dtype="object")
    diagnostics: dict[str, Any] = {"dataset_source": dataset_source, "tiers": []}

    unresolved = entity_series == ""
    for tier in tiers:
        tier_key = _tier_series(features.loc[unresolved], tier)
        mask = tier_key != ""
        assigned = int(mask.sum())
        if assigned > 0:
            entity_series.loc[tier_key.index[mask]] = f"{tier.name}:" + tier_key.loc[mask]
        unresolved = entity_series == ""
        diagnostics["tiers"].append(
            {
                "tier": tier.name,
                "configured_columns": list(tier.columns),
                "available_columns": [c for c in tier.columns if c in features.columns],
                "assigned_rows": assigned,
                "remaining_rows": int(unresolved.sum()),
            }
        )

    if unresolved.any():
        fallback_ids = [f"fallback:{idx}" for idx in features.index[unresolved]]
        entity_series.loc[unresolved] = fallback_ids

    diagnostics.update(
        {
            "rows_total": int(len(features)),
            "unique_entities": int(entity_series.nunique(dropna=False)),
            "fallback_rows": int((entity_series.astype(str).str.startswith("fallback:")).sum()),
        }
    )

    return entity_series.astype(str), diagnostics
