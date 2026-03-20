from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Protocol, Any
import statistics
import threading
import time

from .domain_exceptions import UserProfileMismatchError
from .profile_store import InMemoryProfileStore


@dataclass
class TransactionRecord:
    user_id: str
    amount: float
    hour_of_day: int
    location_risk_score: float


@dataclass
class UserBehaviorProfile:
    user_id: str
    amounts: List[float] = field(default_factory=list)
    hours: List[int] = field(default_factory=list)
    location_risks: List[float] = field(default_factory=list)
    event_timestamps: List[float] = field(default_factory=list)
    geo_device_mismatch_flags: List[int] = field(default_factory=list)
    counterparties_24h: List[int] = field(default_factory=list)
    total_transactions: int = 0
    geo_device_mismatch_count: int = 0
    payload_schema_version: int = 2
    version: int = 0
    updated_at: float = 0.0

    def update(
        self,
        amount: float,
        hour_of_day: int,
        location_risk_score: float,
        event_timestamp: float = 0.0,
        geo_device_mismatch: bool = False,
        counterparties_24h: int = 0,
        history_limit: int = 100,
    ):
        self.amounts.append(float(amount))
        self.hours.append(int(hour_of_day))
        self.location_risks.append(float(location_risk_score))
        self.event_timestamps.append(float(event_timestamp))
        mismatch_flag = 1 if geo_device_mismatch else 0
        self.geo_device_mismatch_flags.append(mismatch_flag)
        self.counterparties_24h.append(max(0, int(counterparties_24h)))
        self.total_transactions += 1
        self.geo_device_mismatch_count += mismatch_flag

        self.amounts = self.amounts[-history_limit:]
        self.hours = self.hours[-history_limit:]
        self.location_risks = self.location_risks[-history_limit:]
        self.event_timestamps = self.event_timestamps[-history_limit:]
        self.geo_device_mismatch_flags = self.geo_device_mismatch_flags[-history_limit:]
        self.counterparties_24h = self.counterparties_24h[-history_limit:]

    def has_enough_history(self, min_history: int = 5) -> bool:
        return len(self.amounts) >= min_history

    def to_store_payload(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "amounts": self.amounts,
            "hours": self.hours,
            "location_risks": self.location_risks,
            "event_timestamps": self.event_timestamps,
            "geo_device_mismatch_flags": self.geo_device_mismatch_flags,
            "counterparties_24h": self.counterparties_24h,
            "total_transactions": self.total_transactions,
            "geo_device_mismatch_count": self.geo_device_mismatch_count,
            "payload_schema_version": self.payload_schema_version,
            "version": self.version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_store_payload(cls, payload: Dict[str, Any]) -> "UserBehaviorProfile":
        payload_user_id = str(payload.get("user_id", "")).strip()
        if not payload_user_id:
            raise UserProfileMismatchError("Persisted profile payload is missing user_id")

        payload_schema_version = int(payload.get("payload_schema_version", 1))
        if payload_schema_version < 1:
            raise UserProfileMismatchError(
                f"Persisted profile payload has invalid schema version: {payload_schema_version}"
            )

        return cls(
            user_id=payload_user_id,
            amounts=[float(x) for x in payload.get("amounts", [])],
            hours=[int(x) for x in payload.get("hours", [])],
            location_risks=[float(x) for x in payload.get("location_risks", [])],
            event_timestamps=[float(x) for x in payload.get("event_timestamps", [])],
            geo_device_mismatch_flags=[int(x) for x in payload.get("geo_device_mismatch_flags", [])],
            counterparties_24h=[int(x) for x in payload.get("counterparties_24h", [])],
            total_transactions=int(payload.get("total_transactions", len(payload.get("amounts", [])))),
            geo_device_mismatch_count=int(payload.get("geo_device_mismatch_count", 0)),
            payload_schema_version=payload_schema_version,
            version=int(payload.get("version", 0)),
            updated_at=float(payload.get("updated_at", 0.0)),
        )


class ProfileStore(Protocol):
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        ...

    def save_profile(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class BehaviorProfiler:
    def __init__(
        self,
        profile_store: Optional[ProfileStore] = None,
        history_limit: int = 100,
        min_history: int = 5,
        aggregate_cache_ttl_seconds: float = 2.0,
    ):
        self.profile_store = profile_store or InMemoryProfileStore()
        self.history_limit = history_limit
        self.min_history = min_history
        self.aggregate_cache_ttl_seconds = float(max(0.0, aggregate_cache_ttl_seconds))
        self._aggregate_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    def _cache_get(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.aggregate_cache_ttl_seconds <= 0:
            return None
        now = time.time()
        with self._cache_lock:
            item = self._aggregate_cache.get(user_id)
            if item is None:
                return None
            expires_at, payload = item
            if expires_at <= now:
                del self._aggregate_cache[user_id]
                return None
            return dict(payload)

    def _cache_set(self, user_id: str, payload: Dict[str, Any]) -> None:
        if self.aggregate_cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            self._aggregate_cache[user_id] = (
                time.time() + self.aggregate_cache_ttl_seconds,
                dict(payload),
            )

    def _cache_invalidate(self, user_id: str) -> None:
        with self._cache_lock:
            self._aggregate_cache.pop(user_id, None)

    def get_or_create_profile(self, user_id: str) -> UserBehaviorProfile:
        normalized_user_id = str(user_id).strip()
        if not normalized_user_id:
            raise UserProfileMismatchError("user_id must be a non-empty string")
        payload = self._cache_get(normalized_user_id)
        if payload is None:
            payload = self.profile_store.get_profile(normalized_user_id)
            if payload is not None:
                self._cache_set(normalized_user_id, payload)
        if payload is None:
            return UserBehaviorProfile(user_id=normalized_user_id)
        profile = UserBehaviorProfile.from_store_payload(payload)
        if profile.user_id != normalized_user_id:
            raise UserProfileMismatchError(
                f"Profile user_id mismatch: expected '{normalized_user_id}' observed '{profile.user_id}'"
            )
        return profile

    def save_profile(self, profile: UserBehaviorProfile) -> UserBehaviorProfile:
        self._cache_invalidate(profile.user_id)
        payload = self.profile_store.save_profile(profile.user_id, profile.to_store_payload())
        self._cache_set(profile.user_id, payload)
        return UserBehaviorProfile.from_store_payload(payload)

    def record_transaction(
        self,
        user_id: str,
        amount: float,
        hour_of_day: int,
        location_risk_score: float,
        event_timestamp: float = 0.0,
        geo_device_mismatch: bool = False,
        counterparties_24h: int = 0,
    ) -> UserBehaviorProfile:
        profile = self.get_or_create_profile(user_id)
        profile.update(
            amount=amount,
            hour_of_day=hour_of_day,
            location_risk_score=location_risk_score,
            event_timestamp=event_timestamp,
            geo_device_mismatch=geo_device_mismatch,
            counterparties_24h=counterparties_24h,
            history_limit=self.history_limit,
        )
        return self.save_profile(profile)

    def seed_profile(
        self,
        user_id: str,
        historical_transactions: List[Tuple[float, int, float]]
    ):
        """
        historical_transactions = [(amount, hour_of_day, location_risk_score), ...]
        """
        for amount, hour_of_day, location_risk_score in historical_transactions:
            self.record_transaction(user_id, amount, hour_of_day, location_risk_score)

    def compute_behavior_features(
        self,
        user_id: str,
        amount: float,
        hour_of_day: int,
        location_risk_score: float,
        event_timestamp: float = 0.0,
        geo_device_mismatch: bool = False,
        counterparties_24h: int = 0,
    ) -> Dict[str, float]:
        profile = self.get_or_create_profile(user_id)
        event_timestamp = float(event_timestamp)
        counterparties_24h = max(0, int(counterparties_24h))
        geo_device_mismatch = bool(geo_device_mismatch)

        if not profile.has_enough_history(self.min_history):
            history_count = len(profile.amounts)
            history_confidence = min(1.0, history_count / max(1, self.min_history))
            return {
                "amount_deviation_score": 0.10,
                "time_deviation_score": 0.10,
                "velocity_risk_score": 0.10,
                "location_deviation_score": location_risk_score,
                "behavior_risk_score": 0.10 + (0.20 * location_risk_score),
                "history_count": history_count,
                "is_low_history": True,
                "history_confidence": round(history_confidence, 4),
                "velocity_count_5m": 0.0,
                "velocity_count_1h": 0.0,
                "velocity_count_24h": 0.0,
                "geo_device_mismatch_rate": 0.0,
                "geo_device_mismatch_flag": float(geo_device_mismatch),
                "counterpart_diversity_avg_24h": 0.0,
                "counterpart_diversity_delta": float(counterparties_24h),
                "amount_anomaly_delta": 0.0,
                "location_anomaly_delta": 0.0,
            }

        median_amount = statistics.median(profile.amounts)
        if median_amount <= 0:
            median_amount = 1.0

        amount_ratio = amount / median_amount

        if amount_ratio < 1.5:
            amount_deviation_score = 0.05
        elif amount_ratio < 3:
            amount_deviation_score = 0.20
        elif amount_ratio < 6:
            amount_deviation_score = 0.45
        else:
            amount_deviation_score = 0.70

        avg_hour = round(statistics.mean(profile.hours))
        hour_diff = abs(hour_of_day - avg_hour)
        hour_diff = min(hour_diff, 24 - hour_diff)

        if hour_diff <= 2:
            time_deviation_score = 0.05
        elif hour_diff <= 5:
            time_deviation_score = 0.20
        elif hour_diff <= 8:
            time_deviation_score = 0.45
        else:
            time_deviation_score = 0.70

        recent_count = len(profile.amounts[-10:])
        avg_recent_amount = statistics.mean(profile.amounts[-10:]) if recent_count > 0 else 0.0

        if recent_count <= 3:
            velocity_risk_score = 0.05
        elif recent_count <= 7:
            velocity_risk_score = 0.20
        else:
            velocity_risk_score = 0.35

        if avg_recent_amount > 0 and amount > 3 * avg_recent_amount:
            velocity_risk_score = min(1.0, velocity_risk_score + 0.20)

        historical_avg_location_risk = statistics.mean(profile.location_risks) if profile.location_risks else 0.0
        location_deviation_score = min(
            1.0,
            max(location_risk_score, abs(location_risk_score - historical_avg_location_risk))
        )

        velocity_count_5m = 0.0
        velocity_count_1h = 0.0
        velocity_count_24h = 0.0
        if profile.event_timestamps and event_timestamp > 0:
            prior_times = [float(ts) for ts in profile.event_timestamps if float(ts) <= event_timestamp]
            velocity_count_5m = float(sum(1 for ts in prior_times if event_timestamp - ts <= 300))
            velocity_count_1h = float(sum(1 for ts in prior_times if event_timestamp - ts <= 3600))
            velocity_count_24h = float(sum(1 for ts in prior_times if event_timestamp - ts <= 86400))

        mismatch_count = float(sum(profile.geo_device_mismatch_flags))
        mismatch_rate = mismatch_count / max(1.0, float(len(profile.geo_device_mismatch_flags)))

        historical_counterpart_avg = (
            statistics.mean(profile.counterparties_24h) if profile.counterparties_24h else 0.0
        )
        counterpart_delta = float(counterparties_24h) - float(historical_counterpart_avg)
        amount_anomaly_delta = abs(amount - statistics.mean(profile.amounts))
        location_anomaly_delta = abs(location_risk_score - historical_avg_location_risk)

        behavior_risk_score = (
            0.35 * amount_deviation_score +
            0.25 * time_deviation_score +
            0.20 * velocity_risk_score +
            0.15 * location_deviation_score +
            0.05 * min(1.0, mismatch_rate)
        )

        return {
            "amount_deviation_score": round(amount_deviation_score, 4),
            "time_deviation_score": round(time_deviation_score, 4),
            "velocity_risk_score": round(velocity_risk_score, 4),
            "location_deviation_score": round(location_deviation_score, 4),
            "behavior_risk_score": round(behavior_risk_score, 4),
            "history_count": len(profile.amounts),
            "is_low_history": False,
            "history_confidence": 1.0,
            "velocity_count_5m": round(velocity_count_5m, 4),
            "velocity_count_1h": round(velocity_count_1h, 4),
            "velocity_count_24h": round(velocity_count_24h, 4),
            "geo_device_mismatch_rate": round(mismatch_rate, 4),
            "geo_device_mismatch_flag": float(geo_device_mismatch),
            "counterpart_diversity_avg_24h": round(historical_counterpart_avg, 4),
            "counterpart_diversity_delta": round(counterpart_delta, 4),
            "amount_anomaly_delta": round(amount_anomaly_delta, 4),
            "location_anomaly_delta": round(location_anomaly_delta, 4),
        }

    def generate_behavior_reasons(
        self,
        behavior_features: Dict[str, float]
    ) -> List[str]:
        reasons = []

        if behavior_features["amount_deviation_score"] >= 0.45:
            reasons.append("Transaction amount deviates strongly from normal user pattern")
        elif behavior_features["amount_deviation_score"] >= 0.20:
            reasons.append("Transaction amount is above usual user behavior")

        if behavior_features["time_deviation_score"] >= 0.45:
            reasons.append("Transaction time is highly unusual for this user")
        elif behavior_features["time_deviation_score"] >= 0.20:
            reasons.append("Transaction time is outside the user's normal pattern")

        if behavior_features["velocity_risk_score"] >= 0.35:
            reasons.append("Recent transaction activity suggests burst behavior")
        elif behavior_features["velocity_risk_score"] >= 0.20:
            reasons.append("Recent activity level is moderately elevated")

        if behavior_features["location_deviation_score"] >= 0.70:
            reasons.append("Transaction location risk is highly inconsistent with prior behavior")
        elif behavior_features["location_deviation_score"] >= 0.30:
            reasons.append("Transaction location risk is moderately elevated")

        if behavior_features.get("geo_device_mismatch_rate", 0.0) >= 0.40:
            reasons.append("Frequent geo/device mismatch pattern observed in recent history")

        if behavior_features.get("counterpart_diversity_delta", 0.0) >= 5.0:
            reasons.append("Counterparty diversity jumped above historical baseline")

        if not reasons:
            reasons.append("Behavior is consistent with the user's normal baseline")

        return reasons
