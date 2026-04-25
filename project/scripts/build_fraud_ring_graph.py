"""
Build fraud ring graph features from backend events.

Ingests event data with columns/keys:
  - user_id
  - device_id
  - ip_subnet
  - card_prefix
  - timestamp

Builds rolling-window graphs (1d, 7d, 30d by default) and persists a
versioned per-account feature artifact for inference.

Outputs (under --output-dir):
  monitoring/fraud_ring_summary.json                      - run summary
  monitoring/fraud_ring_reports_<window>.json             - ring reports per window
  monitoring/fraud_ring_reports.json                      - canonical primary-window ring reports
  monitoring/fraud_ring_evidence_links_<window>.json      - exact account↔attribute evidence links
  monitoring/fraud_ring_evidence_links.json               - canonical primary-window evidence links
  monitoring/fraud_ring_attribute_index_<window>.json     - per-attribute risk index
  monitoring/fraud_ring_attribute_index.json              - canonical primary-window attribute index
  monitoring/fraud_ring_features_latest.json              - latest pointer artifact
  monitoring/fraud_ring_features_<version>.json           - immutable, versioned artifact
  monitoring/fraud_ring_features_manifest.json            - metadata + latest artifact path
  monitoring/fraud_ring_scores.json                       - compatibility map (30d ring_score)

Usage:
  python project/scripts/build_fraud_ring_graph.py --events-path path/to/events.jsonl
  python project/scripts/build_fraud_ring_graph.py --events-path path/to/events.csv --windows 1d 7d 30d
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.data.fraud_ring_graph import FraudRingGraph

OUTPUT_DIR = REPO_ROOT / "project" / "outputs"
MONITORING_DIRNAME = "monitoring"


@dataclass
class BackendEvent:
    user_id: str
    device_id: str | None
    ip_subnet: str | None
    card_prefix: str | None
    timestamp: datetime
    is_fraud: bool | None = None


def _parse_timestamp(raw: Any) -> datetime:
    if raw is None:
        raise ValueError("timestamp is required")
    ts = str(raw).strip()
    if not ts:
        raise ValueError("timestamp is empty")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_optional_bool(value: Any, *, source: str, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value for {field_name} in {source}: {value}")


def _coerce_event(record: dict[str, Any], source: str, label_field: str | None) -> BackendEvent:
    user_id = _normalize_optional(record.get("user_id"))
    if not user_id:
        raise ValueError(f"missing user_id in {source}")
    return BackendEvent(
        user_id=user_id,
        device_id=_normalize_optional(record.get("device_id")),
        ip_subnet=_normalize_optional(record.get("ip_subnet")),
        card_prefix=_normalize_optional(record.get("card_prefix")),
        timestamp=_parse_timestamp(record.get("timestamp")),
        is_fraud=_parse_optional_bool(record.get(label_field), source=source, field_name=label_field)
        if label_field
        else None,
    )


def load_backend_events(path: Path, *, label_field: str | None) -> list[BackendEvent]:
    if not path.exists():
        raise FileNotFoundError(f"events file not found: {path}")

    events: list[BackendEvent] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                    events.append(_coerce_event(record, f"{path}:{idx}", label_field))
                except Exception as exc:  # pragma: no cover (defensive parse error context)
                    raise ValueError(f"invalid JSONL event at {path}:{idx}: {exc}") from exc
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 2):
                events.append(_coerce_event(row, f"{path}:{idx}", label_field))
    else:
        raise ValueError("--events-path must point to a .jsonl or .csv file")

    if not events:
        raise ValueError(f"no valid events found in {path}")

    events.sort(key=lambda e: e.timestamp)
    return events


def _parse_window_spec(spec: str) -> tuple[str, timedelta]:
    raw = spec.strip().lower()
    if raw.endswith("d"):
        return raw, timedelta(days=int(raw[:-1]))
    if raw.endswith("h"):
        return raw, timedelta(hours=int(raw[:-1]))
    raise ValueError(f"invalid window '{spec}'. Use formats like 1d, 7d, 30d, 12h")


def _events_in_window(events: Iterable[BackendEvent], start_ts: datetime) -> list[BackendEvent]:
    return [event for event in events if event.timestamp >= start_ts]


def _compute_shared_attribute_counts(events: list[BackendEvent]) -> dict[str, dict[str, int]]:
    """Per-account counts of attributes also seen on at least one other account."""
    attr_to_users: dict[str, set[str]] = defaultdict(set)
    user_to_attrs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for event in events:
        for attr_type, value in (
            ("device", event.device_id),
            ("ip", event.ip_subnet),
            ("card", event.card_prefix),
        ):
            if not value:
                continue
            token = f"{attr_type}:{value}"
            attr_to_users[token].add(event.user_id)
            user_to_attrs[event.user_id][attr_type].add(token)

    features: dict[str, dict[str, int]] = {}
    for user_id, grouped_attrs in user_to_attrs.items():
        shared_device = sum(1 for token in grouped_attrs.get("device", set()) if len(attr_to_users[token]) >= 2)
        shared_ip = sum(1 for token in grouped_attrs.get("ip", set()) if len(attr_to_users[token]) >= 2)
        shared_card = sum(1 for token in grouped_attrs.get("card", set()) if len(attr_to_users[token]) >= 2)
        features[user_id] = {
            "shared_device_count": shared_device,
            "shared_ip_count": shared_ip,
            "shared_card_count": shared_card,
            "shared_attribute_count": shared_device + shared_ip + shared_card,
        }
    return features


def _component_sizes_by_account(graph: FraudRingGraph) -> dict[str, int]:
    component_sizes: dict[str, int] = {}
    for report in graph.get_ring_reports():
        for user_id in report.member_accounts:
            component_sizes[user_id] = report.ring_size
    return component_sizes


def _resolve_label_mode(events: list[BackendEvent], requested_mode: str) -> str:
    labeled_count = sum(1 for event in events if event.is_fraud is not None)
    if requested_mode == "topology_only":
        return "topology_only"
    if requested_mode == "labeled":
        if labeled_count != len(events):
            raise ValueError(
                "label_mode=labeled requires every event to include the configured label field"
            )
        return "labeled"
    return "labeled" if labeled_count == len(events) and events else "topology_only"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rolling-window fraud ring features from backend events")
    p.add_argument("--events-path", type=Path, required=True,
                   help="Input backend events file (.jsonl or .csv) with user_id/device_id/ip_subnet/card_prefix/timestamp")
    p.add_argument(
        "--label-field",
        default="is_fraud",
        help="Optional fraud-label field on each event (default: is_fraud). Set empty string to ignore labels.",
    )
    p.add_argument(
        "--label-mode",
        choices=["auto", "labeled", "topology_only"],
        default="auto",
        help="Whether ring artifacts may use fraud labels for scoring (default: auto).",
    )
    p.add_argument("--windows", nargs="+", default=["1d", "7d", "30d"],
                   help="Rolling windows to build (examples: 1d 7d 30d)")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--high-risk-threshold", type=float, default=0.5)
    p.add_argument("--min-ring-size", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    monitoring_dir = args.output_dir / MONITORING_DIRNAME
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    label_field = str(args.label_field).strip() or None
    events = load_backend_events(args.events_path, label_field=label_field)
    latest_ts = events[-1].timestamp
    windows = [_parse_window_spec(spec) for spec in args.windows]
    effective_label_mode = _resolve_label_mode(events, args.label_mode)

    generated_at = datetime.now(timezone.utc)
    version = generated_at.strftime("%Y%m%dT%H%M%SZ")

    per_account_features: dict[str, dict[str, Any]] = defaultdict(dict)
    window_summaries: dict[str, dict[str, Any]] = {}
    account_union: set[str] = set()
    report_paths: dict[str, str] = {}
    evidence_link_paths: dict[str, str] = {}
    attribute_index_paths: dict[str, str] = {}

    print(f"Loaded {len(events)} backend events from {args.events_path}")
    print(
        f"Building rolling windows anchored at {latest_ts.isoformat()} "
        f"(label_mode={effective_label_mode}, label_field={label_field or 'disabled'}) ..."
    )

    for window_name, delta in windows:
        window_start = latest_ts - delta
        window_events = _events_in_window(events, window_start)
        _weights_file = monitoring_dir / "ring_weight_model.json"
        graph = FraudRingGraph(
            min_ring_size=args.min_ring_size,
            high_risk_threshold=args.high_risk_threshold,
            label_mode=effective_label_mode,
            weights_path=str(_weights_file) if _weights_file.exists() else None,
        )
        for event in window_events:
            graph.add_transaction(
                event.user_id,
                device_id=event.device_id,
                ip_subnet=event.ip_subnet,
                card_prefix=event.card_prefix,
                is_fraud=bool(event.is_fraud) if effective_label_mode == "labeled" else False,
                timestamp=event.timestamp,
            )

        summary = graph.build()
        ring_reports = graph.get_ring_reports()
        ring_reports.sort(key=lambda r: r.ring_score, reverse=True)
        ring_report_rows = [
            {
                **r.to_dict(),
                "window": window_name,
                "label_field": label_field if effective_label_mode == "labeled" else None,
            }
            for r in ring_reports
        ]
        evidence_links = [item.to_dict() for item in graph.get_evidence_links(window=window_name)]
        attribute_index = graph.get_attribute_risk_index(window=window_name)
        reports_path = monitoring_dir / f"fraud_ring_reports_{window_name}.json"
        evidence_links_path = monitoring_dir / f"fraud_ring_evidence_links_{window_name}.json"
        attribute_index_path = monitoring_dir / f"fraud_ring_attribute_index_{window_name}.json"
        reports_path.write_text(json.dumps(ring_report_rows, indent=2), encoding="utf-8")
        evidence_links_path.write_text(json.dumps(evidence_links, indent=2), encoding="utf-8")
        attribute_index_path.write_text(json.dumps(attribute_index, indent=2), encoding="utf-8")
        report_paths[window_name] = reports_path.name
        evidence_link_paths[window_name] = evidence_links_path.name
        attribute_index_paths[window_name] = attribute_index_path.name

        shared_counts = _compute_shared_attribute_counts(window_events)
        component_sizes = _component_sizes_by_account(graph)
        window_scores = graph.dump_scores()

        users_in_window = {event.user_id for event in window_events}
        account_union.update(users_in_window)

        for user_id in users_in_window:
            base_counts = shared_counts.get(
                user_id,
                {
                    "shared_device_count": 0,
                    "shared_ip_count": 0,
                    "shared_card_count": 0,
                    "shared_attribute_count": 0,
                },
            )
            per_account_features[user_id][window_name] = {
                "ring_score": round(float(window_scores.get(user_id, 0.0)), 4),
                "component_size": int(component_sizes.get(user_id, 1)),
                **base_counts,
            }

        window_summaries[window_name] = {
            "window_start_utc": window_start.isoformat(),
            "window_end_utc": latest_ts.isoformat(),
            "event_count": len(window_events),
            "unique_accounts": len(users_in_window),
            "label_mode": effective_label_mode,
            "label_field": label_field if effective_label_mode == "labeled" else None,
            "graph_stats": summary.to_dict(),
            "high_risk_rings_ge_threshold": sum(
                1 for report in ring_reports if report.ring_score >= args.high_risk_threshold
            ),
            "reports_artifact": reports_path.name,
            "evidence_links_artifact": evidence_links_path.name,
            "attribute_index_artifact": attribute_index_path.name,
        }

        print(
            f"  [{window_name}] events={len(window_events)} accounts={len(users_in_window)} "
            f"rings={summary.rings_detected} high_risk={window_summaries[window_name]['high_risk_rings_ge_threshold']}"
        )

    ordered_windows = [name for name, _ in windows]
    artifact = {
        "artifact_type": "fraud_ring_account_features",
        "artifact_version": version,
        "generated_at_utc": generated_at.isoformat(),
        "label_mode": effective_label_mode,
        "label_field": label_field if effective_label_mode == "labeled" else None,
        "source": {
            "events_path": str(args.events_path),
            "events_loaded": len(events),
            "anchor_timestamp_utc": latest_ts.isoformat(),
        },
        "windows": ordered_windows,
        "window_specs": ordered_windows,
        "report_artifacts": report_paths,
        "evidence_link_artifacts": evidence_link_paths,
        "attribute_index_artifacts": attribute_index_paths,
        "window_summaries": window_summaries,
        "features_by_account": {user_id: per_account_features.get(user_id, {}) for user_id in sorted(account_union)},
    }

    versioned_path = monitoring_dir / f"fraud_ring_features_{version}.json"
    latest_path = monitoring_dir / "fraud_ring_features_latest.json"
    manifest_path = monitoring_dir / "fraud_ring_features_manifest.json"
    scores_compat_path = monitoring_dir / "fraud_ring_scores.json"
    reports_compat_path = monitoring_dir / "fraud_ring_reports.json"
    evidence_links_compat_path = monitoring_dir / "fraud_ring_evidence_links.json"
    attribute_index_compat_path = monitoring_dir / "fraud_ring_attribute_index.json"
    summary_path = monitoring_dir / "fraud_ring_summary.json"

    versioned_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    shutil.copyfile(versioned_path, latest_path)

    manifest = {
        "artifact_type": "fraud_ring_account_features",
        "latest_version": version,
        "latest_artifact": latest_path.name,
        "versioned_artifact": versioned_path.name,
        "generated_at_utc": generated_at.isoformat(),
        "label_mode": effective_label_mode,
        "label_field": label_field if effective_label_mode == "labeled" else None,
        "event_count": len(events),
        "windows": ordered_windows,
        "window_specs": ordered_windows,
        "anchor_timestamp_utc": latest_ts.isoformat(),
        "report_artifacts": report_paths,
        "evidence_link_artifacts": evidence_link_paths,
        "attribute_index_artifacts": attribute_index_paths,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Compatibility output for current inference path (maps account_id -> 30d ring_score).
    primary_window = "30d" if "30d" in ordered_windows else ordered_windows[-1]
    compatibility_scores = {
        user_id: float(window_payload.get(primary_window, {}).get("ring_score", 0.0))
        for user_id, window_payload in artifact["features_by_account"].items()
    }
    scores_compat_path.write_text(json.dumps(compatibility_scores, indent=2), encoding="utf-8")
    shutil.copyfile(monitoring_dir / report_paths[primary_window], reports_compat_path)
    shutil.copyfile(monitoring_dir / evidence_link_paths[primary_window], evidence_links_compat_path)
    shutil.copyfile(monitoring_dir / attribute_index_paths[primary_window], attribute_index_compat_path)

    summary_payload = {
        "generated_at_utc": generated_at.isoformat(),
        "mode": effective_label_mode,
        "label_field": label_field if effective_label_mode == "labeled" else None,
        "artifact_version": version,
        "artifact_path": str(versioned_path),
        "latest_artifact_path": str(latest_path),
        "manifest_path": str(manifest_path),
        "primary_inference_window": primary_window,
        "canonical_reports_path": str(reports_compat_path),
        "canonical_evidence_links_path": str(evidence_links_compat_path),
        "canonical_attribute_index_path": str(attribute_index_compat_path),
        "top_10_rings_by_risk": json.loads((monitoring_dir / report_paths[primary_window]).read_text(encoding="utf-8"))[:10],
        "window_summaries": window_summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\nArtifacts written:")
    print(f"  Versioned features : {versioned_path}")
    print(f"  Latest features    : {latest_path}")
    print(f"  Manifest           : {manifest_path}")
    print(f"  Inference scores   : {scores_compat_path} (window={primary_window})")
    print(f"  Ring reports       : {reports_compat_path} (window={primary_window})")
    print(f"  Evidence links     : {evidence_links_compat_path} (window={primary_window})")
    print(f"  Attribute index    : {attribute_index_compat_path} (window={primary_window})")
    print(f"  Summary            : {summary_path}")


if __name__ == "__main__":
    main()
