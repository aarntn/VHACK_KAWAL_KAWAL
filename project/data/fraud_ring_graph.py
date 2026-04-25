"""
Fraud ring detection via graph analysis.

Builds a bipartite graph where:
  - Account nodes  = user / entity IDs
  - Attribute nodes = shared risk attributes (device_id, card prefix, IP subnet)
  - Edges = account <-> attribute membership

Connected components in the account projection identify potential "mule rings":
groups of accounts linked through shared devices, cards, or IP ranges.

Each connected component is scored by:
  - fraud_rate:   fraction of accounts in the ring with known fraud label
  - ring_size:    number of account nodes in the component
  - ring_density: edge density of the component
  - ring_score:   composite risk score in [0, 1]

Offline batch module — intended to run nightly or on-demand against the
transaction store. The API can then look up ring_score for any incoming user.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AccountNode:
    account_id: str
    is_fraud: bool = False
    fraud_count: int = 0
    tx_count: int = 0


@dataclass
class RingReport:
    ring_id: str
    member_accounts: List[str]
    ring_size: int
    fraud_count: int | None
    fraud_rate: float | None
    shared_attributes: List[str]        # attribute values linking this ring
    attribute_types: List[str]          # e.g. ["device_id", "ip_subnet"]
    ring_score: float                   # composite risk score [0, 1]
    label_mode: str = "topology_only"
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphEvidenceLink:
    ring_id: str
    account_id: str
    attribute: str
    attr_type: str
    ring_score: float
    ring_size: int
    support_count: int
    member_count: int
    first_seen_utc: str | None = None
    last_seen_utc: str | None = None
    window: str | None = None
    label_mode: str = "topology_only"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphBuildSummary:
    account_nodes: int
    attribute_nodes: int
    edges: int
    components_total: int
    rings_detected: int               # components with >= 2 accounts
    high_risk_rings: int              # ring_score >= 0.5
    accounts_in_rings: int
    build_time_ms: float
    label_mode: str = "topology_only"
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class FraudRingGraph:
    """
    Builds a bipartite (account ↔ attribute) graph and exposes:
      - per-ring reports
      - per-account ring_score lookup
    """

    # Default hand-tuned weights (used when no trained model is available)
    _DEFAULT_SUPERVISED  = {"fraud_rate": 0.60, "size_factor": 0.25, "sharing_factor": 0.15, "attr_diversity": 0.0}
    _DEFAULT_TOPOLOGY    = {"size_factor": 0.25, "sharing_factor": 0.15, "attr_diversity": 0.0}

    def __init__(
        self,
        min_ring_size: int = 2,
        high_risk_threshold: float = 0.5,
        *,
        label_mode: str = "topology_only",
        weights_path: Optional[str] = None,
    ):
        if not _HAS_NETWORKX:
            raise ImportError(
                "networkx is required for fraud ring detection. "
                "Install it with: pip install networkx"
            )
        self.min_ring_size = min_ring_size
        self.high_risk_threshold = high_risk_threshold
        self.label_mode = label_mode

        self._supervised_w, self._topology_w, self._weights_source = \
            self._load_weights(weights_path)

        self._graph: "nx.Graph" = nx.Graph()
        self._accounts: Dict[str, AccountNode] = {}
        self._ring_reports: List[RingReport] = []
        self._account_ring_score: Dict[str, float] = {}
        self._account_ring_id: Dict[str, Optional[str]] = {}
        self._generated_at = time.time()
        self._built = False

    @classmethod
    def _load_weights(cls, weights_path: Optional[str]) -> tuple:
        """Load trained weights from JSON artifact; fall back to hand-tuned defaults."""
        if weights_path is None:
            return cls._DEFAULT_SUPERVISED.copy(), cls._DEFAULT_TOPOLOGY.copy(), "default"
        try:
            artifact = json.loads(Path(weights_path).read_text(encoding="utf-8"))
            sup  = artifact.get("supervised_weights", cls._DEFAULT_SUPERVISED)
            topo = artifact.get("topology_weights",   cls._DEFAULT_TOPOLOGY)
            return sup, topo, weights_path
        except Exception:
            return cls._DEFAULT_SUPERVISED.copy(), cls._DEFAULT_TOPOLOGY.copy(), "default_fallback"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_transaction(
        self,
        account_id: str,
        *,
        device_id: Optional[str] = None,
        ip_subnet: Optional[str] = None,
        card_prefix: Optional[str] = None,
        is_fraud: bool = False,
        timestamp: Any = None,
    ) -> None:
        """Register a transaction, connecting account to shared attributes."""
        self._built = False
        ts = _normalize_timestamp(timestamp)

        if account_id not in self._accounts:
            self._accounts[account_id] = AccountNode(account_id=account_id)
            self._graph.add_node(account_id, ntype="account")

        node = self._accounts[account_id]
        node.tx_count += 1
        if is_fraud:
            node.is_fraud = True
            node.fraud_count += 1

        for attr_type, attr_value in [
            ("device", device_id),
            ("ip",     ip_subnet),
            ("card",   card_prefix),
        ]:
            if not attr_value:
                continue
            attr_node = f"{attr_type}:{attr_value}"
            if not self._graph.has_node(attr_node):
                self._graph.add_node(attr_node, ntype="attribute", attr_type=attr_type)
            if not self._graph.has_edge(account_id, attr_node):
                self._graph.add_edge(
                    account_id,
                    attr_node,
                    edge_type=attr_type,
                    support_count=0,
                    first_seen_ts=None,
                    last_seen_ts=None,
                )
            edge_data = self._graph[account_id][attr_node]
            edge_data["support_count"] = int(edge_data.get("support_count", 0)) + 1
            if ts is not None:
                first_seen_ts = edge_data.get("first_seen_ts")
                last_seen_ts = edge_data.get("last_seen_ts")
                edge_data["first_seen_ts"] = ts if first_seen_ts is None else min(float(first_seen_ts), ts)
                edge_data["last_seen_ts"] = ts if last_seen_ts is None else max(float(last_seen_ts), ts)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def build(self) -> GraphBuildSummary:
        """
        Project bipartite graph onto account nodes, find connected components,
        score each ring.
        """
        if not _HAS_NETWORKX:
            raise ImportError("networkx is required")

        t0 = time.perf_counter()
        self._generated_at = time.time()

        account_nodes = {n for n, d in self._graph.nodes(data=True) if d.get("ntype") == "account"}
        attr_nodes    = {n for n, d in self._graph.nodes(data=True) if d.get("ntype") == "attribute"}

        # Project to account-only graph: two accounts are connected if they
        # share at least one attribute node.
        account_graph: "nx.Graph" = nx.Graph()
        account_graph.add_nodes_from(account_nodes)

        for attr_node in attr_nodes:
            neighbors = [n for n in self._graph.neighbors(attr_node) if n in account_nodes]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = neighbors[i], neighbors[j]
                    if not account_graph.has_edge(a, b):
                        account_graph.add_edge(a, b)

        # Find connected components
        self._ring_reports = []
        self._account_ring_score = {}
        self._account_ring_id = {acc: None for acc in account_nodes}
        components_total = 0
        rings_detected = 0
        high_risk_rings = 0
        accounts_in_rings = 0

        for comp in nx.connected_components(account_graph):
            components_total += 1
            members = sorted(comp)
            if len(members) < self.min_ring_size:
                for m in members:
                    self._account_ring_score[m] = 0.0
                continue

            rings_detected += 1
            if self.label_mode == "labeled":
                fraud_count: int | None = sum(1 for m in members if self._accounts.get(m, AccountNode("")).is_fraud)
                fraud_rate: float | None = fraud_count / len(members) if members else 0.0
            else:
                fraud_count = None
                fraud_rate = None

            # Collect shared attributes for this ring
            shared_attrs: List[str] = []
            attr_types_seen: List[str] = []
            for m in members:
                for nbr in self._graph.neighbors(m):
                    if self._graph.nodes[nbr].get("ntype") == "attribute":
                        # Only include attributes shared by ≥ 2 members
                        ring_neighbors = [n for n in self._graph.neighbors(nbr) if n in comp]
                        if len(ring_neighbors) >= 2 and nbr not in shared_attrs:
                            shared_attrs.append(nbr)
                            attr_types_seen.append(self._graph.nodes[nbr].get("attr_type", "unknown"))

            # Ring score: learned-weight combination of fraud_rate and topology features
            size_factor    = min(1.0, len(members) / 20.0)
            sharing_factor = min(1.0, len(shared_attrs) / 5.0)
            known_types    = {"device", "ip", "card"}
            attr_diversity = len(set(attr_types_seen) & known_types) / 3.0
            if fraud_rate is None:
                tw = self._topology_w
                ring_score = min(1.0,
                    tw.get("size_factor",    0.25) * size_factor +
                    tw.get("sharing_factor", 0.15) * sharing_factor +
                    tw.get("attr_diversity", 0.0)  * attr_diversity)
            else:
                sw = self._supervised_w
                ring_score = min(1.0,
                    sw.get("fraud_rate",     0.60) * fraud_rate +
                    sw.get("size_factor",    0.25) * size_factor +
                    sw.get("sharing_factor", 0.15) * sharing_factor +
                    sw.get("attr_diversity", 0.0)  * attr_diversity)

            ring_id = f"ring_{hashlib.sha1('|'.join(members).encode()).hexdigest()[:8]}"
            report = RingReport(
                ring_id=ring_id,
                member_accounts=members,
                ring_size=len(members),
                fraud_count=fraud_count,
                fraud_rate=round(fraud_rate, 4) if fraud_rate is not None else None,
                shared_attributes=sorted(set(shared_attrs)),
                attribute_types=sorted(set(attr_types_seen)),
                ring_score=round(ring_score, 4),
                label_mode=self.label_mode,
                generated_at=self._generated_at,
            )
            self._ring_reports.append(report)
            accounts_in_rings += len(members)

            for m in members:
                self._account_ring_score[m] = ring_score
                self._account_ring_id[m]    = ring_id

            if ring_score >= self.high_risk_threshold:
                high_risk_rings += 1

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._built = True

        return GraphBuildSummary(
            account_nodes=len(account_nodes),
            attribute_nodes=len(attr_nodes),
            edges=self._graph.number_of_edges(),
            components_total=components_total,
            rings_detected=rings_detected,
            high_risk_rings=high_risk_rings,
            accounts_in_rings=accounts_in_rings,
            build_time_ms=elapsed_ms,
            label_mode=self.label_mode,
            generated_at=self._generated_at,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_ring_score(self, account_id: str) -> float:
        """Return ring risk score [0, 1] for an account. 0.0 if not in any ring."""
        return self._account_ring_score.get(account_id, 0.0)

    def get_ring_id(self, account_id: str) -> Optional[str]:
        """Return the ring_id the account belongs to, or None."""
        return self._account_ring_id.get(account_id)

    def get_ring_reports(self) -> List[RingReport]:
        return list(self._ring_reports)

    def get_evidence_links(self, window: str | None = None) -> List[GraphEvidenceLink]:
        if not self._built:
            self.build()
        links: List[GraphEvidenceLink] = []
        seen: set[tuple[str, str, str]] = set()

        for report in self._ring_reports:
            component_members = set(report.member_accounts)
            for attr_token in report.shared_attributes:
                ring_neighbors = sorted(
                    n for n in self._graph.neighbors(attr_token) if n in component_members
                )
                member_count = len(ring_neighbors)
                attr_type = str(self._graph.nodes[attr_token].get("attr_type", "unknown"))
                for account_id in ring_neighbors:
                    key = (report.ring_id, account_id, attr_token)
                    if key in seen:
                        continue
                    seen.add(key)
                    edge_data = self._graph.get_edge_data(account_id, attr_token) or {}
                    links.append(
                        GraphEvidenceLink(
                            ring_id=report.ring_id,
                            account_id=account_id,
                            attribute=attr_token,
                            attr_type=attr_type,
                            ring_score=float(report.ring_score),
                            ring_size=int(report.ring_size),
                            support_count=int(edge_data.get("support_count", 0) or 0),
                            member_count=member_count,
                            first_seen_utc=_timestamp_to_utc_iso(edge_data.get("first_seen_ts")),
                            last_seen_utc=_timestamp_to_utc_iso(edge_data.get("last_seen_ts")),
                            window=window,
                            label_mode=report.label_mode,
                        )
                    )
        return links

    def get_attribute_risk_index(self, window: str | None = None) -> Dict[str, Dict[str, Any]]:
        if not self._built:
            self.build()

        account_nodes = {n for n, d in self._graph.nodes(data=True) if d.get("ntype") == "account"}
        attr_index: Dict[str, Dict[str, Any]] = {}

        for report in self._ring_reports:
            component_members = set(report.member_accounts)
            for attr_token in report.shared_attributes:
                attr_type = str(self._graph.nodes[attr_token].get("attr_type", "unknown"))
                all_neighbors = {n for n in self._graph.neighbors(attr_token) if n in account_nodes}
                ring_neighbors = sorted(n for n in all_neighbors if n in component_members)
                first_seen_values = []
                last_seen_values = []
                for account_id in ring_neighbors:
                    edge_data = self._graph.get_edge_data(account_id, attr_token) or {}
                    if edge_data.get("first_seen_ts") is not None:
                        first_seen_values.append(float(edge_data["first_seen_ts"]))
                    if edge_data.get("last_seen_ts") is not None:
                        last_seen_values.append(float(edge_data["last_seen_ts"]))

                entry = attr_index.setdefault(
                    attr_token,
                    {
                        "attr_type": attr_type,
                        "max_ring_score": 0.0,
                        "ring_count": 0,
                        "max_ring_size": 0,
                        "member_count": 0,
                        "ring_ids": [],
                        "generated_at": self._generated_at,
                        "generated_at_utc": _timestamp_to_utc_iso(self._generated_at),
                        "window": window,
                        "label_mode": report.label_mode,
                        "first_seen_utc": None,
                        "last_seen_utc": None,
                    },
                )
                entry["max_ring_score"] = round(
                    max(float(entry["max_ring_score"]), float(report.ring_score)),
                    4,
                )
                entry["ring_count"] = int(entry["ring_count"]) + 1
                entry["max_ring_size"] = max(int(entry["max_ring_size"]), int(report.ring_size))
                entry["member_count"] = max(int(entry["member_count"]), len(all_neighbors))
                entry["label_mode"] = report.label_mode
                entry["window"] = window
                entry["generated_at"] = self._generated_at
                entry["generated_at_utc"] = _timestamp_to_utc_iso(self._generated_at)
                entry["ring_ids"] = sorted({*entry["ring_ids"], report.ring_id})
                if first_seen_values:
                    first_seen = min(first_seen_values)
                    current_first = _normalize_timestamp(entry.get("first_seen_utc"))
                    entry["first_seen_utc"] = _timestamp_to_utc_iso(
                        first_seen if current_first is None else min(current_first, first_seen)
                    )
                if last_seen_values:
                    last_seen = max(last_seen_values)
                    current_last = _normalize_timestamp(entry.get("last_seen_utc"))
                    entry["last_seen_utc"] = _timestamp_to_utc_iso(
                        last_seen if current_last is None else max(current_last, last_seen)
                    )
        return attr_index

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def dump_scores(self) -> Dict[str, float]:
        """Return {account_id: ring_score} mapping."""
        return dict(self._account_ring_score)

    def dump_rings_json(self, path: Path) -> None:
        """Write ring reports to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([r.to_dict() for r in self._ring_reports], indent=2),
            encoding="utf-8",
        )

    def dump_scores_json(self, path: Path) -> None:
        """Write {account_id: ring_score} to a JSON file for API lookup."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._account_ring_score, indent=2), encoding="utf-8")

    def dump_evidence_links_json(self, path: Path, window: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([item.to_dict() for item in self.get_evidence_links(window=window)], indent=2),
            encoding="utf-8",
        )

    def dump_attribute_index_json(self, path: Path, window: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.get_attribute_risk_index(window=window), indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Convenience loader for runtime lookup
# ---------------------------------------------------------------------------

class RingScoreLookup:
    """
    Lightweight read-only lookup loaded by the API at startup.
    Falls back to 0.0 for unknown accounts.
    """

    def __init__(self, scores_path: Path):
        self._scores: Dict[str, float] = {}
        if scores_path.exists():
            try:
                raw = json.loads(scores_path.read_text(encoding="utf-8"))
                self._scores = {str(k): float(v) for k, v in raw.items()}
            except Exception:
                pass

    def get(self, account_id: str, default: float = 0.0) -> float:
        return self._scores.get(str(account_id), default)

    def __len__(self) -> int:
        return len(self._scores)


class RingAttributeLookup:
    """Read-only attribute-risk lookup loaded by the API at startup."""

    def __init__(self, index_path: Path):
        self._index: Dict[str, Dict[str, Any]] = {}
        if index_path.exists():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._index = {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
            except Exception:
                pass

    def get(self, attribute_token: str, default: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        return self._index.get(str(attribute_token), default)

    def __len__(self) -> int:
        return len(self._index)


def _normalize_timestamp(timestamp: Any) -> float | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raw = str(timestamp).strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _timestamp_to_utc_iso(timestamp: Any) -> str | None:
    normalized = _normalize_timestamp(timestamp)
    if normalized is None:
        return None
    return datetime.fromtimestamp(normalized, tz=timezone.utc).isoformat()
