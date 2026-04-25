from __future__ import annotations

import json
import pickle
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

try:
    from project.app import hybrid_fraud_api
    from project.scripts import build_fraud_ring_graph
    from project.scripts import evaluate_ring_replay
except ModuleNotFoundError:
    import app.hybrid_fraud_api as hybrid_fraud_api
    from scripts import build_fraud_ring_graph
    from scripts import evaluate_ring_replay


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "fraud_payloads.json"


def load_payload_fixtures() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def serving_payload(payload: dict) -> dict:
    trimmed = dict(payload)
    for feature_idx in range(18, 29):
        trimmed.pop(f"V{feature_idx}", None)
    return trimmed


class MappingLookup:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key: str, default=None):
        return self.values.get(str(key), default)

    def __len__(self) -> int:
        return len(self.values)


class RingRemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(hybrid_fraud_api.app)
        cls.fixtures = load_payload_fixtures()

    def test_ring_graph_endpoint_uses_exact_evidence_links(self) -> None:
        reports = [
            {
                "ring_id": "ring_exact",
                "member_accounts": ["acct_1", "acct_2"],
                "shared_attributes": ["device:dev_shared", "ip:10.0.0.0/24"],
                "attribute_types": ["device", "ip"],
                "ring_size": 2,
                "ring_score": 0.84,
                "fraud_rate": 0.5,
                "label_mode": "labeled",
            }
        ]
        exact_links = [
            {
                "ring_id": "ring_exact",
                "account_id": "acct_1",
                "attribute": "device:dev_shared",
                "attr_type": "device",
                "ring_score": 0.84,
                "ring_size": 2,
                "support_count": 2,
                "member_count": 2,
                "first_seen_utc": "2026-04-24T10:00:00+00:00",
                "last_seen_utc": "2026-04-24T10:05:00+00:00",
                "window": "30d",
            },
            {
                "ring_id": "ring_exact",
                "account_id": "acct_2",
                "attribute": "ip:10.0.0.0/24",
                "attr_type": "ip",
                "ring_score": 0.84,
                "ring_size": 2,
                "support_count": 1,
                "member_count": 2,
                "first_seen_utc": "2026-04-24T10:01:00+00:00",
                "last_seen_utc": "2026-04-24T10:06:00+00:00",
                "window": "30d",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_path = tmp_path / "fraud_ring_reports.json"
            evidence_path = tmp_path / "fraud_ring_evidence_links.json"
            reports_path.write_text(json.dumps(reports), encoding="utf-8")
            evidence_path.write_text(json.dumps(exact_links), encoding="utf-8")

            with patch.object(hybrid_fraud_api, "RING_REPORTS_PATH", reports_path), patch.object(
                hybrid_fraud_api, "RING_EVIDENCE_LINKS_PATH", evidence_path
            ), patch.object(
                hybrid_fraud_api,
                "ring_score_lookup",
                MappingLookup({"acct_1": 0.91, "acct_2": 0.73}),
            ):
                response = self.client.get("/ring/graph")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["summary"]["evidence_links_available"])
        self.assertEqual(len(body["links"]), 2)
        observed_pairs = {(item["source"], item["target"]) for item in body["links"]}
        self.assertEqual(
            observed_pairs,
            {("acct_1", "attr_device:dev_shared"), ("acct_2", "attr_ip:10.0.0.0/24")},
        )
        first_link = body["links"][0]
        self.assertIn("support_count", first_link)
        self.assertIn("first_seen_utc", first_link)
        self.assertIn("window", first_link)

    def test_ring_graph_endpoint_without_evidence_links_returns_no_fabricated_edges(self) -> None:
        reports = [
            {
                "ring_id": "ring_sparse",
                "member_accounts": ["acct_1", "acct_2"],
                "shared_attributes": ["device:dev_shared", "ip:10.0.0.0/24"],
                "attribute_types": ["device", "ip"],
                "ring_size": 2,
                "ring_score": 0.62,
                "fraud_rate": None,
                "label_mode": "topology_only",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_path = Path(tmpdir) / "fraud_ring_reports.json"
            reports_path.write_text(json.dumps(reports), encoding="utf-8")
            with patch.object(hybrid_fraud_api, "RING_REPORTS_PATH", reports_path), patch.object(
                hybrid_fraud_api, "RING_EVIDENCE_LINKS_PATH", Path(tmpdir) / "missing.json"
            ), patch.object(
                hybrid_fraud_api,
                "ring_score_lookup",
                MappingLookup({"acct_1": 0.62, "acct_2": 0.62}),
            ):
                response = self.client.get("/ring/graph")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertFalse(body["summary"]["evidence_links_available"])
        self.assertEqual(body["links"], [])
        self.assertEqual(len(body["rings"]), 1)
        self.assertEqual(body["rings"][0]["label_mode"], "topology_only")

    def test_resolve_ring_signal_uses_attribute_match_for_unseen_account(self) -> None:
        attribute_lookup = MappingLookup(
            {
                "device:dev_risky": {
                    "attr_type": "device",
                    "max_ring_score": 0.88,
                    "ring_count": 2,
                    "max_ring_size": 6,
                    "member_count": 4,
                    "ring_ids": ["ring_attr"],
                    "generated_at": time.time(),
                    "label_mode": "topology_only",
                }
            }
        )
        with patch.object(hybrid_fraud_api, "ring_score_lookup", MappingLookup()), patch.object(
            hybrid_fraud_api, "ring_evidence_lookup", {}
        ), patch.object(hybrid_fraud_api, "ring_attribute_lookup", attribute_lookup), patch.object(
            hybrid_fraud_api, "RING_ATTRIBUTE_DEVICE_MEMBER_CAP", 10
        ), patch.object(
            hybrid_fraud_api, "RING_ATTRIBUTE_MATCH_MIN", 1
        ), patch.object(
            hybrid_fraud_api, "RING_ATTRIBUTE_MATCH_COMPONENT_FLOOR", 4
        ):
            score, match_type, reason_codes, evidence_summary, gate_evidence = hybrid_fraud_api.resolve_ring_signal(
                {"user_id": "acct_new", "device_id": "dev_risky"}
            )

        self.assertGreater(score, 0.0)
        self.assertEqual(match_type, "attribute_match")
        self.assertIn("RING_MATCH_ATTRIBUTE", reason_codes)
        self.assertEqual(evidence_summary["match_type"], "attribute_match")
        self.assertEqual(gate_evidence["ring_size"], 6)

    def test_resolve_ring_signal_suppresses_noisy_attribute_matches(self) -> None:
        attribute_lookup = MappingLookup(
            {
                "device:dev_hub": {
                    "attr_type": "device",
                    "max_ring_score": 0.91,
                    "ring_count": 3,
                    "max_ring_size": 8,
                    "member_count": 99,
                    "ring_ids": ["ring_hub"],
                    "generated_at": time.time(),
                    "label_mode": "topology_only",
                }
            }
        )
        with patch.object(hybrid_fraud_api, "ring_score_lookup", MappingLookup()), patch.object(
            hybrid_fraud_api, "ring_evidence_lookup", {}
        ), patch.object(hybrid_fraud_api, "ring_attribute_lookup", attribute_lookup), patch.object(
            hybrid_fraud_api, "RING_ATTRIBUTE_DEVICE_MEMBER_CAP", 10
        ):
            score, match_type, reason_codes, evidence_summary, gate_evidence = hybrid_fraud_api.resolve_ring_signal(
                {"user_id": "acct_new", "device_id": "dev_hub"}
            )

        self.assertEqual(score, 0.0)
        self.assertEqual(match_type, "none")
        self.assertIn("RING_ATTRIBUTE_NOISY_SUPPRESSED", reason_codes)
        self.assertEqual(evidence_summary["match_type"], "none")
        self.assertEqual(gate_evidence, {})

    def test_ring_block_fairness_guard_clips_ring_only_block_escalation(self) -> None:
        segment_thresholds = hybrid_fraud_api.SegmentThresholds(
            approve_threshold=0.30,
            block_threshold=0.70,
        )
        with patch.object(hybrid_fraud_api, "RING_BLOCK_FAIRNESS_GUARD_ENABLED", True), patch.object(
            hybrid_fraud_api, "RING_BLOCK_GUARD_SEGMENTS", {"*"}
        ):
            adjustment, reason_codes, guard_details = hybrid_fraud_api.apply_ring_block_fairness_guard(
                user_segment="new:standard:APP",
                segment_thresholds=segment_thresholds,
                pre_ring_score=0.68,
                ring_adjustment=0.08,
                ring_match_type="attribute_match",
                context_adjustment=0.02,
                behavior_adjustment=0.01,
                external_adjustment=0.0,
                hard_rule_hits=[],
            )

        self.assertLess(adjustment, 0.08)
        self.assertGreaterEqual(adjustment, 0.0)
        self.assertIn("RING_BLOCK_ESCALATION_SUPPRESSED_FAIRNESS", reason_codes)
        self.assertTrue(guard_details["enabled"])
        self.assertFalse(guard_details["corroborated"])

    def test_score_transaction_response_includes_ring_provenance(self) -> None:
        payload = serving_payload(self.fixtures["approve_case"])
        payload.update(
            {
                "user_id": "acct_member",
                "connectivity_mode": "offline_buffered",
            }
        )
        with patch.object(hybrid_fraud_api, "ring_score_lookup", MappingLookup({"acct_member": 0.93})), patch.object(
            hybrid_fraud_api,
            "ring_evidence_lookup",
            {
                "acct_member": {
                    "ring_id": "ring_live",
                    "ring_size": 6,
                    "attribute_type_count": 2,
                    "shared_attribute_count": 3,
                    "generated_at": time.time(),
                    "label_mode": "labeled",
                }
            },
        ), patch.object(hybrid_fraud_api, "ring_attribute_lookup", MappingLookup()), patch.object(
            hybrid_fraud_api, "RING_BLOCK_FAIRNESS_GUARD_ENABLED", False
        ):
            response = self.client.post("/score_transaction", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["ring_match_type"], "account_member")
        self.assertEqual(body["ring_evidence_summary"]["match_type"], "account_member")
        self.assertIn("ring_match_type", body["context_summary"])
        self.assertIn("ring_evidence_summary", body["context_summary"])


class RingBuildArtifactTests(unittest.TestCase):
    def test_build_fraud_ring_graph_writes_label_safe_artifacts(self) -> None:
        events = [
            {
                "user_id": "acct_1",
                "device_id": "dev_shared",
                "ip_subnet": "10.0.0.0/24",
                "card_prefix": "411111",
                "timestamp": "2026-04-24T09:00:00Z",
                "is_fraud": True,
            },
            {
                "user_id": "acct_2",
                "device_id": "dev_shared",
                "ip_subnet": "10.0.0.0/24",
                "card_prefix": "411111",
                "timestamp": "2026-04-24T09:05:00Z",
                "is_fraud": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            events_path = tmp_path / "events.jsonl"
            output_dir = tmp_path / "out"
            events_path.write_text(
                "\n".join(json.dumps(row) for row in events),
                encoding="utf-8",
            )

            argv = [
                "build_fraud_ring_graph.py",
                "--events-path",
                str(events_path),
                "--output-dir",
                str(output_dir),
                "--windows",
                "30d",
                "--label-mode",
                "labeled",
            ]
            with patch.object(sys, "argv", argv):
                build_fraud_ring_graph.main()

            monitoring_dir = output_dir / "monitoring"
            summary = json.loads((monitoring_dir / "fraud_ring_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((monitoring_dir / "fraud_ring_features_manifest.json").read_text(encoding="utf-8"))
            reports = json.loads((monitoring_dir / "fraud_ring_reports.json").read_text(encoding="utf-8"))
            evidence_links = json.loads((monitoring_dir / "fraud_ring_evidence_links.json").read_text(encoding="utf-8"))
            attr_index = json.loads((monitoring_dir / "fraud_ring_attribute_index.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["mode"], "labeled")
        self.assertEqual(manifest["label_mode"], "labeled")
        self.assertEqual(reports[0]["label_mode"], "labeled")
        self.assertIsNotNone(reports[0]["fraud_rate"])
        self.assertTrue(evidence_links)
        self.assertIn("device:dev_shared", attr_index)

    def test_build_fraud_ring_graph_rejects_incomplete_labels_in_labeled_mode(self) -> None:
        events = [
            {
                "user_id": "acct_1",
                "device_id": "dev_shared",
                "timestamp": "2026-04-24T09:00:00Z",
                "is_fraud": True,
            },
            {
                "user_id": "acct_2",
                "device_id": "dev_shared",
                "timestamp": "2026-04-24T09:05:00Z",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            events_path = tmp_path / "events.jsonl"
            output_dir = tmp_path / "out"
            events_path.write_text(
                "\n".join(json.dumps(row) for row in events),
                encoding="utf-8",
            )
            argv = [
                "build_fraud_ring_graph.py",
                "--events-path",
                str(events_path),
                "--output-dir",
                str(output_dir),
                "--windows",
                "30d",
                "--label-mode",
                "labeled",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(ValueError):
                    build_fraud_ring_graph.main()

    def test_evaluate_ring_replay_writes_measured_report(self) -> None:
        replay_rows = [
            {
                "user_id": "acct_member",
                "device_id": "dev_known",
                "baseline_score": 0.42,
                "is_fraud": True,
                "account_age_days": 180,
                "channel": "APP",
            },
            {
                "user_id": "acct_new",
                "device_id": "dev_attr",
                "baseline_score": 0.21,
                "is_fraud": False,
                "account_age_days": 5,
                "channel": "AGENT",
                "is_agent_assisted": True,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            events_path = tmp_path / "replay.jsonl"
            thresholds_path = tmp_path / "thresholds.pkl"
            ring_scores_path = tmp_path / "fraud_ring_scores.json"
            attr_index_path = tmp_path / "fraud_ring_attribute_index.json"
            output_json = tmp_path / "ring_replay_report.json"
            output_md = tmp_path / "ring_replay_report.md"

            events_path.write_text(
                "\n".join(json.dumps(row) for row in replay_rows),
                encoding="utf-8",
            )
            thresholds_path.write_bytes(
                pickle.dumps({"approve_threshold": 0.30, "block_threshold": 0.70})
            )
            ring_scores_path.write_text(json.dumps({"acct_member": 0.90}), encoding="utf-8")
            attr_index_path.write_text(
                json.dumps(
                    {
                        "device:dev_attr": {
                            "attr_type": "device",
                            "max_ring_score": 0.75,
                            "ring_count": 1,
                            "max_ring_size": 5,
                            "member_count": 3,
                            "generated_at": time.time(),
                        }
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "evaluate_ring_replay.py",
                "--events-path",
                str(events_path),
                "--thresholds-path",
                str(thresholds_path),
                "--ring-scores-path",
                str(ring_scores_path),
                "--attribute-index-path",
                str(attr_index_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
            ]
            with patch.object(sys, "argv", argv):
                exit_code = evaluate_ring_replay.main()

            report = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["evidence_class"], "measured_replay")
        segments = {row["segment"] for row in report["match_cohorts"]}
        self.assertIn("match_type:account_member", segments)
        self.assertIn("match_type:attribute_match", segments)


if __name__ == "__main__":
    unittest.main()
