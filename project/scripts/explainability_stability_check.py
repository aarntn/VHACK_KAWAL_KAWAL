import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_FIXTURES = REPO_ROOT / "project" / "tests" / "fixtures" / "fraud_payloads.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "project" / "outputs" / "monitoring" / "explainability_stability.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "project" / "outputs" / "monitoring" / "explainability_stability_cases.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reason-code stability for similar transaction perturbations.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--samples-per-case", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=0.05, help="Relative jitter applied to numeric context-like fields.")
    return parser.parse_args()


def load_fixtures(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def perturb_payload(payload: Dict[str, Any], rng: random.Random, jitter: float) -> Dict[str, Any]:
    out = dict(payload)
    numeric_fields = [
        "TransactionAmt",
        "TransactionDT",
        "device_risk_score",
        "ip_risk_score",
        "location_risk_score",
        "cash_flow_velocity_1h",
        "p2p_counterparties_24h",
    ]
    for field in numeric_fields:
        if field not in out:
            continue
        base = float(out[field])
        delta = base * rng.uniform(-jitter, jitter)
        if field in {"device_risk_score", "ip_risk_score", "location_risk_score"}:
            out[field] = clamp(base + delta, 0.0, 1.0)
        elif field == "TransactionDT":
            out[field] = max(0.0, base + abs(delta))
        else:
            out[field] = max(0.0, base + delta)

    if "device_shared_users_24h" in out:
        out["device_shared_users_24h"] = max(0, int(out["device_shared_users_24h"] + rng.choice([-1, 0, 1])))
    if "account_age_days" in out:
        out["account_age_days"] = max(0, int(out["account_age_days"] + rng.choice([-1, 0, 1])))

    return out


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        from project.app import hybrid_fraud_api
    except ModuleNotFoundError:
        import app.hybrid_fraud_api as hybrid_fraud_api

    client = TestClient(hybrid_fraud_api.app)
    fixtures = load_fixtures(args.fixtures)

    rows: List[dict] = []

    for case_name, payload in fixtures.items():
        baseline_resp = client.post("/score_transaction", json=payload)
        if baseline_resp.status_code != 200:
            rows.append(
                {
                    "case": case_name,
                    "status": "baseline_error",
                    "baseline_status_code": baseline_resp.status_code,
                    "decision_consistency": 0.0,
                    "reason_jaccard_mean": 0.0,
                    "samples": 0,
                }
            )
            continue

        base_json = baseline_resp.json()
        base_decision = base_json.get("decision")
        base_reasons = base_json.get("reasons", [])

        same_decision = 0
        compared = 0
        jaccards: List[float] = []

        for _ in range(args.samples_per_case):
            perturbed = perturb_payload(payload, rng, args.jitter)
            resp = client.post("/score_transaction", json=perturbed)
            if resp.status_code != 200:
                continue
            compared += 1
            body = resp.json()
            if body.get("decision") == base_decision:
                same_decision += 1
                jaccards.append(jaccard(base_reasons, body.get("reasons", [])))

        rows.append(
            {
                "case": case_name,
                "status": "ok",
                "baseline_decision": base_decision,
                "samples": compared,
                "decision_consistency": (same_decision / compared) if compared else 0.0,
                "reason_jaccard_mean": (sum(jaccards) / len(jaccards)) if jaccards else 0.0,
                "same_decision_samples": same_decision,
            }
        )

    df = pd.DataFrame(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    ok_df = df[df["status"] == "ok"] if "status" in df.columns else pd.DataFrame()
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures": str(args.fixtures),
        "samples_per_case": args.samples_per_case,
        "seed": args.seed,
        "jitter": args.jitter,
        "summary": {
            "cases_evaluated": int(len(ok_df)),
            "decision_consistency_mean": float(ok_df["decision_consistency"].mean()) if not ok_df.empty else 0.0,
            "reason_jaccard_mean": float(ok_df["reason_jaccard_mean"].mean()) if not ok_df.empty else 0.0,
        },
        "cases": rows,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Explainability stability evaluation complete")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved explainability JSON: {args.output_json}")
    print(f"Saved explainability CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
