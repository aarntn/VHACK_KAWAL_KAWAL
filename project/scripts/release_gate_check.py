import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate backend execution gate and emit backend readiness report with promotion decision."
        )
    )
    parser.add_argument(
        "--ops-summary-json",
        type=Path,
        default=Path("project/outputs/monitoring/nightly_ops_summary.json"),
    )
    parser.add_argument(
        "--inference-candidate-report-json",
        type=Path,
        default=Path("project/outputs/monitoring/ieee_cis_inference_candidate_report.json"),
        help="Optional promotion candidate report emitted by benchmark_inference_candidates.py",
    )
    parser.add_argument(
        "--archive-runs-dir",
        type=Path,
        default=Path("project/outputs/ops_runs"),
        help="Archived nightly runs used for consecutive streak checks.",
    )
    parser.add_argument(
        "--required-startup-streak",
        type=int,
        default=3,
        help="Require N consecutive runs with successful service startup + artifact validation pass.",
    )
    parser.add_argument(
        "--require-payload-contract-tests",
        action="store_true",
        default=True,
        help="Require payload contract test results to be present and passing.",
    )
    parser.add_argument(
        "--max-endpoint-error-rate-pct",
        type=float,
        default=5.0,
        help="Policy threshold for endpoint error rates from benchmark_sla.sla_evaluation.",
    )
    parser.add_argument(
        "--warn-mode-exception-approval",
        default="",
        help=(
            "Explicit approval string required when benchmark_sla_mode=warn and benchmark_sla is not ok. "
            "Gate fails without this value in that condition."
        ),
    )
    parser.add_argument(
        "--approve-retrain-policy-review",
        action="store_true",
        help=(
            "Acknowledge retraining policy review when streak diagnostics indicate should_retrain=true. "
            "Without this ack, gate fails."
        ),
    )
    parser.add_argument(
        "--backend-readiness-report",
        type=Path,
        default=Path("project/outputs/monitoring/backend_readiness_report.md"),
        help="Path to one-page readiness report Markdown output.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_payload_contract_tests(payload: dict[str, Any]) -> dict[str, Any]:
    contract = payload.get("payload_contract_tests")
    if isinstance(contract, dict):
        ok = bool(contract.get("ok", False))
        failures = contract.get("failing_suites")
        return {
            "present": True,
            "ok": ok,
            "reason": "ok" if ok else contract.get("reason", "payload_contract_tests_failed"),
            "failing_suites": list(failures) if isinstance(failures, list) else [],
        }

    benchmark_payload = payload.get("benchmark")
    if isinstance(benchmark_payload, dict):
        stdout = benchmark_payload.get("stdout", "")
        if isinstance(stdout, str) and stdout.strip():
            try:
                benchmark_json = json.loads(stdout)
                contract_validation = benchmark_json.get("contract_validation")
                if isinstance(contract_validation, dict):
                    ok = bool(contract_validation.get("ok", False))
                    errors = contract_validation.get("errors")
                    return {
                        "present": True,
                        "ok": ok,
                        "reason": "ok" if ok else "payload_contract_tests_failed",
                        "failing_suites": list(errors) if isinstance(errors, list) else [],
                    }
            except json.JSONDecodeError:
                pass

    return {
        "present": False,
        "ok": False,
        "reason": "payload_contract_tests_missing",
        "failing_suites": [],
    }


def evaluate_benchmark_sla(payload: dict[str, Any]) -> dict[str, Any]:
    def _collect_uncategorized_breaches(sla_evaluation: Any) -> list[str]:
        if not isinstance(sla_evaluation, dict):
            return []
        breaches: list[str] = []
        for endpoint, endpoint_eval in sla_evaluation.items():
            checks = (endpoint_eval or {}).get("checks") or {}
            guardrail = checks.get("unknown_internal_guardrail") or {}
            if guardrail.get("pass") is False:
                breaches.append(str(endpoint))
        return breaches

    benchmark_sla = payload.get("benchmark_sla")
    if isinstance(benchmark_sla, dict):
        failing = benchmark_sla.get("failing_endpoints") or []
        sla_eval = benchmark_sla.get("sla_evaluation")
        uncategorized = benchmark_sla.get("uncategorized_endpoints")
        if not isinstance(uncategorized, list):
            uncategorized = _collect_uncategorized_breaches(sla_eval)
        return {
            "ok": bool(benchmark_sla.get("ok", False)),
            "failing_endpoints": list(failing) if isinstance(failing, list) else [],
            "reason": benchmark_sla.get("reason", "benchmark_sla_failed"),
            "benchmark_mode": benchmark_sla.get("benchmark_mode"),
            "failure_mode": benchmark_sla.get("failure_mode"),
            "uncategorized_endpoints": list(uncategorized),
            "sla_evaluation": sla_eval if isinstance(sla_eval, dict) else {},
        }

    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict) or not benchmark.get("ok", False):
        return {
            "ok": True,
            "failing_endpoints": [],
            "reason": "benchmark_not_run_or_failed_process",
            "uncategorized_endpoints": [],
            "sla_evaluation": {},
        }

    stdout = benchmark.get("stdout", "")
    if not isinstance(stdout, str) or not stdout.strip():
        return {
            "ok": False,
            "failing_endpoints": [],
            "reason": "benchmark_stdout_missing",
            "uncategorized_endpoints": [],
            "sla_evaluation": {},
        }

    try:
        benchmark_payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "failing_endpoints": [],
            "reason": "benchmark_stdout_not_json",
            "uncategorized_endpoints": [],
            "sla_evaluation": {},
        }

    if str(benchmark_payload.get("benchmark_mode", "")).lower() == "preflight_failed":
        failed_preflight = benchmark_payload.get("failed_preflight_endpoints") or []
        return {
            "ok": False,
            "failing_endpoints": list(failed_preflight) if isinstance(failed_preflight, list) else [],
            "reason": "benchmark_unreachable_services",
            "benchmark_mode": benchmark_payload.get("benchmark_mode"),
            "failure_mode": benchmark_payload.get("failure_mode"),
            "uncategorized_endpoints": [],
            "sla_evaluation": {},
        }

    sla_evaluation = benchmark_payload.get("sla_evaluation")
    if not isinstance(sla_evaluation, dict):
        return {
            "ok": False,
            "failing_endpoints": [],
            "reason": "benchmark_sla_evaluation_missing",
            "uncategorized_endpoints": [],
            "sla_evaluation": {},
        }

    failing_endpoints = [
        endpoint
        for endpoint, endpoint_eval in sla_evaluation.items()
        if str((endpoint_eval or {}).get("real_time_viability", "")).upper() != "PASS"
    ]
    return {
        "ok": len(failing_endpoints) == 0,
        "failing_endpoints": failing_endpoints,
        "reason": "ok" if len(failing_endpoints) == 0 else "benchmark_sla_failed",
        "benchmark_mode": benchmark_payload.get("benchmark_mode"),
        "failure_mode": benchmark_payload.get("failure_mode"),
        "uncategorized_endpoints": _collect_uncategorized_breaches(sla_evaluation),
        "sla_evaluation": sla_evaluation,
    }


def evaluate_endpoint_error_policy(benchmark_sla: dict[str, Any], threshold_pct: float) -> dict[str, Any]:
    sla_eval = benchmark_sla.get("sla_evaluation")
    if not isinstance(sla_eval, dict):
        return {"ok": True, "breaches": [], "threshold_pct": threshold_pct, "reason": "sla_evaluation_missing"}

    breaches: list[dict[str, Any]] = []
    for endpoint, endpoint_eval in sla_eval.items():
        checks = (endpoint_eval or {}).get("checks") or {}
        error_rate_payload = checks.get("error_rate_pct") or {}
        actual = error_rate_payload.get("actual")
        try:
            rate = float(actual)
        except (TypeError, ValueError):
            continue

        if rate > threshold_pct:
            breaches.append({"endpoint": str(endpoint), "error_rate_pct": rate})

    return {
        "ok": len(breaches) == 0,
        "breaches": sorted(breaches, key=lambda item: item["endpoint"]),
        "threshold_pct": threshold_pct,
        "reason": "ok" if len(breaches) == 0 else "endpoint_error_rate_policy_failed",
    }


def evaluate_service_startup_streak(
    current_payload: dict[str, Any], archive_runs_dir: Path, required_streak: int
) -> dict[str, Any]:
    def _startup_ok(summary: dict[str, Any]) -> bool:
        if summary.get("status") != "ok":
            return False
        artifact = summary.get("artifact_validation_report")
        if not isinstance(artifact, dict) or not artifact.get("ok", False):
            return False
        sla = evaluate_benchmark_sla(summary)
        return bool(sla.get("reason") != "benchmark_unreachable_services")

    if required_streak <= 1:
        return {
            "ok": _startup_ok(current_payload),
            "required": required_streak,
            "current_streak": 1 if _startup_ok(current_payload) else 0,
            "evaluated_runs": 1,
        }

    streak = 0
    evaluated_runs = 0
    sequence: list[dict[str, Any]] = [current_payload]

    if archive_runs_dir.exists():
        run_dirs = sorted([p for p in archive_runs_dir.iterdir() if p.is_dir()], reverse=True)
        for run_dir in run_dirs:
            summary_path = run_dir / "nightly_ops_summary.json"
            if not summary_path.exists():
                continue
            try:
                sequence.append(_load_json(summary_path))
            except (OSError, json.JSONDecodeError):
                continue
            if len(sequence) >= required_streak:
                break

    for summary in sequence:
        evaluated_runs += 1
        if _startup_ok(summary):
            streak += 1
            if streak >= required_streak:
                break
        else:
            break

    return {
        "ok": streak >= required_streak,
        "required": required_streak,
        "current_streak": streak,
        "evaluated_runs": evaluated_runs,
    }


def evaluate_retrain_policy_review(payload: dict[str, Any], acknowledged: bool) -> dict[str, Any]:
    retrain_trigger = payload.get("retrain_trigger")
    if not isinstance(retrain_trigger, dict):
        return {"ok": False, "reason": "retrain_trigger_missing", "requires_ack": False, "details": {}}

    details = retrain_trigger.get("details")
    if not isinstance(details, dict):
        return {"ok": False, "reason": "retrain_trigger_details_missing", "requires_ack": False, "details": {}}

    required_keys = ["warn_drift_streak", "benchmark_sla_fail_streak", "endpoint_error_rate_streaks"]
    missing = [key for key in required_keys if key not in details]
    if missing:
        return {
            "ok": False,
            "reason": "retrain_policy_streaks_missing",
            "requires_ack": False,
            "missing": missing,
            "details": details,
        }

    requires_ack = bool(retrain_trigger.get("should_retrain", False))
    if requires_ack and not acknowledged:
        return {
            "ok": False,
            "reason": "retrain_policy_review_ack_required",
            "requires_ack": True,
            "details": details,
            "reasons": retrain_trigger.get("reasons", []),
        }

    return {
        "ok": True,
        "reason": "ok",
        "requires_ack": requires_ack,
        "details": details,
        "reasons": retrain_trigger.get("reasons", []),
    }


def evaluate_inference_regression_gate(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {"ok": True, "reason": "report_missing"}

    payload = _load_json(report_path)
    gate = payload.get("regression_gate")
    if not isinstance(gate, dict):
        return {"ok": False, "reason": "regression_gate_missing", "path": str(report_path)}

    if gate.get("blocked", False):
        return {
            "ok": False,
            "reason": gate.get("reason", "regression_gate_blocked"),
            "path": str(report_path),
            "details": gate,
        }

    status = str(payload.get("status", "")).lower()
    if status == "hold":
        return {"ok": False, "reason": "promotion_hold", "path": str(report_path)}

    return {"ok": True, "reason": "ok", "path": str(report_path)}


def _resolve_stable_endpoints(benchmark_sla: dict[str, Any]) -> list[str]:
    sla_eval = benchmark_sla.get("sla_evaluation")
    if not isinstance(sla_eval, dict):
        return []

    stable: list[str] = []
    for endpoint, endpoint_eval in sla_eval.items():
        endpoint_eval = endpoint_eval or {}
        checks = endpoint_eval.get("checks") or {}
        error_rate_actual = (checks.get("error_rate_pct") or {}).get("actual")
        try:
            error_rate_val = float(error_rate_actual)
        except (TypeError, ValueError):
            error_rate_val = None

        if str(endpoint_eval.get("real_time_viability", "")).upper() == "PASS" and (
            error_rate_val is None or error_rate_val >= 0
        ):
            stable.append(str(endpoint))
    return sorted(stable)


def write_backend_readiness_report(
    report_path: Path,
    gate_result: dict[str, Any],
    startup_streak: dict[str, Any],
    payload_contract: dict[str, Any],
    benchmark_sla: dict[str, Any],
    endpoint_policy: dict[str, Any],
    retrain_review: dict[str, Any],
    regression_gate: dict[str, Any],
    freeze_backend_contracts: bool,
    stable_endpoints: list[str],
    warn_exception_used: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "PASS" if gate_result.get("status") == "ok" else "FAIL"
    promotion_decision = "PROMOTE_BACKEND" if readiness == "PASS" else "HOLD_BACKEND"
    frontend_action = (
        f"Begin frontend integration against stable endpoints only: {', '.join(stable_endpoints) or 'none'}"
        if freeze_backend_contracts
        else "Do not start frontend integration; backend endpoints are not yet stable."
    )

    lines = [
        "# Backend Readiness Report",
        "",
        f"- Generated at (UTC): {_utc_now_iso()}",
        f"- Gate status: **{readiness}**",
        f"- Promotion decision: **{promotion_decision}**",
        "",
        "## Gate Criteria Evidence",
        f"1. Consecutive successful startups + artifact validation: {'PASS' if startup_streak.get('ok') else 'FAIL'} "
        f"(required={startup_streak.get('required')}, current={startup_streak.get('current_streak')}, "
        f"evaluated={startup_streak.get('evaluated_runs')})",
        f"2. Payload contract checks: {'PASS' if payload_contract.get('ok') else 'FAIL'} "
        f"(present={payload_contract.get('present')}, reason={payload_contract.get('reason')})",
        f"3. Benchmark SLA policy: {'PASS' if benchmark_sla.get('ok') else 'FAIL'} "
        f"(reason={benchmark_sla.get('reason')}, warn_exception_used={warn_exception_used})",
        f"4. Endpoint error-rate policy: {'PASS' if endpoint_policy.get('ok') else 'FAIL'} "
        f"(threshold_pct={endpoint_policy.get('threshold_pct')}, breaches={endpoint_policy.get('breaches')})",
        f"5. Nightly retraining trigger policy review: {'PASS' if retrain_review.get('ok') else 'FAIL'} "
        f"(reason={retrain_review.get('reason')}, requires_ack={retrain_review.get('requires_ack')})",
        f"6. Inference regression gate: {'PASS' if regression_gate.get('ok') else 'FAIL'} "
        f"(reason={regression_gate.get('reason')})",
        "",
        "## Promotion Controls",
        f"- Backend API contracts frozen: {'YES' if freeze_backend_contracts else 'NO'}",
        f"- Frontend integration policy: {frontend_action}",
        "",
        "## Evidence",
        f"- Gate result JSON: `{json.dumps(gate_result, sort_keys=True)}`",
        f"- Benchmark failing endpoints: `{benchmark_sla.get('failing_endpoints', [])}`",
        f"- Retraining review details: `{retrain_review.get('details', {})}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.ops_summary_json.exists():
        result = {"status": "fail", "reason": "missing_summary", "path": str(args.ops_summary_json)}
        print(json.dumps(result))
        return 1

    payload = _load_json(args.ops_summary_json)
    status = payload.get("status")

    failures: list[dict[str, Any]] = []

    if status != "ok":
        failures.append({"reason": "nightly_status_not_ok", "nightly_status": status})

    artifact_validation = payload.get("artifact_validation_report")
    if not isinstance(artifact_validation, dict) or not artifact_validation.get("ok", False):
        failures.append(
            {
                "reason": "artifact_validation_failed",
                "failed_checks": (artifact_validation or {}).get("failed_checks", []),
            }
        )

    startup_streak = evaluate_service_startup_streak(payload, args.archive_runs_dir, max(1, args.required_startup_streak))
    if not startup_streak.get("ok", False):
        failures.append(
            {
                "reason": "startup_streak_failed",
                "required": startup_streak.get("required"),
                "current_streak": startup_streak.get("current_streak"),
                "evaluated_runs": startup_streak.get("evaluated_runs"),
            }
        )

    payload_contract = evaluate_payload_contract_tests(payload)
    if args.require_payload_contract_tests and not payload_contract.get("present", False):
        failures.append({"reason": payload_contract.get("reason")})
    elif args.require_payload_contract_tests and not payload_contract.get("ok", False):
        failures.append(
            {
                "reason": payload_contract.get("reason"),
                "failing_suites": payload_contract.get("failing_suites", []),
            }
        )

    benchmark_sla = evaluate_benchmark_sla(payload)
    benchmark_sla_mode = str(payload.get("benchmark_sla_mode", "enforce")).lower()
    if benchmark_sla_mode != "enforce":
        failures.append(
            {
                "reason": "benchmark_sla_mode_not_enforce",
                "benchmark_sla_mode": benchmark_sla_mode,
            }
        )
    warn_exception_used = False
    if not benchmark_sla.get("ok", False):
        if benchmark_sla_mode == "warn" and args.warn_mode_exception_approval.strip():
            warn_exception_used = True
        else:
            failures.append(
                {
                    "reason": benchmark_sla.get("reason"),
                    "failing_endpoints": benchmark_sla.get("failing_endpoints", []),
                    "benchmark_mode": benchmark_sla.get("benchmark_mode"),
                    "failure_mode": benchmark_sla.get("failure_mode"),
                    "benchmark_sla_mode": benchmark_sla_mode,
                    "warn_mode_exception_required": benchmark_sla_mode == "warn",
                }
            )

    uncategorized_endpoints = benchmark_sla.get("uncategorized_endpoints") or []
    if uncategorized_endpoints:
        failures.append(
            {
                "reason": "uncategorized_errors_detected",
                "failing_endpoints": uncategorized_endpoints,
                "benchmark_mode": benchmark_sla.get("benchmark_mode"),
                "failure_mode": benchmark_sla.get("failure_mode"),
            }
        )

    endpoint_policy = evaluate_endpoint_error_policy(benchmark_sla, args.max_endpoint_error_rate_pct)
    if not endpoint_policy.get("ok", False):
        failures.append(
            {
                "reason": endpoint_policy.get("reason"),
                "threshold_pct": endpoint_policy.get("threshold_pct"),
                "breaches": endpoint_policy.get("breaches", []),
            }
        )

    retrain_review = evaluate_retrain_policy_review(payload, args.approve_retrain_policy_review)
    if not retrain_review.get("ok", False):
        failures.append(
            {
                "reason": retrain_review.get("reason"),
                "requires_ack": retrain_review.get("requires_ack", False),
                "missing": retrain_review.get("missing", []),
                "reasons": retrain_review.get("reasons", []),
            }
        )

    regression_gate = evaluate_inference_regression_gate(args.inference_candidate_report_json)
    if not regression_gate.get("ok", False):
        failures.append(
            {
                "reason": regression_gate.get("reason"),
                "inference_candidate_report": regression_gate.get("path"),
                "details": regression_gate.get("details"),
            }
        )

    gate_ok = len(failures) == 0
    stable_endpoints = _resolve_stable_endpoints(benchmark_sla)
    gate_result = {
        "status": "ok" if gate_ok else "fail",
        "nightly_status": status,
        "path": str(args.ops_summary_json),
        "checked_at_utc": _utc_now_iso(),
        "startup_streak": startup_streak,
        "benchmark_sla_mode": benchmark_sla_mode,
        "warn_mode_exception_used": warn_exception_used,
        "stable_endpoints": stable_endpoints,
        "api_contract_state": "frozen" if gate_ok else "mutable",
        "frontend_integration_allowed": gate_ok,
        "frontend_integration_policy": (
            "stable_endpoints_only" if gate_ok else "blocked_until_gate_pass"
        ),
        "failures": failures,
    }

    write_backend_readiness_report(
        report_path=args.backend_readiness_report,
        gate_result=gate_result,
        startup_streak=startup_streak,
        payload_contract=payload_contract,
        benchmark_sla=benchmark_sla,
        endpoint_policy=endpoint_policy,
        retrain_review=retrain_review,
        regression_gate=regression_gate,
        freeze_backend_contracts=gate_ok,
        stable_endpoints=stable_endpoints,
        warn_exception_used=warn_exception_used,
    )

    print(json.dumps(gate_result))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
