# Fraud Decisioning Architecture, Calibration Rationale, and Release Gates

## 1) Architecture update (March 18, 2026)

The runtime now includes an explicit **human-in-the-loop review track** for high-risk and borderline decisions.

### Serving flow
1. `/score_transaction` computes model score + context + behavior adjustments.
2. A privacy-safe audit record is written to `outputs/audit/fraud_audit_log.jsonl`.
3. Decisions that are high-risk or borderline are automatically queued into `outputs/audit/review_queue.jsonl`.
4. Analysts submit outcomes via `/review_queue/{request_id}/outcome`.
5. Outcomes are persisted to `outputs/audit/analyst_outcomes.jsonl` and transformed into curated retraining examples at `outputs/audit/retraining_curation.jsonl`.
6. Monitoring dashboards are served by `/dashboard/views`.

### New operational endpoints
- `GET /review_queue?status=pending|resolved|all`
- `POST /review_queue/{request_id}/outcome`
- `GET /retraining/curation`
- `GET /dashboard/views?window_hours=24`

## 2) Calibration rationale

The calibration strategy remains **progressive-friction first**:

- `APPROVE`: low-risk payments proceed with minimal friction.
- `FLAG`: medium-risk and low-history uncertainty route to verification/manual checks.
- `BLOCK`: high-confidence risk receives hard stop to prevent immediate loss.

### Why a review queue was added
- Borderline and high-risk predictions represent the highest leverage area for reducing false positives while preserving fraud catch-rate.
- Analyst labels from these cases produce high-information training examples for threshold refresh and retraining set curation.

### Borderline policy
A case is queued for manual review when:
- decision is `FLAG` or `BLOCK`, or
- score is within configurable margin (`FRAUD_REVIEW_BORDERLINE_MARGIN`, default `0.03`) of approve threshold.

## 3) Release-gate decisions

Release gate policy for this iteration:

1. **Data integrity gate**
   - Audit logging, review queue writes, analyst outcome writes, and retraining curation writes must all succeed in integration tests.
2. **Safety gate**
   - Review queue must capture all `BLOCK` and `FLAG` decisions.
3. **Monitoring gate**
   - Dashboard endpoint must expose:
     - latency/throughput/error view,
     - drift/score distribution view,
     - fraud loss/false positives/analyst agreement view.
4. **Model iteration gate**
   - No retraining promotion without analyst-curated examples and review sign-off.

## 4) Risk and mitigation notes

- If analyst outcomes are sparse, dashboard agreement metrics may be noisy; monitor sample size before taking threshold actions.
- Fraud-loss estimate is currently outcome-amount based and should be upgraded with chargeback/confirmed-loss joins in production systems.

