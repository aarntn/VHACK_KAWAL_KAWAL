# External and internal fraud payload contracts

This document defines the canonical **external** request schema and clarifies how it is transformed into the internal IEEE-style serving shape.

## Endpoint-to-schema mapping (canonical)

### `/score_transaction` (fraud engine)
- **Expected request model:** `ScoreTransactionRequest` (`project/app/schemas/api.py`)
- **Canonical external amount field:** `transaction_amount`
- **Compatibility alias accepted:** `TransactionAmt` (mapped to `transaction_amount` at validation time)
- **Not expected from clients:** `TransactionDT`, `V1..V17`
- **ASEAN runtime context fields:** optional `currency`, `source_country`, `destination_country`, `is_agent_assisted`, `connectivity_mode`

### `/wallet/authorize_payment` (wallet gateway)
- **Expected request model:** `WalletAuthorizeRequest` (`project/app/schemas/api.py`)
- Includes the same fraud fields as `ScoreTransactionRequest`, plus:
  - `wallet_id`
  - `merchant_name`
  - `currency`
- Wallet forwarding to `/score_transaction` uses the same canonical fraud field names (`transaction_amount`, risk/context fields).

## Canonical external payload example (`/score_transaction`)

```json
{
  "schema_version": "ieee_fraud_tx_v1",
  "user_id": "frontend_demo_user_001",
  "transaction_amount": 72.5,
  "device_risk_score": 0.1,
  "ip_risk_score": 0.05,
  "location_risk_score": 0.05,
  "device_id": "device-abc-123",
  "device_shared_users_24h": 1,
  "account_age_days": 180,
  "sim_change_recent": false,
  "tx_type": "MERCHANT",
  "channel": "APP",
  "cash_flow_velocity_1h": 1,
  "p2p_counterparties_24h": 0,
  "is_cross_border": false,
  "currency": "SGD",
  "source_country": "SG",
  "destination_country": "SG",
  "is_agent_assisted": false,
  "connectivity_mode": "online"
}
```

## Internal serving contract (IEEE-style)

`FraudTransactionContract` in `project/app/contracts.py` is **internal-only** and represents model-serving feature names (`TransactionDT`, `TransactionAmt`, `V1..V17`). The fraud API transforms canonical external requests into this shape before feature construction/inference.

## Architecture note: public schema -> internal IEEE features

The external request body and internal serving payload intentionally differ:

| Public/API field (external) | Internal serving field(s) | Mapping behavior |
| --- | --- | --- |
| `transaction_amount` | `TransactionAmt` | Direct rename from public schema into internal contract. |
| `currency` | `TransactionAmt` + ASEAN provenance fields | Used with the checked-in normalization artifact to derive a shared model reference amount. |
| `source_country`, `destination_country` | `source_country`, `destination_country`, `corridor` | Validated as ISO 3166-1 alpha-2 within the supported ASEAN demo set, then corridor is derived server-side. |
| `is_agent_assisted`, `connectivity_mode` | Same field names in serving payload | Drives corridor/channel-aware context adjustments, reasons, and runtime provenance. |
| Context fields (`device_risk_score`, `ip_risk_score`, `location_risk_score`, etc.) | Same field names in serving payload | Passed through with validation/defaults. |
| *(not provided by frontend)* | `TransactionDT` | Derived server-side from current request timestamp. |
| *(not provided by frontend)* | `V1..V17` | Derived/imputed internally for serving compatibility (not a public caller responsibility). |

Implementation anchors:
- External normalization: `normalize_external_score_transaction_request` in `project/app/hybrid_fraud_api.py`
- Mapping into serving shape: `build_model_features_from_normalized_request` in `project/app/hybrid_fraud_api.py`
- Internal contract: `FraudTransactionContract` in `project/app/contracts.py`

## Field-name compatibility rules

- External callers should send canonical names from `ScoreTransactionRequest` / `WalletAuthorizeRequest`.
- `TransactionAmt` is only a backwards-compatibility alias for `transaction_amount`.
- Legacy names like `Time` or `Amount` are invalid.

## Preset scenario source of truth (UI + contract checks)

To keep demo behavior and contract assertions aligned, both the frontend scenario selector and `preset_contract_check.py` read from a single shared definition file:

- `project/frontend/src/scenarioPresets.json`

This file includes, per scenario:
- UI defaults (amount, tx type, cross-border flag, wallet/merchant defaults)
- ASEAN-local corridor/runtime defaults (currency, source/destination country, agent-assisted flow, connectivity mode)
- Elevated risk/context fields (device/ip/location risk, shared users, account age, velocity, counterparties, channel)

`UserApp.tsx` now forwards these risk fields when a scenario is submitted, and `project/scripts/preset_contract_check.py` loads the same file to build its preset payloads.
