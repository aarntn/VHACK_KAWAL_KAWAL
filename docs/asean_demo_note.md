# ASEAN Demo Note

This note is the demo-specific reference for ASEAN runtime behavior. It describes what is already live in the codebase versus what remains on the roadmap.

## Canonical demo scenarios

- `ID-ID` domestic QR merchant payment in `IDR`
  - expected outcome: `APPROVE`
  - story: normal local QR payment with low device/network risk
- `SG-PH` first-time remittance in `SGD`
  - expected outcome: `FLAG` plus wallet `PENDING_VERIFICATION`
  - story: new cross-border corridor triggers step-up rather than silent approval
- `MY-MY` agent-assisted cash-out in `MYR` with intermittent connectivity
  - expected outcome: `FLAG`
  - story: agent-assisted withdrawal and unstable connectivity trigger conservative review
- `TH-VN` suspicious repeated transfer pattern in `THB`
  - expected outcome: `BLOCK`
  - story: new-wallet, high-risk corridor behavior lands in block / hard-rule territory

## Live runtime additions

- Supported ASEAN demo countries: `SG`, `MY`, `ID`, `TH`, `PH`, `VN`
- Static currency normalization snapshot:
  - file: `project/data/asean_currency_normalization.json`
  - artifact version: `asean_currency_normalization_v1`
  - basis: `static_demo_reference_snapshot_2026q2`
- Response provenance now includes:
  - `corridor`
  - `normalized_amount_reference`
  - `normalization_basis`
  - `runtime_mode`
  - ASEAN-specific `reason_codes`

## Runtime mode meanings

- `primary`: full live scoring path with normal context/signal access
- `cached_context`: scoring used cached feature/context sources
- `degraded_local`: live upstream signals were unavailable or the request explicitly used `offline_buffered`

## Honest boundaries

- The checked-in normalization artifact is intentionally static for demo stability and explainability.
- Real-time FX, embedded offline ONNX scoring, and country-specific regulatory adapters remain roadmap items.
- ASEAN fairness evidence is still grounded in current governance artifacts plus threshold mitigation, not in a fully retrained region-specific model.
