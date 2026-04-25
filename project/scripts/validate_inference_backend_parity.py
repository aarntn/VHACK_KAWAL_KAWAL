import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.app.inference_backends import create_inference_backend
from project.data.preprocessing import prepare_preprocessing_inputs, load_preprocessing_bundle, transform_with_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate score parity between default predict_proba and alternative inference backends."
    )
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--preprocessing-bundle-file", type=Path, required=True)
    parser.add_argument(
        "--fixtures-file",
        type=Path,
        default=Path("project/tests/fixtures/fraud_payloads.json"),
        help="JSON fixture with sample payloads used to build a tiny parity batch.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["xgboost_inplace_predict", "onnx_hummingbird"],
        help="Backends to compare against predict_proba baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--latency-runs", type=int, default=400)
    return parser.parse_args()


def build_feature_matrix(fixtures_file: Path, feature_columns: list[str], bundle: object, batch_size: int) -> np.ndarray:
    with fixtures_file.open("r", encoding="utf-8") as handle:
        fixtures = json.load(handle)

    rows = list(fixtures.values())[: max(1, batch_size)]
    data = pd.DataFrame(rows)
    canonical_df, passthrough_df, _ = prepare_preprocessing_inputs(
        data,
        bundle.dataset_source,
        usage_context="runtime_serving",
    )
    transformed = transform_with_bundle(bundle, canonical_df, passthrough_df)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    index_map = {str(name): idx for idx, name in enumerate(bundle.feature_names_out)}
    ordered_indices = [index_map[name] for name in feature_columns]
    return np.take(dense, ordered_indices, axis=1).astype(np.float32, copy=False)


def measure_avg_latency_ms(backend: object, model: object, features: np.ndarray, runs: int) -> float:
    for _ in range(5):
        backend.predict_positive_proba(model, features)

    start = time.perf_counter()
    for _ in range(max(1, runs)):
        backend.predict_positive_proba(model, features)
    return ((time.perf_counter() - start) * 1000.0) / max(1, runs)


def main() -> int:
    args = parse_args()

    with args.model_file.open("rb") as handle:
        model = pickle.load(handle)
    with args.feature_file.open("rb") as handle:
        feature_columns = [str(col) for col in pickle.load(handle)]
    bundle = load_preprocessing_bundle(args.preprocessing_bundle_file)

    sample_features = build_feature_matrix(args.fixtures_file, feature_columns, bundle, args.batch_size)

    baseline = create_inference_backend("predict_proba")
    baseline_scores = baseline.predict_positive_proba(model, sample_features)
    baseline_latency = measure_avg_latency_ms(baseline, model, sample_features, args.latency_runs)

    all_ok = True
    print(f"Baseline backend=predict_proba scores={np.round(baseline_scores, 8).tolist()} avg_latency_ms={baseline_latency:.6f}")

    fastest_backend = "predict_proba"
    fastest_latency = baseline_latency

    for backend_name in args.backends:
        backend = create_inference_backend(backend_name)
        try:
            scores = backend.predict_positive_proba(model, sample_features)
            avg_latency = measure_avg_latency_ms(backend, model, sample_features, args.latency_runs)
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] backend={backend_name} could not run: {exc}")
            continue

        if avg_latency < fastest_latency:
            fastest_latency = avg_latency
            fastest_backend = backend_name

        max_abs_err = float(np.max(np.abs(scores - baseline_scores)))
        parity_ok = bool(np.allclose(scores, baseline_scores, atol=args.atol, rtol=0.0))
        status = "PASS" if parity_ok else "FAIL"
        delta_ms = avg_latency - baseline_latency
        print(
            f"[{status}] backend={backend_name} runtime={getattr(backend, 'runtime_name', backend.backend_name)} "
            f"max_abs_err={max_abs_err:.8g} avg_latency_ms={avg_latency:.6f} latency_delta_vs_predict_proba_ms={delta_ms:.6f} "
            f"scores={np.round(scores, 8).tolist()}"
        )
        all_ok = all_ok and parity_ok

    print(
        f"Recommendation: default backend should be '{fastest_backend}' "
        f"(avg_latency_ms={fastest_latency:.6f})"
    )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
