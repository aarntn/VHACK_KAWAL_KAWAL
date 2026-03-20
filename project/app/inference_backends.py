from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class InferenceBackend(Protocol):
    backend_name: str

    def predict_positive_proba(self, model: Any, features: np.ndarray) -> np.ndarray:
        ...


def _as_numeric_matrix(features: Any) -> np.ndarray:
    """Normalize model input into a finite, 2D float32 matrix."""
    if hasattr(features, "to_numpy"):
        matrix = np.asarray(features.to_numpy(dtype=np.float32), dtype=np.float32)
    else:
        matrix = np.asarray(features, dtype=np.float32)

    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    elif matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2:
        raise RuntimeError(f"Expected 2D feature matrix, got shape {matrix.shape}")

    if not np.isfinite(matrix).all():
        raise RuntimeError("Feature matrix contains NaN/inf values")

    return np.ascontiguousarray(matrix, dtype=np.float32)


@dataclass
class PredictProbaBackend:
    backend_name: str = "predict_proba"

    def predict_positive_proba(self, model: Any, features: np.ndarray) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Selected backend requires model.predict_proba")

        probs = np.asarray(model.predict_proba(_as_numeric_matrix(features)), dtype=np.float64)
        if probs.ndim == 1:
            return probs
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise RuntimeError(
                f"predict_proba returned unexpected shape {probs.shape}; expected [n] or [n, >=2]"
            )
        return probs[:, 1]


@dataclass
class XGBoostInplacePredictBackend:
    backend_name: str = "xgboost_inplace_predict"

    def predict_positive_proba(self, model: Any, features: np.ndarray) -> np.ndarray:
        if not hasattr(model, "get_booster"):
            raise RuntimeError("xgboost_inplace_predict backend requires XGBClassifier-like model.get_booster()")

        booster = model.get_booster()
        if not hasattr(booster, "inplace_predict"):
            raise RuntimeError("xgboost booster does not support inplace_predict")

        data = _as_numeric_matrix(features)
        raw = np.asarray(booster.inplace_predict(data, validate_features=False), dtype=np.float64)
        if raw.ndim == 1:
            return raw
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return raw[:, 1]
        raise RuntimeError(f"inplace_predict returned unexpected shape {raw.shape}")


class OnnxOrHummingbirdBackend:
    backend_name: str = "onnx_hummingbird"

    def __init__(self) -> None:
        self._runtime_model: Any | None = None
        self._runtime_name: str | None = None
        self._bound_model_id: int | None = None

    def _build_runtime(self, model: Any, features: np.ndarray) -> tuple[Any, str]:
        feature_count = int(features.shape[1])

        # Prefer ONNX Runtime when available.
        try:
            import onnxruntime as ort
            from onnxmltools import convert_xgboost
            from skl2onnx.common.data_types import FloatTensorType

            onx = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, feature_count]))])
            session = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])

            class _OnnxWrapper:
                def __init__(self, sess: Any):
                    self._sess = sess
                    self._input_name = sess.get_inputs()[0].name

                def predict_positive_proba(self, x: np.ndarray) -> np.ndarray:
                    out = self._sess.run(None, {self._input_name: np.asarray(x, dtype=np.float32)})
                    if len(out) >= 2:
                        probs = np.asarray(out[1], dtype=np.float64)
                        if probs.ndim == 2 and probs.shape[1] >= 2:
                            return probs[:, 1]
                    flat = np.asarray(out[0], dtype=np.float64).reshape(-1)
                    return flat

            return _OnnxWrapper(session), "onnxruntime"
        except Exception:
            pass

        try:
            from hummingbird.ml import convert

            hb_model = convert(model, "torch")

            class _HummingbirdWrapper:
                def __init__(self, runtime_model: Any):
                    self._runtime_model = runtime_model

                def predict_positive_proba(self, x: np.ndarray) -> np.ndarray:
                    probs = np.asarray(self._runtime_model.predict_proba(np.asarray(x, dtype=np.float32)))
                    if probs.ndim == 1:
                        return probs.astype(np.float64)
                    if probs.ndim == 2 and probs.shape[1] >= 2:
                        return probs[:, 1].astype(np.float64)
                    raise RuntimeError(f"hummingbird predict_proba returned unexpected shape {probs.shape}")

            return _HummingbirdWrapper(hb_model), "hummingbird"
        except Exception as exc:
            raise RuntimeError(
                "onnx_hummingbird backend requires ONNX Runtime (onnxruntime/onnxmltools/skl2onnx) "
                "or Hummingbird (hummingbird-ml + torch) dependencies"
            ) from exc

    def predict_positive_proba(self, model: Any, features: np.ndarray) -> np.ndarray:
        matrix = _as_numeric_matrix(features)
        if self._runtime_model is None or self._bound_model_id != id(model):
            self._runtime_model, self._runtime_name = self._build_runtime(model, matrix)
            self._bound_model_id = id(model)

        probs = self._runtime_model.predict_positive_proba(matrix)
        return np.asarray(probs, dtype=np.float64).reshape(-1)

    @property
    def runtime_name(self) -> str:
        return self._runtime_name or "uninitialized"


def create_inference_backend(name: str) -> InferenceBackend:
    normalized = (name or "xgboost_inplace_predict").strip().lower()
    aliases = {
        "sklearn_predict_proba": "predict_proba",
        "sklearn": "predict_proba",
        "xgboost": "xgboost_inplace_predict",
        "xgboost_inplace": "xgboost_inplace_predict",
        "inplace_predict": "xgboost_inplace_predict",
        "onnx": "onnx_hummingbird",
        "hummingbird": "onnx_hummingbird",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized == "predict_proba":
        return PredictProbaBackend()
    if normalized == "xgboost_inplace_predict":
        return XGBoostInplacePredictBackend()
    if normalized == "onnx_hummingbird":
        return OnnxOrHummingbirdBackend()

    supported = ["predict_proba", "xgboost_inplace_predict", "onnx_hummingbird"]
    raise ValueError(f"Unsupported FRAUD_INFERENCE_BACKEND '{name}'. Supported backends: {supported}")
