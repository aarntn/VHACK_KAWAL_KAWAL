# DEPRECATED: Legacy fraud API retained for historical reference only.
# Canonical runtime services are project/app/hybrid_fraud_api.py and
# project/app/wallet_gateway_api.py.

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

from project.data.preprocessing import (
    load_preprocessing_bundle,
    prepare_preprocessing_inputs,
    transform_with_bundle,
)
from project.app.artifact_runtime_validator import (
    load_promoted_artifact_manifest,
    validate_promoted_artifacts,
    raise_on_failed_validation,
)

print("Loading model artifacts...")

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
PROMOTED_ARTIFACT_MANIFEST_FILE = Path(
    os.getenv(
        "FRAUD_ARTIFACT_MANIFEST_FILE",
        str(PROJECT_ROOT / "models" / "promoted_artifact_manifest.json"),
    )
)
promoted_manifest = load_promoted_artifact_manifest(PROMOTED_ARTIFACT_MANIFEST_FILE)

# =========================
# LOAD SAVED ARTIFACTS
# =========================
with open(promoted_manifest.model_file, "rb") as f:
    model = pickle.load(f)

with open(promoted_manifest.feature_file, "rb") as f:
    feature_columns = pickle.load(f)

with open(promoted_manifest.threshold_file, "rb") as f:
    thresholds = pickle.load(f)

preprocessing_bundle = load_preprocessing_bundle(promoted_manifest.preprocessing_bundle_file)
promoted_artifact_validation = validate_promoted_artifacts(
    promoted_manifest,
    feature_columns=feature_columns,
    preprocessing_bundle=preprocessing_bundle,
)
raise_on_failed_validation(promoted_artifact_validation)

APPROVE_THRESHOLD = thresholds["approve_threshold"]
BLOCK_THRESHOLD = thresholds["block_threshold"]

print("Model loaded successfully.")
print("Approve threshold:", APPROVE_THRESHOLD)
print("Block threshold:", BLOCK_THRESHOLD)

# =========================
# FASTAPI APP
# =========================
@asynccontextmanager
async def lifespan(_: FastAPI):
    startup_validation = validate_promoted_artifacts(
        promoted_manifest,
        feature_columns=feature_columns,
        preprocessing_bundle=preprocessing_bundle,
    )
    raise_on_failed_validation(startup_validation)
    yield


app = FastAPI(
    title="Fraud Risk API",
    description="Real-time fraud scoring engine for Approve / Flag / Block decisions",
    version="1.0",
    lifespan=lifespan,
)

# =========================
# INPUT SCHEMA
# =========================
class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# =========================
# HELPER FUNCTION
# =========================
def get_decision(risk_score: float) -> str:
    if risk_score < APPROVE_THRESHOLD:
        return "APPROVE"
    elif risk_score < BLOCK_THRESHOLD:
        return "FLAG"
    else:
        return "BLOCK"

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def root():
    return {
        "message": "Fraud Risk API is running",
        "approve_threshold": APPROVE_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD
    }

# =========================
# SCORE ENDPOINT
# =========================
@app.post("/score_transaction")
def score_transaction(tx: TransactionInput):
    input_df = pd.DataFrame([tx.model_dump()])
    canonical_df, passthrough_df, _ = prepare_preprocessing_inputs(
        input_df,
        preprocessing_bundle.dataset_source,
        usage_context="runtime_serving",
    )
    transformed = transform_with_bundle(preprocessing_bundle, canonical_df, passthrough_df)
    transformed_df = pd.DataFrame(transformed, columns=preprocessing_bundle.feature_names_out)

    # Ensure exact feature order
    input_df = transformed_df[feature_columns]

    risk_score = float(model.predict_proba(input_df)[0][1])
    decision = get_decision(risk_score)

    return {
        "risk_score": round(risk_score, 4),
        "decision": decision
    }


@app.get("/health/artifacts")
def health_artifacts():
    return {
        "status": "ok",
        "manifest_path": str(promoted_manifest.manifest_path),
        "model_metadata_family": promoted_artifact_validation.get("model_metadata_family"),
        "preprocessing_bundle_version": promoted_artifact_validation.get("preprocessing_bundle_version"),
        "feature_schema_hash": promoted_artifact_validation.get("feature_schema_hash"),
        "checks": promoted_artifact_validation.get("checks"),
    }
