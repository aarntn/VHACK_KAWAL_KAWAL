import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "aarntn/kawal-models"
MODEL_DIR = Path("/app/project/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "final_xgboost_model_promoted_preproc.pkl",
    "feature_columns_promoted_preproc.pkl",
    "decision_thresholds_promoted_preproc.pkl",
    "preprocessing_artifact_promoted.pkl",
    "promoted_artifact_manifest.json",
]

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN is required for private repo download.")

for filename in FILES:
    dest = MODEL_DIR / filename
    if dest.exists():
        print(f"Already exists: {filename}")
        continue

    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
        token=token,
        local_dir=str(MODEL_DIR),
    )
    print(f"Done: {filename}")

print("All artifacts ready.")