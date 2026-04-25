from pathlib import Path
import os
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "aarntn/kawal-models"
REPO_TYPE = "model"
PRIVATE_REPO = True

FILES = [
    "project/models/final_xgboost_model_promoted_preproc.pkl",
    "project/models/feature_columns_promoted_preproc.pkl",
    "project/models/decision_thresholds_promoted_preproc.pkl",
    "project/models/preprocessing_artifact_promoted.pkl",
    "project/models/promoted_artifact_manifest.json",
]

def main() -> int:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set.")
        print('PowerShell: $env:HF_TOKEN="hf_your_token_here"')
        return 1

    api = HfApi(token=HF_TOKEN)

    # Ensure repo exists
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type=REPO_TYPE)
        print(f"Repo exists: {HF_REPO_ID}")
    except HfHubHTTPError:
        print(f"Repo not found. Creating private repo: {HF_REPO_ID}")
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
            private=PRIVATE_REPO,
            exist_ok=True,
        )

    # Validate files
    missing = [f for f in FILES if not Path(f).exists()]
    if missing:
        print("ERROR: Missing files:")
        for m in missing:
            print(f" - {m}")
        return 1

    # Upload
    for rel in FILES:
        path = Path(rel)
        print(f"Uploading {path.name} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=HF_REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"Done: {path.name}")

    print("\nAll files uploaded successfully.")
    print(f"https://huggingface.co/{HF_REPO_ID}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())