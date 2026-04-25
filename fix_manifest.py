import json
from pathlib import Path

manifest_path = Path("project/models/promoted_artifact_manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

print("Before:")
print(json.dumps(manifest, indent=2))

def rewrite(obj):
    if isinstance(obj, dict):
        return {k: rewrite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(x) for x in obj]
    if isinstance(obj, str):
        normalized = obj.replace("\\", "/")
        if ("Vhack" in normalized) or normalized.startswith("D:") or ("/workspace/" in normalized):
            return f"/app/project/models/{Path(normalized).name}"
    return obj

fixed = rewrite(manifest)

print("\nAfter:")
print(json.dumps(fixed, indent=2))

manifest_path.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")
print("\nManifest fixed and saved.")