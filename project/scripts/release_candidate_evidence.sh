#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <release_tag>"
  exit 2
fi

RELEASE_TAG="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python project/scripts/build_evidence_bundle.py \
  --bundle-name "evidence_bundle_${RELEASE_TAG}" \
  --release-tag "$RELEASE_TAG" \
  --require-complete
