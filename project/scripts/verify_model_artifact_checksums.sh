#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKSUM_FILE="project/models/artifact_checksums.sha256"
if [[ ! -f "$CHECKSUM_FILE" ]]; then
  echo "Checksum file missing: $CHECKSUM_FILE"
  exit 1
fi

sha256sum --check "$CHECKSUM_FILE"
