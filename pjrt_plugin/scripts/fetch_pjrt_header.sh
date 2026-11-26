#!/usr/bin/env bash
set -euo pipefail

# Fetches xla/pjrt/c/pjrt_c_api.h from openxla/xla into pjrt_plugin/third_party/pjrt_c_api.h.
# Optionally pass a git ref / commit SHA matching your installed jaxlib, e.g.:
#   ./fetch_pjrt_header.sh main

REF="${1:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/../third_party/pjrt_c_api.h"

mkdir -p "$(dirname "${DEST}")"
URL="https://raw.githubusercontent.com/openxla/xla/${REF}/xla/pjrt/c/pjrt_c_api.h"
echo "Fetching ${URL} -> ${DEST}"
curl -fsSL "${URL}" -o "${DEST}"
grep -E "PJRT_API_MAJOR|PJRT_API_MINOR" "${DEST}" | head -n 4
