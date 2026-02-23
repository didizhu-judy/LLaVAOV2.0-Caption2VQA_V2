#!/bin/bash
set -euo pipefail
BACKEND_PROFILE=azure_multi bash "$(cd "$(dirname "$0")" && pwd)/run_clean_pipeline.sh" "$@"
