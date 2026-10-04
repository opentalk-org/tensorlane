#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
jq --rawfile validation queries/validation.sql --rawfile training queries/training.sql '.config.queries |= map(if .key == "training" then .sql = $training elif .key == "validation" then .sql = $validation else . end)' "${1:-queries/examples/sample-configs.json}"
