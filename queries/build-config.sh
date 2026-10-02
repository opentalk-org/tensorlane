#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
jq --rawfile validation queries/validation.sql --rawfile training queries/training.sql '.config.queries = {validation: $validation, training: $training}' "${1:-sample-configs.json}"
