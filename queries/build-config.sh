#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
jq --rawfile validation queries/examples/validation.sql --rawfile training queries/examples/training.sql '.config.queries = {validation: $validation, training: $training}' sample-configs.json
