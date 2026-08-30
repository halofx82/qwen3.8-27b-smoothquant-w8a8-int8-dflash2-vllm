#!/usr/bin/env bash
# Extract W8A8 teacher states, bridge them, and log every stage.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPECULATORS_DIR=${SPECULATORS_DIR:-"$ROOT/../speculators"}
PYTHON=${PYTHON:-"$SPECULATORS_DIR/.venv/bin/python"}

# One endpoint only. Both forms are accepted:
#   ENDPOINT=http://127.0.0.1:8001
#   ENDPOINT=http://127.0.0.1:8001/v1
# Internally, prepare_data needs the server root while
# data_generation_offline needs the OpenAI /v1 base URL.
ENDPOINT=${ENDPOINT:-${EXTRACT_ENDPOINT:-http://127.0.0.1:8001}}
ENDPOINT=${ENDPOINT%/}
BASE_ENDPOINT=${ENDPOINT%/v1}
API_ENDPOINT="$BASE_ENDPOINT/v1"

FREAKSTERZ_MODEL=${FREAKSTERZ_MODEL:-Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8}
ADAPTER=${ADAPTER:-"$ROOT/artifacts/freaksterz-dflash-adapter.safetensors"}
DATASET=${DATASET:?Set DATASET to a Speculators-supported dataset or local conversation dataset.}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/freaksterz-dflash2-$(date -u +%Y%m%dT%H%M%SZ)"}
MAX_SAMPLES=${MAX_SAMPLES:-5000}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
CONCURRENCY=${CONCURRENCY:-4}

mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/logs/collect.log") 2>&1

echo "Endpoint:       $BASE_ENDPOINT"
echo "OpenAI API:     $API_ENDPOINT"
echo "Dataset:        $DATASET"
echo "Run dir:        $RUN_DIR"
echo "Max samples:    $MAX_SAMPLES"
echo "Sequence length:$SEQ_LENGTH"

"$PYTHON" "$SPECULATORS_DIR/scripts/prepare_data.py" \
  --model "$FREAKSTERZ_MODEL" \
  --data "$DATASET" \
  --output "$RUN_DIR/data" \
  --max-samples "$MAX_SAMPLES" \
  --seq-length "$SEQ_LENGTH" \
  --render-endpoint "$BASE_ENDPOINT"

"$PYTHON" "$SPECULATORS_DIR/scripts/data_generation_offline.py" \
  --endpoint "$API_ENDPOINT" \
  --model "$FREAKSTERZ_MODEL" \
  --preprocessed-data "$RUN_DIR/data" \
  --output "$RUN_DIR/raw-hidden-states" \
  --max-samples "$MAX_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --validate-outputs

"$PYTHON" "$ROOT/scripts/bridge_hidden_states.py" \
  --input "$RUN_DIR/raw-hidden-states" \
  --output "$RUN_DIR/hidden-states" \
  --adapter "$ADAPTER" \
  --report "$RUN_DIR/bridge-report.json"
