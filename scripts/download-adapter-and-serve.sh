#!/usr/bin/env bash
set -euo pipefail

adapter_repo=${FREAKSTERZ_DFLASH_ADAPTER_REPO:-halofx/freaksterz-dflash2-vllm-adapter}
adapter_filename=${FREAKSTERZ_DFLASH_ADAPTER_FILENAME:-freaksterz-dflash-adapter.safetensors}

adapter_path=$(python3 - "$adapter_repo" "$adapter_filename" <<'PY'
import sys
from huggingface_hub import hf_hub_download

print(hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2]))
PY
)

export FREAKSTERZ_DFLASH_ADAPTER="$adapter_path"
exec vllm "$@"
