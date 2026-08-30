#!/usr/bin/env bash
# Fine-tune z-lab DFlash2 on bridged Freaksterz W8A8 hidden-state traces.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPECULATORS_DIR=${SPECULATORS_DIR:-"$ROOT/../speculators"}
PYTHON=${PYTHON:-"$SPECULATORS_DIR/.venv/bin/python"}
TORCHRUN=${TORCHRUN:-"$SPECULATORS_DIR/.venv/bin/torchrun"}
FREAKSTERZ_DIR=${FREAKSTERZ_DIR:?Set FREAKSTERZ_DIR to the downloaded Freaksterz main snapshot.}
ZLAB_DRAFT=${ZLAB_DRAFT:-z-lab/Qwen3.8-27B-DFlash2}
ADAPTER=${ADAPTER:-"$ROOT/artifacts/freaksterz-dflash-adapter.safetensors"}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to the completed collection run directory.}
# The converted config embeds the native-basis training-verifier path. Keep it
# per run so a later run never silently reuses another run's frozen verifier.
CONVERTED_DRAFT=${CONVERTED_DRAFT:-"$RUN_DIR/zlab-qwen38-27b-dflash2-speculators"}
CONVERT_OVERWRITE=${CONVERT_OVERWRITE:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPUS=${GPUS:-4}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
# DFlash2 materializes full-vocabulary teacher targets and draft logits for
# ``MAX_ANCHORS * block_size`` tokens.  At Qwen3.8's 248,320-token vocabulary,
# the upstream default of 512 anchors produces two ~2 GiB BF16 tensors on each
# GPU.  32 anchors is the verified default for four 24 GiB GPUs with FSDP;
# increase it only when there is measured memory headroom.
MAX_ANCHORS=${MAX_ANCHORS:-32}
# Hidden-state samples are large.  The upstream defaults (12 workers and four
# prefetched samples *per rank*) can retain hundreds of GiB before the first
# batch reaches the GPU on a four-rank run.
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
MAX_STEPS=${MAX_STEPS:-}
FSDP_SHARD=${FSDP_SHARD:-1}
FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD:-0}
MODEL_COMPILE=${MODEL_COMPILE:-0}
REBUILD_TRAINING_VERIFIER=${REBUILD_TRAINING_VERIFIER:-0}
SAVE_PATH=${SAVE_PATH:-"$RUN_DIR/checkpoints"}

IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne $GPUS ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${#visible_gpus[@]} GPUs, but GPUS=$GPUS" >&2
  exit 2
fi
if [[ $GPUS -gt 1 && "$FSDP_SHARD" != "1" ]]; then
  echo "Multi-GPU DFlash2 training requires FSDP_SHARD=1 (DDP does not fit per GPU)." >&2
  exit 2
fi
for required in "$PYTHON" "$TORCHRUN"; do
  if [[ ! -x "$required" ]]; then
    echo "Required executable not found: $required" >&2
    exit 2
  fi
done
for required in "$FREAKSTERZ_DIR" "$RUN_DIR/data" "$RUN_DIR/hidden-states"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input not found: $required" >&2
    exit 2
  fi
done

# Never inherit yesterday's experimental settings silently.  These values are
# resolved by this wrapper and recorded in the log for every launch.
export CUDA_VISIBLE_DEVICES
export SPECULATORS_FSDP_CPU_OFFLOAD="$FSDP_CPU_OFFLOAD"
export SPECULATORS_TORCH_COMPILE="$MODEL_COMPILE"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/training-verifier" "$SAVE_PATH"
exec > >(tee -a "$RUN_DIR/logs/train.log") 2>&1

printf 'Training launch: CUDA_VISIBLE_DEVICES=%s GPUS=%s FSDP_SHARD=%s FSDP_CPU_OFFLOAD=%s MODEL_COMPILE=%s SEQ_LENGTH=%s MAX_ANCHORS=%s NUM_WORKERS=%s SAVE_PATH=%s\n' \
  "$CUDA_VISIBLE_DEVICES" "$GPUS" "$FSDP_SHARD" "$FSDP_CPU_OFFLOAD" "$MODEL_COMPILE" "$SEQ_LENGTH" "$MAX_ANCHORS" "$NUM_WORKERS" "$SAVE_PATH"

if [[ ! -f "$RUN_DIR/training-verifier/model.safetensors" || "$REBUILD_TRAINING_VERIFIER" == "1" ]]; then
  "$PYTHON" "$ROOT/scripts/make_training_verifier.py" --freaksterz "$FREAKSTERZ_DIR" --adapter "$ADAPTER" --output "$RUN_DIR/training-verifier"
else
  echo "Reusing existing training verifier: $RUN_DIR/training-verifier"
fi
if [[ ! -f "$CONVERTED_DRAFT/config.json" || "$CONVERT_OVERWRITE" == "1" ]]; then
  converter_args=(
    "$PYTHON" "$ROOT/scripts/convert_zlab_dflash2_to_speculators.py"
    --source "$ZLAB_DRAFT" --output "$CONVERTED_DRAFT"
    --verifier "$RUN_DIR/training-verifier" --local-files-only
  )
  if [[ "$CONVERT_OVERWRITE" == "1" ]]; then
    converter_args+=(--overwrite)
  fi
  "${converter_args[@]}"
fi
train_args=(
  "$TORCHRUN" --standalone --nproc_per_node "$GPUS" "$SPECULATORS_DIR/scripts/train.py"
  --verifier-name-or-path "$RUN_DIR/training-verifier"
  --from-pretrained "$CONVERTED_DRAFT"
  --data-path "$RUN_DIR/data" --hidden-states-path "$RUN_DIR/hidden-states" --on-missing raise
  --save-path "$SAVE_PATH" --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH"
  --max-anchors "$MAX_ANCHORS"
  --num-workers "$NUM_WORKERS"
  --speculator-type dflash2
  --loss-fn kl_div --per-position-loss-weight fixed-exp-decay --run-name freaksterz-dflash2
)
if [[ "$NUM_WORKERS" -gt 0 ]]; then
  train_args+=(--prefetch-factor "$PREFETCH_FACTOR")
fi
if [[ -n "$MAX_STEPS" ]]; then
  train_args+=(--max-steps "$MAX_STEPS")
fi
if [[ "$FSDP_SHARD" == "1" ]]; then
  train_args+=(--fsdp-shard)
fi
printf 'Executing:'
printf ' %q' "${train_args[@]}"
printf '\n'
"${train_args[@]}"
