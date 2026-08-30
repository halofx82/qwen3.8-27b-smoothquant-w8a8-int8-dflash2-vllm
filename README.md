# Qwen3.8-27B SmoothQuant W8A8 INT8 + DFlash2 on vLLM

Run the rotated `main` revision of
[Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)
with DFlash2 speculative decoding on vLLM 0.28.0.

This project makes two native-basis DFlash2 checkpoints compatible with the
rotated target:

- [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2):
  original BF16 DFlash2; the default.
- [lued/Qwen3.8-27B-DFlash2-W8](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8):
  W8A16 quantization of the z-lab drafter, using less GPU memory.

The target uses W8A8 dynamic INT8 kernels, the draft uses BF16 compute, and
the Compose configuration uses TP=4 and an FP8 KV cache. Adjust the `command`
in `docker-compose.yaml` for a different topology or capacity target.

## Build and deploy

```bash
git clone git@github.com:halofx82/qwen3.8-27b-smoothquant-w8a8-int8-dflash2-vllm.git
cd qwen3.8-27b-smoothquant-w8a8-int8-dflash2-vllm
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f
```

The first launch downloads the selected public target, drafter, and the
[Freaksterz DFlash2 adapter](https://huggingface.co/halofx/freaksterz-dflash2-vllm-adapter)
into the standard host cache, `~/.cache/huggingface`, then compiles vLLM. It
can take several minutes. Later launches reuse both that cache and
`./.cache/vllm`.

The OpenAI-compatible endpoint is `http://127.0.0.1:8000/v1` and the served
model name is `freaksterz-qwen38-27b`.

To use the smaller Lued W8A16 draft, set this in `.env` and recreate the
service:

```bash
DRAFT_MODEL=lued/Qwen3.8-27B-DFlash2-W8
docker compose up -d --force-recreate
```

The default is the z-lab BF16 drafter:

```bash
DRAFT_MODEL=z-lab/Qwen3.8-27B-DFlash2
```

Stop the service with `docker compose down`. The repository intentionally does
not apply a custom P2P or all-reduce patch; networking behavior is stock vLLM.

## Fine-tune DFlash2 on Freaksterz traces

The included training path preserves the official Z-Lab DFlash2 weights.  It
does **not** train a new drafter from scratch:

```text
SWE-smith trajectory -> Freaksterz on-policy conversation
  -> Freaksterz auxiliary hidden states -> native-basis bridge
  -> lossless Z-Lab-to-Speculators conversion -> DFlash2 fine-tune
```

It requires a nearby checkout of
[vllm-project/speculators](https://github.com/vllm-project/speculators) at
`../speculators` (or set `SPECULATORS_DIR`), with its Python environment
installed. Apply the included training patch to that checkout before using the
launchers below. A minimal source installation is:

```bash
git clone https://github.com/vllm-project/speculators.git ../speculators
cd ../speculators
uv venv .venv
uv pip install --python .venv/bin/python -e .
cd -

# The DFlash2 large-trace training patch is required for this workflow.
git -C ../speculators apply --check "$PWD/patches/speculators-dflash2-training.patch"
git -C ../speculators apply "$PWD/patches/speculators-dflash2-training.patch"
```

The patch does only two functional things: it reads only the configured token
prefix from each hidden-state safetensors file rather than materializing an
entire long trace, and it makes full-model `torch.compile` opt-out controllable
through `SPECULATORS_TORCH_COMPILE`. The training launcher sets the latter to
`0`, avoiding compilation of an FSDP-wrapped draft model. It intentionally
does not include diagnostic logging or optional attention-kernel tuning.

You also need a local copy of `freaksterz-dflash-adapter.safetensors` and must
set `ADAPTER` to it. `FREAKSTERZ_DIR` below is the cached **`main`** snapshot of
`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`; do not use its unrotated
`v1-smoothquant-rtn` revision.

### 1. Generate the on-policy SWE-smith corpus

Start the normal Compose server on port 8000 and generate the conversations
first. Set `COLLECT_STATES=0`: this deliberately stops the helper after
generation, so the 4-GPU serving instance and the 4-GPU extractor never run at
the same time.

```bash
docker compose up -d

export RUN_DIR="$PWD/runs/swe-smith-freaksterz-$(date -u +%Y%m%dT%H%M%SZ)"
export ADAPTER="$PWD/artifacts/freaksterz-dflash-adapter.safetensors"

RUN_DIR="$RUN_DIR" \
MAX_SAMPLES=5000 \
SEQ_LENGTH=8192 \
CONCURRENCY=4 \
COLLECT_STATES=0 \
./scripts/prepare_swe_smith_freaksterz_training_data.sh
```

This creates `$RUN_DIR/freaksterz-on-policy.jsonl`. Once it completes, stop the
normal server to release its GPUs:

```bash
docker compose down
```

### 2. Extract and bridge hidden states

Start the target-only extractor using the now-free GPUs. `RUN_DIR` must be an
absolute path because it is bind-mounted into Docker. Leave this command
running while the collection command in the next terminal executes.

```bash
RUN_DIR="$RUN_DIR" ./scripts/run-hidden-states-extractor.sh
```

Then prepare the rendered rows, extract their target hidden states from port
8001, and bridge them to the native DFlash2 basis:

```bash
DATASET="$RUN_DIR/freaksterz-on-policy.jsonl" \
RUN_DIR="$RUN_DIR" \
ADAPTER="$ADAPTER" \
EXTRACT_ENDPOINT=http://127.0.0.1:8001/v1 \
MAX_SAMPLES=5000 \
SEQ_LENGTH=8192 \
CONCURRENCY=4 \
./scripts/collect_freaksterz_dflash2_data.sh
```

The useful products are `$RUN_DIR/data`, `$RUN_DIR/hidden-states`, and
`$RUN_DIR/bridge-report.json`. Confirm that the bridge report completed and
that the number of files in `data` and `hidden-states` agrees before stopping
the extractor or removing `raw-hidden-states`. The trainer reads only the
bridged copy. At the current 8192-token format, 479 examples used about 60 GiB
each for raw and bridged states; 5,000 examples therefore need roughly 1.25
TiB while both copies exist, or about 625 GiB after raw states are removed.

### 3. Fine-tune without discarding Z-Lab weights

The launcher creates a frozen native-basis training verifier, losslessly
converts the Z-Lab checkpoint to the Speculators checkpoint format on first
use, then launches FSDP over four GPUs. The converter copies the 81 trained
DFlash2 tensors unchanged; it never reinitializes the convolution or candidate
selector.

```bash
export RUN_DIR=/absolute/path/to/the/completed/run
export FREAKSTERZ_DIR=/absolute/path/to/models--Freaksterz--Qwen3.8-27B-SmoothQuant-W8A8-INT8/snapshots/MAIN_REVISION
export ADAPTER="$PWD/artifacts/freaksterz-dflash-adapter.safetensors"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPUS=4 \
SEQ_LENGTH=8192 \
MAX_ANCHORS=32 \
EPOCHS=2 \
./scripts/train_freaksterz_dflash2.sh
```

`MAX_ANCHORS=32` is the verified safe default for four 24 GiB GPUs. DFlash2
materializes full-vocabulary tensors, so increasing it needs measured VRAM
headroom. The launch log is written to `$RUN_DIR/logs/train.log`, and every
epoch checkpoint is under `$RUN_DIR/checkpoints`.

Validation loss is useful as a training signal but is not a reliable serving
selection criterion for DFlash2. Export and measure each epoch on a fixed,
held-out Hermes/SWE prompt suite, then keep the checkpoint with the best real
vLLM acceptance rate.

## How the bridge works

Freaksterz `main` applies one global orthogonal rotation to its residual
stream. In row-vector notation:

```text
h_rot = h_native @ R.T
```

The DFlash2 checkpoints were trained in the native basis. vLLM normally shares
the target token embedding and LM head with DFlash2 and passes target auxiliary
hidden-state taps to its `fc` projection. That is incompatible with a rotated
target, so the included vLLM patch applies three exact runtime bridges:

1. It maps shared target embeddings from rotated to native basis before draft
   inference: `E_rot @ R`.
2. It independently maps each of the five target auxiliary taps with
   `h_rot @ R` before the trained DFlash2 `fc` projection.
3. It maps native DFlash hidden states into Freaksterz's rotated LM-head basis
   with the inverse final-norm gain and `R.T`; the candidate selector remains
   native, exactly as trained.

The adapter is runtime-only: neither the Freaksterz target nor either DFlash2
checkpoint is rewritten or retrained.

## Why Lued needs one more fallback

z-lab's BF16 QKV projections expose dense weights, so vLLM fuses the five
context-KV projections into one GEMM. Lued's compressed-tensors W8A16 QKV
projections intentionally have no dense `weight` tensor. The patch detects
that representation and runs the five native quantized QKV projections instead,
then uses the normal K-norm, RoPE, and KV-cache write path. This preserves the
W8 drafter's memory saving, though its long-context draft prefill can be slower
than z-lab's fused path.

## Verify startup

For z-lab, the logs should contain:

```text
Enabled Freaksterz DFlash2 basis adapter
```

For Lued, they should additionally contain:

```text
DFlash context-KV uses quantized per-layer QKV projections
```

## Recreate the adapter locally

This is only needed to reproduce the published runtime adapter; normal Docker
deployment downloads it automatically. The scripts use the default Hugging Face
cache (`~/.cache/huggingface`) and download the exact pinned base and
Freaksterz `main` revisions when they are not already cached.

Create a small rebuild environment with `uv`:

```bash
uv venv .venv-rebuild
uv pip install --python .venv-rebuild/bin/python -r scripts/requirements-rebuild.txt
```

The rotation is deterministic. Generate it first; this verifies the original
rotation-file SHA-256, so a different PyTorch/LAPACK result fails immediately:

```bash
.venv-rebuild/bin/python scripts/generate_rotation.py --output rotation-R.pt
```

Build and validate the adapter. CUDA is used only for the sampled matrix
validation; use `--device cpu` if no GPU is available.

```bash
.venv-rebuild/bin/python scripts/build_adapter.py \
  --rotation rotation-R.pt \
  --output freaksterz-dflash-adapter.safetensors

.venv-rebuild/bin/python scripts/verify_adapter.py \
  --adapter freaksterz-dflash-adapter.safetensors
```

To build from pre-downloaded snapshots instead, pass `--base /path/to/qwen`
and `--main /path/to/freaksterz-main` to both scripts. The expected original
rotation SHA-256 is `8d6dd7bb2278c288f4e74583807bbc6429200839ecf8fe6e9319a23326e6a505`;
the existing published adapter SHA-256 is
`1c0d668abd1e1bbfe7a4d98cb6dbf40e2a34b909bffd7c849f40648f1ef64f09`.
