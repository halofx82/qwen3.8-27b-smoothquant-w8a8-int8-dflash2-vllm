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
