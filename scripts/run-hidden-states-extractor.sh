#!/usr/bin/env bash

docker run --rm \
  --gpus all \
  --ipc=host \
  --network host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$RUN_DIR:$RUN_DIR" \
  --entrypoint python3 \
  local/qwen38-smoothquant-dflash2:vllm-0.28.0 \
  -m vllm.entrypoints.cli.main serve \
  Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --served-model-name Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 4 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[5,19,33,47,61,64]}}}' \
  --kv-transfer-config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$RUN_DIR/raw-hidden-states\"}}"
