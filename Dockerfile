# vLLM 0.28.0 with the Freaksterz rotated-residual DFlash2 bridge.
FROM vllm/vllm-openai:v0.28.0

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*

COPY patches/vllm-freaksterz-dflash2.patch /tmp/vllm-freaksterz-dflash2.patch
RUN site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && patch --batch --forward -p0 -d "$site_dir" < /tmp/vllm-freaksterz-dflash2.patch \
    && python3 -m py_compile \
        "$site_dir/vllm/model_executor/models/qwen3_dflash.py" \
        "$site_dir/vllm/model_executor/models/qwen3_dflash2.py" \
    && rm /tmp/vllm-freaksterz-dflash2.patch

COPY scripts/download-adapter-and-serve.sh /opt/freaksterz-dflash2-vllm/download-adapter-and-serve.sh
RUN chmod +x /opt/freaksterz-dflash2-vllm/download-adapter-and-serve.sh

ENV FREAKSTERZ_DFLASH_ADAPTER_REPO=halofx/freaksterz-dflash2-vllm-adapter
ENV FREAKSTERZ_DFLASH_ADAPTER_FILENAME=freaksterz-dflash-adapter.safetensors
ENTRYPOINT ["/opt/freaksterz-dflash2-vllm/download-adapter-and-serve.sh"]
