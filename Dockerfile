# Worker image = official vLLM-Omni image + RunPod serverless wrapper.
# vLLM-Omni upgrades are a single build ARG:
#   docker buildx build --build-arg VLLM_OMNI_VERSION=v0.28.0 ...
ARG VLLM_OMNI_VERSION=v0.28.0
FROM vllm/vllm-omni:${VLLM_OMNI_VERSION}
ARG VLLM_OMNI_VERSION

# RunPod serverless SDK + HTTP proxy deps (vLLM-Omni itself comes from the base image).
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

COPY src/handler.py /handler.py
COPY src/main.py /main.py

# Model selection is runtime-only for now: the worker downloads MODEL_NAME from
# HF at cold start. A bake-the-model build stage like worker-vllm's Option 2 is a
# natural follow-up once the runtime path is proven.
#
# The cache paths match worker-vllm exactly, because they are not ours to
# choose: Runpod mounts a `modelReferences` model under
# $BASE_PATH/huggingface-cache/hub, and huggingface_hub only finds it if
# HUGGINGFACE_HUB_CACHE points at that directory. Anything else -- including the
# obvious $BASE_PATH/huggingface -- means every cold start re-downloads tens of
# gigabytes that were already on the host.
ARG BASE_PATH="/runpod-volume"
ENV VLLM_OMNI_PORT=8091 \
    BASE_PATH="${BASE_PATH}" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets"

CMD ["python3", "-u", "/main.py"]
