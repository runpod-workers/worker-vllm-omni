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
# One thing should decide how long a request may run, and that thing is the
# endpoint's execution timeout: it is the number the user can see and change.
#
# vLLM-Omni disagrees by default -- POST /v1/videos/sync is wrapped in a 600s
# deadline that answers 504 -- and the platform does not tell the container what
# the endpoint's timeout is, so the two cannot be kept in step. Left at 600 it
# killed a 40-step clip two thirds of the way through on an endpoint configured
# for 1800. Set to any specific value it silently caps whatever the user set in
# the console the moment they raise it past this. So it is set high enough never
# to be the binding constraint, and the endpoint decides.
ARG BASE_PATH="/runpod-volume"
ENV VLLM_OMNI_VIDEO_SYNC_TIMEOUT=86400 \
    VLLM_OMNI_PORT=8091 \
    BASE_PATH="${BASE_PATH}" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets"

CMD ["python3", "-u", "/main.py"]
