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
# HF at cold start (cache it on a network volume via HF_HOME to make cold
# starts sane). A bake-the-model build stage like worker-vllm's Option 2 is a
# natural follow-up once the runtime path is proven.
ENV VLLM_OMNI_PORT=8091 \
    HF_HOME=/runpod-volume/huggingface

CMD ["python3", "-u", "/main.py"]
