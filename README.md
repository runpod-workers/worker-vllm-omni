# worker-vllm-omni

Deploy any [vLLM-Omni-supported](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)
image, video, audio, or omni model as a Runpod Serverless endpoint with **one
standardized, OpenAI-compatible request schema** — the same JSON for every
model family, the way [worker-vllm](https://github.com/runpod-workers/worker-vllm)
does it for LLMs.

Set `MODEL_NAME` to a Hugging Face repo id, deploy, and send OpenAI-style
requests. No workflow JSON, no per-model request formats.

```
MODEL_NAME=Tongyi-MAI/Z-Image-Turbo        # text-to-image
MODEL_NAME=Wan-AI/Wan2.2-TI2V-5B-Diffusers # text/image-to-video
MODEL_NAME=Qwen/Qwen-Image-Edit-2511       # image editing
```

## Request shapes

Everything goes through the normal Runpod queue API (`/run`, `/runsync`). Three
input styles, checked in this order:

**1. OpenAI passthrough** (what the platform sends on `/openai/v1/...` routes):

```json
{"input": {"openai_route": "/v1/images/generations",
           "openai_input": {"prompt": "a cup of coffee", "size": "1024x1024"}}}
```

**2. Generic proxy** to any vLLM-Omni route (job polling, model listing, ...):

```json
{"input": {"route": "/v1/videos/vid_abc123", "method": "GET"}}
```

**3. Media shorthand** — a bare OpenAI-style body; the route is inferred
(`messages` → chat, `seconds`/`fps` → video, otherwise image). Force it with
`"task": "image" | "video" | "video_async" | "image_edit" | "chat" | "speech" | "audio"`:

```json
{"input": {"prompt": "a small robot reading beside a window",
           "size": "1024x1024", "seed": 42}}
```

```json
{"input": {"task": "video", "prompt": "a mountain lake at sunrise",
           "size": "1280x704", "seconds": 5}}
```

Binary uploads on multipart routes (image edits, image-to-video) are passed
base64-encoded as `<field>_b64`, e.g. `"image_b64": "<base64 png>"`.

## Responses

- Image generations return the OpenAI response JSON (`data[0].b64_json`), plus
  vLLM-Omni's `metrics` (stage durations, peak VRAM).
- Binary responses (`/v1/videos/sync`, `response_format: "file"`) return
  `{"data_b64", "content_type", "size_bytes"}`. Videos are large — for real
  workloads prefer `task: "video_async"` and poll `/v1/videos/{id}` via the
  generic proxy, or front the endpoint with a load balancer (below).
- `stream: true` chat bodies return the SSE chunks as a list, in order. The
  handler is not a generator on purpose — see the note in `src/handler.py`; for
  real incremental streaming, front the endpoint with a load balancer (below).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | (required) | HF repo id of a vLLM-Omni supported model |
| `LORA_REPO` | — | HF repo id of a LoRA adapter to fuse at startup |
| `LORA_FILE` | — | File or subfolder within `LORA_REPO` (whole repo when unset) |
| `LORA_PATH` | — | Local adapter path (baked image / network volume) — alternative to `LORA_REPO` |
| `LORA_BACKEND` | `distill` | `distill` fuses at startup (online-serving mode); `peft` uses the adapter manager |
| `LORA_SCALE` | — | Passed to `--lora-scale` |
| `OMNI_EXTRA_ARGS` | — | Extra `vllm serve` args, e.g. `--tensor-parallel-size 2` |
| `VLLM_OMNI_PORT` | `8091` | Loopback port for the API server |
| `OMNI_STARTUP_TIMEOUT` | `1800` | Seconds to wait for model load before failing |
| `REQUEST_TIMEOUT` | `3600` | Per-request proxy timeout (video gens are minutes) |
| `BASE_PATH` | `/runpod-volume` | Root of the HF cache. Baked into the image env; matches worker-vllm |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `$BASE_PATH/huggingface-cache/hub` | Where weights are cached. This exact path is where Runpod mounts a `modelReferences` model, so leaving it alone is what lets a cold start skip the download |
| `HF_TOKEN` | — | For gated models (e.g. `krea/Krea-2-Turbo`) |

## Model support and sizing (validated 2026-09-02, vllm-omni v0.28.0)

Coverage is an **allowlist of ~72 architectures in diffusers/HF format** —
including FLUX.1/FLUX.2/Kontext, SDXL, and SD3.5. What it does NOT cover:
single-file (.safetensors) community checkpoints, ComfyUI-format repos, and
bare LoRA repos — those stay on the ComfyUI path. Many top models are gated on
HF (FLUX, Krea 2, SD3.5): set HF_TOKEN and accept the license first.

| Model | Task | Works | Peak VRAM | Notes |
|---|---|---|---|---|
| `Tongyi-MAI/Z-Image-Turbo` | t2i | ✅ ~20s/image 1024² | ~24 GB | fits 32 GB GPUs |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | t2v/i2v | ✅ 68s for 3s 1280×704@24fps via /v1/videos/sync | — | 34 GB weights |
| `krea/Krea-2-Turbo` | t2i | untested | — | gated: needs HF_TOKEN |
| `Lightricks/LTX-2.5-Diffusers` | i2v/t2v | untested | — | 174 GB repo; use distilled variants single-GPU |
| `MiniMaxAI/MiniMax-H3` | t2v+audio | out of scope | — | 498 GB — multi-GPU/multi-node |

## LoRA adapters (validated 2026-09-02)

vLLM-Omni loads diffusion LoRAs natively (`--lora-path`, `--lora-backend
peft|distill`, `--lora-scale` — pass via `OMNI_EXTRA_ARGS` after downloading
the adapter). **Support is per-pipeline upstream**, via LoRA loader mixins:
today that's **Qwen-Image, Wan 2.2 (t2v/i2v), SenseNova U1** (+ LTX two-stage
configs). Not yet: Z-Image, Krea 2, FLUX-in-omni.

Validated positive: Qwen-Image + `lightx2v/Qwen-Image-Lightning` 4-step LoRA
(distill backend) — log shows `1440 lora keys loaded into
QwenImageTransformer2DModel`, and a 1024² generation at
`num_inference_steps: 4, guidance_scale: 1.0` completes in **4.0s** (~59 GB
peak VRAM). `num_inference_steps`/`guidance_scale` are accepted in the
standard images request.

**Silent-failure trap (validated the hard way):** on a pipeline WITHOUT LoRA
support, the server logs `Pipeline does not support loading distilled LoRA
weights for now.` and starts anyway — generations silently use the base model
(byte-identical same-seed outputs). A worker configured with a LoRA MUST
treat that warning as fatal at startup rather than serve the wrong model.

## Alternative: load-balancer endpoint

Because vLLM-Omni is a long-lived OpenAI-compatible HTTP server, this image
also works as a Runpod **load-balancer** endpoint exposing `/v1` directly to
OpenAI SDK clients (no queue, no base64 wrapping) — start the server on
`0.0.0.0` and expose the port instead of running the handler. The queue handler
is the default because it gets scale-to-zero and per-job billing semantics.

## Status

Scaffold — builds untested in CI, hub config skeletal. Validation so far was
against the stock `vllm/vllm-omni:v0.28.0` image on Runpod pods (RTX PRO 6000,
CUDA 13.2 host): health, `/v1/images/generations` end-to-end.
