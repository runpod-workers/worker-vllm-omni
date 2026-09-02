"""RunPod Serverless handler that proxies jobs to the local vLLM-Omni server.

`main.py` starts `vllm serve <model> --omni` on 127.0.0.1:VLLM_OMNI_PORT and
only then starts the RunPod serverless loop, so by the time a job arrives the
server is up. We never import vLLM-Omni — the worker stays forwards/backwards
compatible across omni releases.

Accepted job input shapes (all under job["input"]):

1. RunPod OpenAI passthrough (what the platform sends on /openai/v1/...):
       {"openai_route": "/v1/images/generations", "openai_input": {...}}
2. Generic proxy to any omni route:
       {"route": "/v1/videos/abc123", "method": "GET"}
       {"route": "/v1/images/generations", "body": {...}}
3. Media shorthand — a bare OpenAI-style request body; the route is inferred:
       {"prompt": "a cup of coffee", "size": "1024x1024"}          -> images/generations
       {"prompt": "...", "seconds": 5, "size": "1280x720"}         -> videos/sync
       {"messages": [...]}                                          -> chat/completions
   Force the route with "task": "image" | "video" | "chat" | "speech".

JSON-in/JSON-out is preserved for the queue transport:
  * multipart routes (/v1/videos, /v1/videos/sync, /v1/images/edits) are built
    from the JSON body; binary file fields are accepted base64-encoded under
    "<field>_b64" (e.g. "image_b64") and decoded into upload parts.
  * image responses default to response_format=b64_json.
  * binary responses (video bytes, file-format images) are returned as
    {"data_b64": ..., "content_type": ...}. Videos of any real length are
    large — prefer the async /v1/videos flow, or configure a network volume
    and fetch via the job-lifecycle routes.

Chat/completions bodies with "stream": true yield raw SSE chunks as they
arrive (aggregated by the platform unless streaming is requested).
"""

import base64
import json
import logging
import os
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OMNI_PORT = os.getenv("VLLM_OMNI_PORT", "8091")
OMNI_BASE_URL = os.getenv("OMNI_BASE_URL", f"http://127.0.0.1:{OMNI_PORT}")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))

# Routes whose upstream contract is multipart/form-data, not JSON.
MULTIPART_ROUTES = ("/v1/videos", "/v1/images/edits")

TASK_ROUTES = {
    "image": "/v1/images/generations",
    "image_edit": "/v1/images/edits",
    "video": "/v1/videos/sync",
    "video_async": "/v1/videos",
    "chat": "/v1/chat/completions",
    "speech": "/v1/audio/speech",
    "audio": "/v1/audio/generate",
}

# Set by main.py once the omni subprocess is running.
omni_process = None


def _is_omni_alive() -> bool:
    return omni_process is None or omni_process.poll() is None


def _infer_route(body: dict) -> str:
    if "messages" in body:
        return TASK_ROUTES["chat"]
    if "seconds" in body or "num_frames" in body or "fps" in body:
        return TASK_ROUTES["video"]
    if "input" in body and "voice" in body:
        return TASK_ROUTES["speech"]
    return TASK_ROUTES["image"]


def _resolve(job_input: dict) -> Tuple[str, str, Optional[dict]]:
    """Map a job input to (route, method, body)."""
    if job_input.get("openai_input"):
        return job_input.get("openai_route") or TASK_ROUTES["image"], "POST", job_input["openai_input"]
    if job_input.get("openai_route"):
        return job_input["openai_route"], "GET", None

    if job_input.get("route"):
        body = job_input.get("body")
        method = (job_input.get("method") or ("POST" if body is not None else "GET")).upper()
        return job_input["route"], method, body

    body = {k: v for k, v in job_input.items() if k != "task"}
    task = job_input.get("task")
    if task:
        route = TASK_ROUTES.get(task)
        if not route:
            raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_ROUTES)}")
    else:
        route = _infer_route(body)
    return route, "POST", body


def _is_multipart(route: str) -> bool:
    return route.rstrip("/").startswith(MULTIPART_ROUTES) and not route.rstrip("/").endswith("/generations")


def _form_from_body(body: dict) -> aiohttp.FormData:
    form = aiohttp.FormData()
    for key, value in body.items():
        if key.endswith("_b64"):
            field = key[: -len("_b64")]
            form.add_field(field, base64.b64decode(value), filename=f"{field}.bin")
        elif isinstance(value, (dict, list)):
            form.add_field(key, json.dumps(value))
        else:
            form.add_field(key, str(value))
    return form


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    job_input = job.get("input") or {}
    try:
        route, method, body = _resolve(job_input)
    except ValueError as e:
        yield {"error": str(e)}
        return

    if not _is_omni_alive():
        yield {"error": "vLLM-Omni server process has exited; worker is unhealthy"}
        return

    # Queue transport is JSON, so keep image payloads inline by default.
    if route.endswith("/images/generations") and body is not None:
        body.setdefault("response_format", "b64_json")

    url = f"{OMNI_BASE_URL}{route}"
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    stream = bool(body and body.get("stream"))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        kwargs: dict = {}
        if body is not None and method != "GET":
            if _is_multipart(route):
                kwargs["data"] = _form_from_body(body)
            else:
                kwargs["json"] = body

        async with session.request(method, url, **kwargs) as resp:
            content_type = resp.headers.get("Content-Type", "")

            if resp.status >= 400:
                text = await resp.text()
                try:
                    yield {"error": json.loads(text), "status": resp.status}
                except json.JSONDecodeError:
                    yield {"error": text, "status": resp.status}
                return

            if stream and "text/event-stream" in content_type:
                async for chunk in resp.content:
                    text = chunk.decode("utf-8", errors="replace")
                    if text.strip():
                        yield text
                return

            if "application/json" in content_type:
                yield await resp.json()
                return

            # Binary payload (video bytes, image files, audio).
            payload = await resp.read()
            yield {
                "data_b64": base64.b64encode(payload).decode(),
                "content_type": content_type or "application/octet-stream",
                "size_bytes": len(payload),
            }
