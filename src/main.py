"""Container entrypoint: spawn `vllm serve --omni`, wait for it, start RunPod serverless.

Mirrors worker-vllm's design: the worker never imports vLLM-Omni. We launch
`vllm serve <MODEL_NAME> --omni` on the loopback interface, poll /health until
the server (and model) is ready, and only then start the RunPod serverless job
loop so no job is pulled before the backend can serve it.

Environment:
  MODEL_NAME            (required) HF repo id, e.g. Tongyi-MAI/Z-Image-Turbo.
                        Must be a vLLM-Omni supported architecture in
                        diffusers/HF format — see the supported-models list.
  LORA_REPO             Optional HF repo id of a LoRA adapter to fuse/load.
  LORA_FILE             Optional path within LORA_REPO (a file or subfolder,
                        e.g. "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
                        or "z-image-turbo-hpsv3"). Downloads the whole repo
                        when unset.
  LORA_PATH             Local adapter path — alternative to LORA_REPO for
                        adapters baked into the image or on a network volume.
  LORA_BACKEND          "distill" (default: fuse at startup — the documented
                        online-serving mode) or "peft".
  LORA_SCALE            Optional scale passed to --lora-scale.
  OMNI_EXTRA_ARGS       Extra CLI args appended verbatim to `vllm serve`
                        (e.g. "--tensor-parallel-size 2"). Split on whitespace.
  VLLM_OMNI_PORT        Loopback port for the API server (default 8091).
  OMNI_STARTUP_TIMEOUT  Seconds to wait for /health (default 1800 — diffusion
                        weights are tens of GB; first boot downloads them).

LoRA support in vLLM-Omni is per-pipeline (Qwen-Image, Wan 2.2, ... — not yet
Z-Image or Krea 2). When the pipeline lacks it, the server logs a warning and
SILENTLY serves the base model. That is never acceptable when the operator
configured an adapter, so startup scans the server output and fails hard on
that warning instead of serving the wrong model.
"""

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OMNI_HOST = "127.0.0.1"
OMNI_PORT = os.getenv("VLLM_OMNI_PORT", "8091")
STARTUP_TIMEOUT = int(os.getenv("OMNI_STARTUP_TIMEOUT", "1800"))
HEALTH_POLL_INTERVAL = 2  # seconds

LORA_DOWNLOAD_DIR = "/lora"

# Emitted by vllm_omni/diffusion/worker/diffusion_worker.py when the pipeline
# has no LoRA loader mixin: the adapter is skipped and the BASE model serves.
LORA_UNSUPPORTED_MARKER = "does not support loading distilled LoRA weights"
# Emitted by vllm_omni/diffusion/lora/loader.py on a successful distill fuse.
LORA_LOADED_MARKER = "lora keys loaded into"

lora_unsupported_seen = threading.Event()
lora_loaded_seen = threading.Event()


def resolve_lora_path() -> str | None:
    """Return a local adapter path from LORA_PATH or by downloading LORA_REPO."""
    local = os.getenv("LORA_PATH")
    if local:
        return local

    repo = os.getenv("LORA_REPO")
    if not repo:
        return None

    from huggingface_hub import snapshot_download

    subpath = os.getenv("LORA_FILE", "").strip().strip("/")
    allow = [f"{subpath}*"] if subpath else None
    logging.info("Downloading LoRA adapter %s%s", repo, f" ({subpath})" if subpath else "")
    snapshot_download(
        repo_id=repo,
        local_dir=LORA_DOWNLOAD_DIR,
        allow_patterns=allow,
        token=os.getenv("HF_TOKEN") or None,
    )
    return os.path.join(LORA_DOWNLOAD_DIR, subpath) if subpath else LORA_DOWNLOAD_DIR


def pump_output(proc: subprocess.Popen) -> None:
    """Echo server output to our stdout while scanning for LoRA markers."""
    for raw in proc.stdout:  # type: ignore[union-attr]
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        sys.stdout.write(line)
        sys.stdout.flush()
        if LORA_UNSUPPORTED_MARKER in line:
            lora_unsupported_seen.set()
        elif LORA_LOADED_MARKER in line:
            lora_loaded_seen.set()


def start_omni() -> subprocess.Popen:
    model = os.getenv("MODEL_NAME")
    if not model:
        logging.error("MODEL_NAME is required (HF repo id of a vLLM-Omni supported model)")
        sys.exit(1)

    argv = ["vllm", "serve", model, "--omni", "--host", OMNI_HOST, "--port", OMNI_PORT]

    lora_path = resolve_lora_path()
    if lora_path:
        argv += ["--lora-path", lora_path, "--lora-backend", os.getenv("LORA_BACKEND", "distill")]
        scale = os.getenv("LORA_SCALE")
        if scale:
            argv += ["--lora-scale", scale]

    extra = os.getenv("OMNI_EXTRA_ARGS", "").strip()
    if extra:
        argv += shlex.split(extra)

    logging.info("Starting vLLM-Omni: %s", " ".join(argv))
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    threading.Thread(target=pump_output, args=(proc,), daemon=True).start()
    return proc


def wait_for_omni(proc: subprocess.Popen, lora_configured: bool) -> None:
    """Poll GET /health until the server is ready; fail fast if it crashes or times out."""
    url = f"http://{OMNI_HOST}:{OMNI_PORT}/health"

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited during startup with code {proc.returncode}")
        if lora_configured and lora_unsupported_seen.is_set():
            proc.send_signal(signal.SIGTERM)
            raise RuntimeError(
                "A LoRA adapter was configured but this model's pipeline does not "
                "support LoRA loading in vLLM-Omni yet (see the per-pipeline support "
                "list). Refusing to start: the server would silently serve the BASE "
                "model. Remove the LoRA config or pick a supported base model."
            )
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as resp:
                if resp.status == 200:
                    if lora_configured:
                        if lora_unsupported_seen.is_set():
                            proc.send_signal(signal.SIGTERM)
                            raise RuntimeError(
                                "LoRA configured but unsupported by this pipeline; "
                                "refusing to serve the base model as if fine-tuned."
                            )
                        if lora_loaded_seen.is_set():
                            logging.info("LoRA adapter fused successfully")
                        else:
                            logging.warning(
                                "LoRA configured but no fuse confirmation seen in logs "
                                "(expected for the peft backend; verify per-request)"
                            )
                    logging.info("vLLM-Omni is healthy")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    proc.send_signal(signal.SIGTERM)
    raise RuntimeError(f"vLLM-Omni not healthy after {STARTUP_TIMEOUT}s")


def main() -> None:
    lora_configured = bool(os.getenv("LORA_REPO") or os.getenv("LORA_PATH"))
    proc = start_omni()
    try:
        wait_for_omni(proc, lora_configured)
    except RuntimeError:
        logging.exception("vLLM-Omni failed to start")
        sys.exit(1)

    # Let the handler fail fast if the server dies mid-flight.
    handler.omni_process = proc

    import runpod

    runpod.serverless.start(
        {
            "handler": handler.handler,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
