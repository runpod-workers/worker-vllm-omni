"""Container entrypoint: spawn `vllm serve --omni`, wait for it, start RunPod serverless.

Mirrors worker-vllm's design: the worker never imports vLLM-Omni. We launch
`vllm serve <MODEL_NAME> --omni` on the loopback interface, poll /health until
the server (and model) is ready, and only then start the RunPod serverless job
loop so no job is pulled before the backend can serve it.

Environment:
  MODEL_NAME            (required) HF repo id, e.g. Tongyi-MAI/Z-Image-Turbo.
                        Must be a vLLM-Omni supported architecture in
                        diffusers/HF format — see the supported-models list.
  OMNI_EXTRA_ARGS       Extra CLI args appended verbatim to `vllm serve`
                        (e.g. "--tensor-parallel-size 2"). Split on whitespace.
  VLLM_OMNI_PORT        Loopback port for the API server (default 8091).
  OMNI_STARTUP_TIMEOUT  Seconds to wait for /health (default 1800 — diffusion
                        weights are tens of GB; first boot downloads them).
"""

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OMNI_HOST = "127.0.0.1"
OMNI_PORT = os.getenv("VLLM_OMNI_PORT", "8091")
STARTUP_TIMEOUT = int(os.getenv("OMNI_STARTUP_TIMEOUT", "1800"))
HEALTH_POLL_INTERVAL = 2  # seconds


def start_omni() -> subprocess.Popen:
    model = os.getenv("MODEL_NAME")
    if not model:
        logging.error("MODEL_NAME is required (HF repo id of a vLLM-Omni supported model)")
        sys.exit(1)

    argv = ["vllm", "serve", model, "--omni", "--host", OMNI_HOST, "--port", OMNI_PORT]
    extra = os.getenv("OMNI_EXTRA_ARGS", "").strip()
    if extra:
        argv += shlex.split(extra)

    logging.info("Starting vLLM-Omni: %s", " ".join(argv))
    # Child gets our stdout/stderr so server logs land in the worker logs.
    return subprocess.Popen(argv)


def wait_for_omni(proc: subprocess.Popen) -> None:
    """Poll GET /health until the server is ready; fail fast if it crashes or times out."""
    url = f"http://{OMNI_HOST}:{OMNI_PORT}/health"

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited during startup with code {proc.returncode}")
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as resp:
                if resp.status == 200:
                    logging.info("vLLM-Omni is healthy")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    proc.send_signal(signal.SIGTERM)
    raise RuntimeError(f"vLLM-Omni not healthy after {STARTUP_TIMEOUT}s")


def main() -> None:
    proc = start_omni()
    try:
        wait_for_omni(proc)
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
