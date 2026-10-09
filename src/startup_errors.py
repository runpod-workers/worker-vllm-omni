"""Turning a failed engine start into something the user can act on.

`vllm serve` reports its problems as Python tracebacks a hundred lines long,
and the worker only sees them as stdout. When one of the known ones goes by, we
would rather answer the next job with a sentence naming the cause and the fix
than let the container exit and be restarted into the same wall.

Only failures that a restart cannot fix belong here. Anything unrecognised is
left alone so the platform can retry it, which is the right move for a flaky
download or a host that lost its GPU.
"""

import re

GIB = 1 << 30

# torch prints both the request and the card's capacity; the capacity is the
# useful half, since the request is whatever happened to be next.
_OOM = re.compile(r"torch\.OutOfMemoryError|CUDA out of memory", re.I)
_OOM_CAPACITY = re.compile(r"total capacity of ([\d.]+) GiB", re.I)
_NO_SPACE = re.compile(r"No space left on device|ENOSPC|errno 28", re.I)
_BAD_HEADER = re.compile(r"Error while deserializing header|SafetensorError", re.I)
_UNSUPPORTED = re.compile(r"No supported model class found|Unsupported model architecture", re.I)
# A checkpoint whose layer counts disagree with the pipeline the engine built
# for it: the tensors it does not recognise are simply never filled.
# A quantisation config the engine cannot parse. `BitsAndBytesConfig.to_dict()`
# emits the private backing fields alongside the public ones, and the engine
# splats the whole dict into a dataclass that declares only the public names.
_QUANT_CONFIG = re.compile(
    r"unexpected keyword argument '_load_in|DiffusionBitsAndBytesConfig", re.I
)
_SHAPE_MISMATCH = re.compile(
    r"weights were not initialized from checkpoint|size mismatch for", re.I
)


def human_size(num_bytes: float) -> str:
    return f"{num_bytes / GIB:.1f} GiB"


def classify(output: str, model: str | None = None) -> str | None:
    """One actionable sentence for a known failure, or None to let it retry."""
    named = model or "This model"

    if _OOM.search(output):
        capacity = _OOM_CAPACITY.search(output)
        card = f" This GPU has {capacity.group(1)} GiB." if capacity else ""
        return (
            f"{named} ran out of GPU memory while loading."
            f"{card} Redeploy on a larger GPU, or pick a smaller model. "
            f"vLLM-Omni keeps the whole model resident and cannot offload to CPU, "
            f"so a model that does not fit will not run at any batch size."
        )

    if _NO_SPACE.search(output):
        return (
            f"{named} ran out of disk while downloading. Increase the endpoint's "
            f"container disk to comfortably exceed the size of the repository, or "
            f"attach a network volume so the weights are cached there instead."
        )

    if _BAD_HEADER.search(output):
        return (
            f"{named} stores its weights in a format the engine cannot read. "
            f"vLLM-Omni loads each diffusers component from a .safetensors file, "
            f"and repositories that ship .bin pickles instead fail here. Look for "
            f"another copy of the model published with safetensors."
        )

    if _UNSUPPORTED.search(output):
        return (
            f"{named} is not an architecture vLLM-Omni can serve. Check the "
            f"supported-models list for a pipeline that covers it."
        )

    if _QUANT_CONFIG.search(output):
        return (
            f"{named} is quantised in a format this engine version cannot read. Its "
            f"config stores the bitsandbytes settings under private field names, and "
            f"the loader rejects them outright. Deploy the unquantised model it was "
            f"derived from, on a GPU large enough to hold it."
        )

    if _SHAPE_MISMATCH.search(output):
        return (
            f"{named} does not match the pipeline it declares. Its config names a "
            f"supported pipeline, but the checkpoint has a different set of layers, "
            f"so part of the model would be left uninitialised. Pruned and distilled "
            f"variants do this even though they keep the original pipeline's name. "
            f"Use the model this one was derived from, or a fine-tune that kept its "
            f"architecture."
        )

    return None


def classify_runtime_error(response_text: str) -> str | None:
    """One actionable sentence for a request the engine could not serve.

    Distinct from `classify`: the engine is healthy and only this request
    failed, so nothing is exiting and the worker keeps taking jobs. What it
    replaces is the raw text, which for an OOM is several paragraphs of
    allocator state -- too long to read and, as the platform rejects an
    oversized result, sometimes too long to deliver at all.
    """
    if _OOM.search(response_text):
        return (
            "Ran out of GPU memory while generating. The model loaded but left too "
            "little room to run: lower the resolution, the frame count or the number "
            "of steps, or redeploy on a larger GPU."
        )

    return None
