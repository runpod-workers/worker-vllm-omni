"""Which engine failures are worth answering, and which are worth a restart."""
import pytest

from startup_errors import classify, classify_runtime_error

OOM = """
[rank0] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB.
GPU 0 has a total capacity of 44.42 GiB of which 105.81 MiB is free.
"""
NO_SPACE = "SafetensorError: IO Error: No space left on device (os error 28)"
BAD_HEADER = "safetensors._safetensors_rust.SafetensorError: Error while deserializing header: header too large"


class TestKnownFailures:
    def test_out_of_memory_names_the_card_and_the_way_out(self):
        message = classify(OOM, model="Qwen/Qwen-Image-Edit-2511")

        assert "Qwen/Qwen-Image-Edit-2511" in message
        assert "44.42 GiB" in message
        # The advice has to rule out the thing users try first on an LLM worker.
        assert "cannot offload to CPU" in message

    def test_out_of_disk_points_at_container_disk(self):
        assert "container disk" in classify(NO_SPACE)

    def test_unreadable_weights_name_the_format(self):
        message = classify(BAD_HEADER)

        assert ".safetensors" in message and ".bin" in message

    def test_the_model_is_named_when_known(self):
        assert classify(NO_SPACE, model="org/thing").startswith("org/thing")

    def test_falls_back_to_a_generic_subject(self):
        assert classify(NO_SPACE).startswith("This model")


class TestUnknownFailures:
    @pytest.mark.parametrize(
        "output",
        [
            "",
            "Connection reset by peer while downloading",
            "RuntimeError: something nobody has seen before",
        ],
    )
    def test_are_left_for_the_platform_to_retry(self, output):
        # Answering these forever would turn a flaky download into a dead
        # endpoint; a restart is the right response.
        assert classify(output) is None

    def test_out_of_memory_wins_over_surrounding_noise(self):
        assert classify(f"Connection reset\n{OOM}\nmore noise") is not None


def test_pruned_variant_of_a_supported_pipeline_is_named_as_such():
    # segmind/SSD-1B: tagged StableDiffusionXLPipeline, but its UNet drops the
    # mid-block's attention and most of its transformer layers, so the engine
    # builds a full SDXL and leaves those tensors unfilled.
    output = (
        "ValueError: The quantization config is None, and the following weights "
        "were not initialized from checkpoint: "
        "{'unet.mid_block.attentions.0.proj_in.weight'}"
    )
    message = classify(output, "segmind/SSD-1B")
    assert message is not None
    assert "segmind/SSD-1B" in message
    assert "different set of layers" in message


def test_a_shape_mismatch_is_reported_rather_than_retried():
    assert classify("size mismatch for unet.conv_in.weight", "org/model") is not None


def test_bitsandbytes_config_is_named_rather_than_crash_looped():
    # ovedrive/Qwen-Image-Edit-2511-4bit: BitsAndBytesConfig.to_dict() emits the
    # private backing fields, and the engine's dataclass declares only the
    # public ones, so it exits before serving anything.
    output = (
        "TypeError: DiffusionBitsAndBytesConfig.__init__() got an unexpected "
        "keyword argument '_load_in_4bit'"
    )
    message = classify(output, "ovedrive/Qwen-Image-Edit-2511-4bit")
    assert message is not None
    assert "ovedrive/Qwen-Image-Edit-2511-4bit" in message
    assert "quantised" in message


def test_runtime_oom_is_shortened_to_something_readable():
    # Wan 2.2 I2V A14B loads into 64GB of an 80GB card and then fails here. The
    # raw text is several paragraphs of allocator state.
    raw = (
        "RuntimeError: CUDA out of memory. Tried to allocate 1.94 GiB. GPU 0 has a "
        "total capacity of 79.18 GiB of which 1.55 GiB is free. Including non-PyTorch "
        "memory, this process has 76.90 GiB memory in use..."
    )
    message = classify_runtime_error(raw)
    assert message is not None
    assert len(message) < len(raw)
    assert "larger GPU" in message


def test_a_request_that_failed_for_another_reason_is_passed_through():
    assert classify_runtime_error('{"error": "size must be WxH"}') is None
