"""Which engine failures are worth answering, and which are worth a restart."""
import pytest

from startup_errors import classify

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
