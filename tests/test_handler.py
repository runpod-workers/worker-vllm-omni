"""What the queue transport does with a job, without a vLLM-Omni server behind it."""
import inspect

import pytest
from runpod.serverless.modules.rp_handler import is_generator

import handler as handler_module
from handler import _form_from_body, _is_multipart, _resolve, handler


class TestTransportShape:
    """The reason this module is written the way it is."""

    def test_the_handler_is_not_a_generator(self):
        # A generator handler makes the SDK POST every yield to /job-stream,
        # which 400s on a payload the size of a base64 image. Turning this back
        # into a generator reintroduces that, and the job still succeeds -- so
        # nothing but this test would catch it.
        assert is_generator(handler) is False
        assert inspect.isasyncgenfunction(handler) is False

    def test_it_still_returns_a_coroutine_the_sdk_can_await(self):
        assert inspect.iscoroutinefunction(handler)


class TestRouteResolution:
    def test_a_bare_prompt_generates_an_image(self):
        route, method, body = _resolve({"prompt": "a cup of coffee"})
        assert route == "/v1/images/generations"
        assert method == "POST"
        assert body == {"prompt": "a cup of coffee"}

    @pytest.mark.parametrize("key", ["seconds", "fps", "num_frames"])
    def test_anything_time_based_generates_a_video(self, key):
        route, _, _ = _resolve({"prompt": "a fox running", key: 3})
        assert route == "/v1/videos/sync"

    def test_messages_go_to_chat(self):
        route, _, _ = _resolve({"messages": [{"role": "user", "content": "hi"}]})
        assert route == "/v1/chat/completions"

    def test_task_overrides_what_the_body_looks_like(self):
        route, _, body = _resolve({"prompt": "a fox running", "task": "video"})
        assert route == "/v1/videos/sync"
        # The override is instruction, not payload; omni would reject it.
        assert "task" not in body

    def test_an_unknown_task_says_what_it_accepts(self):
        with pytest.raises(ValueError, match="video_async"):
            _resolve({"prompt": "x", "task": "hologram"})

    def test_the_platform_passthrough_shape_wins(self):
        route, method, body = _resolve(
            {"openai_route": "/v1/images/generations", "openai_input": {"prompt": "x"}}
        )
        assert (route, method, body) == ("/v1/images/generations", "POST", {"prompt": "x"})

    def test_a_generic_route_can_be_polled(self):
        route, method, body = _resolve({"route": "/v1/videos/abc123", "method": "GET"})
        assert (route, method, body) == ("/v1/videos/abc123", "GET", None)


class TestMultipart:
    @pytest.mark.parametrize(
        "route,expected",
        [
            ("/v1/videos", True),
            ("/v1/videos/sync", True),
            ("/v1/images/edits", True),
            ("/v1/images/generations", False),
            ("/v1/chat/completions", False),
        ],
    )
    def test_only_the_upload_routes_are_multipart(self, route, expected):
        assert _is_multipart(route) is expected

    def test_a_b64_field_becomes_a_file_part(self):
        form = _form_from_body({"prompt": "x", "image_b64": "aGk="})
        names = [f[0]["name"] for f in form._fields]
        assert "image" in names
        assert "image_b64" not in names


class TestFailingEarly:
    async def test_an_unroutable_job_is_an_error_not_a_crash(self):
        result = await handler({"input": {"prompt": "x", "task": "nope"}})
        assert "error" in result

    async def test_a_dead_engine_is_reported_rather_than_dialled(self, monkeypatch):
        monkeypatch.setattr(handler_module, "_is_omni_alive", lambda: False)
        result = await handler({"input": {"prompt": "x"}})
        assert "exited" in result["error"]
