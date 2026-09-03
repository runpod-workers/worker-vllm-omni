"""The exact first requests the Runpod console prefills after a deploy.

These payloads are authored in main-ui (`helpers/modelDeploySelection.ts`) and
are the first thing a freshly deployed endpoint ever runs, so a mismatch between
them and this worker's routing shows up as a user's opening request failing.
Copied verbatim, with the base64 image shortened.
"""
import base64

import pytest

from handler import _form_from_body, _is_multipart, _resolve

SAMPLE_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"fake").decode()

TEXT_TO_IMAGE = {"prompt": "a lighthouse at sunrise", "size": "1024x1024", "seed": 42}
TEXT_TO_VIDEO = {
    "prompt": "a fox moving through a snowy forest",
    "size": "1280x704",
    "seconds": 3,
    "fps": 24,
    "seed": 42,
}
IMAGE_TO_VIDEO = {
    **TEXT_TO_VIDEO,
    "image_reference": {"image_url": f"data:image/png;base64,{SAMPLE_B64}"},
}
IMAGE_TO_IMAGE = {
    "route": "/v1/images/edits",
    "body": {"prompt": "make the hills snowy", "image_b64": SAMPLE_B64, "size": "1024x1024"},
}


@pytest.mark.parametrize(
    "payload,expected_route",
    [
        (TEXT_TO_IMAGE, "/v1/images/generations"),
        (TEXT_TO_VIDEO, "/v1/videos/sync"),
        (IMAGE_TO_VIDEO, "/v1/videos/sync"),
        (IMAGE_TO_IMAGE, "/v1/images/edits"),
    ],
)
def test_each_prefill_reaches_the_route_it_means_to(payload, expected_route):
    route, method, _ = _resolve(payload)
    assert (route, method) == (expected_route, "POST")


def test_the_reference_image_survives_as_a_form_field():
    """`image_reference` is JSON in a form field, which is what omni parses."""
    route, _, body = _resolve(IMAGE_TO_VIDEO)
    assert _is_multipart(route)

    fields = {f[0]["name"]: f[2] for f in _form_from_body(body)._fields}
    assert "image_reference" in fields
    # A JSON object with image_url and nothing else: the model forbids extras.
    assert fields["image_reference"].startswith('{"image_url": "data:image/png;base64,')
    assert '"type"' not in fields["image_reference"]


def test_the_edited_image_arrives_as_a_file_part():
    route, _, body = _resolve(IMAGE_TO_IMAGE)
    assert _is_multipart(route)

    fields = {f[0]["name"]: f[2] for f in _form_from_body(body)._fields}
    # Decoded to bytes under the name the endpoint declares, not as base64 text.
    assert fields["image"] == base64.b64decode(SAMPLE_B64)
    assert "image_b64" not in fields
    assert fields["prompt"] == "make the hills snowy"


def test_a_text_only_video_request_carries_no_reference():
    _, _, body = _resolve(TEXT_TO_VIDEO)
    assert "image_reference" not in body
