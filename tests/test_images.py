from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest


from ml_pipes.vision import ImagePayload
from ml_pipes.vlm._images import encode_image, load_image


@pytest.mark.parametrize("path_type", [str, Path])
def test_loading_a_path_preserves_pixels_and_dimensions(tmp_path, path_type):
    pixels = np.full((3, 5, 3), (10, 30, 200), dtype=np.uint8)
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), pixels)

    image = load_image(path_type(path))

    assert image.size == (5, 3)
    assert image.color_space == "BGR"
    assert image.layout == "HWC"
    np.testing.assert_array_equal(image.array, pixels)


def test_loading_a_payload_preserves_its_identity():
    image = ImagePayload(np.zeros((3, 5, 3), dtype=np.uint8), color_space="RGB")
    assert load_image(image) is image


def test_loading_an_rgb_array_preserves_pixels_and_dimensions():
    pixels = np.full((3, 5, 3), (200, 30, 10), dtype=np.uint8)[:, ::-1]
    image = load_image(pixels)
    assert image.size == (5, 3) and image.color_space == "RGB"
    assert image.layout == "HWC" and image.array.flags.c_contiguous
    np.testing.assert_array_equal(image.array, pixels)
    decoded = cv2.imdecode(np.frombuffer(encode_image(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, pixels[..., ::-1])


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_loading_a_pil_image_converts_to_rgb_without_changing_the_source(mode):
    source = Image.new("RGB", (5, 3), (200, 30, 10)).convert(mode)
    original = source.tobytes()
    image = load_image(source)
    assert image.size == (5, 3) and image.color_space == "RGB" and image.layout == "HWC"
    assert image.array.dtype == np.uint8 and image.array.flags.c_contiguous
    np.testing.assert_array_equal(image.array, np.asarray(source.convert("RGB")))
    assert source.mode == mode and source.tobytes() == original


@pytest.mark.parametrize("pixels", [
    np.zeros((3, 5, 3), dtype=np.float32),
    np.zeros((3, 5), dtype=np.uint8),
    np.zeros((3, 5, 4), dtype=np.uint8),
    np.zeros((0, 5, 3), dtype=np.uint8),
])
def test_loading_an_array_requires_uint8_hwc_rgb(pixels):
    with pytest.raises(ValueError):
        load_image(pixels)


@pytest.mark.parametrize("color_space", ["RGB", "BGR"])
def test_encoded_png_preserves_colors_and_input(color_space):
    array = np.zeros((3, 5, 3), dtype=np.uint8)
    array[:] = (10, 30, 200)
    original = array.copy()
    image = ImagePayload(array, color_space=color_space)
    encoded = encode_image(image)
    assert encoded.startswith(b"\x89PNG")
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    expected = array[..., ::-1] if color_space == "RGB" else array
    np.testing.assert_array_equal(decoded, expected)
    np.testing.assert_array_equal(array, original)


def test_grayscale_encoding_is_lossless():
    array = np.arange(15, dtype=np.uint8).reshape(3, 5)
    encoded = encode_image(ImagePayload(array, color_space="GRAY", layout="HW"))
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(decoded, array)
