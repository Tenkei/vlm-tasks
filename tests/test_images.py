from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
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
