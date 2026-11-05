from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from ml_pipes.vision import Decode, ImagePayload, LoadFile

ImageInput: TypeAlias = ImagePayload | str | Path


def load_image(image: ImageInput) -> ImagePayload:
    if isinstance(image, (str, Path)):
        data = LoadFile()(image)
        try:
            image = Decode()(data)
        except (ValueError, cv2.error) as exc:
            raise ValueError(f"Could not decode image: {image}") from exc
    validate_image(image)
    return image


def validate_image(image: ImagePayload) -> None:
    if not isinstance(image, ImagePayload):
        raise ValueError("VLM tasks require an ImagePayload")
    array = image.array
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
        raise ValueError("VLM images must contain a uint8 NumPy array")
    color = (
        image.layout == "HWC"
        and image.color_space in {"RGB", "BGR"}
        and array.ndim == 3
        and array.shape[2] == 3
    )
    gray = image.layout == "HW" and image.color_space == "GRAY" and array.ndim == 2
    if not (color or gray) or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("VLM images must be nonempty HWC RGB/BGR or HW GRAY")


def encode_image(image: ImagePayload) -> bytes:
    validate_image(image)
    pixels = image.array
    if image.color_space == "RGB":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", np.ascontiguousarray(pixels))
    if not success:
        raise ValueError("Could not encode the image as PNG")
    return encoded.tobytes()
