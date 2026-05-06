"""Observation preprocessing for Pong image frames."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable

import numpy as np
import torch
from PIL import Image


FRAME_SIZE = 84
FRAME_STACK = 4


def preprocess_frame(observation: np.ndarray | torch.Tensor, size: int = FRAME_SIZE) -> torch.Tensor:
    """Convert one RGB/RGBA/grayscale frame into a normalized 1xHxW tensor."""

    if isinstance(observation, torch.Tensor):
        array = observation.detach().cpu().numpy()
    else:
        array = np.asarray(observation)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim == 3:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB").convert("L")
    elif array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    else:
        raise ValueError(f"Expected image frame with 2 or 3 dims, got shape {array.shape}")

    image = image.resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).unsqueeze(0)


def empty_frame(size: int = FRAME_SIZE) -> torch.Tensor:
    return torch.zeros((1, size, size), dtype=torch.float32)


def make_frame_stack(stack_size: int = FRAME_STACK, size: int = FRAME_SIZE) -> Deque[torch.Tensor]:
    return deque((empty_frame(size) for _ in range(stack_size)), maxlen=stack_size)


def stack_frames(frames: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = list(frames)
    if not tensors:
        raise ValueError("Cannot stack an empty frame collection")
    return torch.cat(tensors, dim=0)

