"""Checkpoint helpers used by agents and training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path | None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if path is None:
        return {}
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {ckpt_path}")
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)

