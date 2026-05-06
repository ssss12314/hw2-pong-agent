"""A tiny baseline agent that samples uniformly from Pong's discrete actions."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from pong_framework.actions import ActionMetadata, VALID_PONG_ACTIONS
from pong_framework.checkpoints import load_checkpoint, save_checkpoint


class RandomAgent:
    def __init__(self, ckpt_path: str | Path | None = None):
        self.action_metadata = ActionMetadata()
        self.stats: dict[str, Any] = {}
        if ckpt_path:
            payload = load_checkpoint(ckpt_path)
            self.stats = dict(payload.get("stats", {}))

    def action(self, observation, reward, termination, truncation, info) -> int:
        if termination or truncation:
            return self.action_metadata.default_action
        return random.choice(VALID_PONG_ACTIONS)

    def save(self, path: str | Path) -> None:
        save_checkpoint(
            path,
            {
                "agent_type": self.__class__.__name__,
                "action_metadata": self.action_metadata.to_dict(),
                "stats": self.stats,
                "torch_version": str(torch.__version__),
            },
        )


Agent = RandomAgent
