"""Action-space helpers for Atari Pong.

The ALE/Pong action meanings used by Gymnasium and PettingZoo are normally:
0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT, 4 RIGHTFIRE, 5 LEFTFIRE.

For Pong, RIGHT/LEFT correspond to moving the paddle along the vertical axis
after ALE's screen rotation. We keep the full six-action interface so agents
can be shared between Gymnasium pretraining and PettingZoo play.
"""

from __future__ import annotations

from dataclasses import dataclass


VALID_PONG_ACTIONS = tuple(range(6))
DEFAULT_ACTION = 0


@dataclass(frozen=True)
class ActionMetadata:
    action_space: str = "atari-pong-discrete-v0"
    actions: tuple[int, ...] = VALID_PONG_ACTIONS
    default_action: int = DEFAULT_ACTION

    def to_dict(self) -> dict[str, object]:
        return {
            "action_space": self.action_space,
            "actions": list(self.actions),
            "default_action": self.default_action,
        }


def clamp_action(action: int, n: int = 6) -> int:
    """Return a valid discrete action for an environment with `n` actions."""

    try:
        value = int(action)
    except (TypeError, ValueError):
        return DEFAULT_ACTION
    if value < 0 or value >= n:
        return DEFAULT_ACTION
    return value

