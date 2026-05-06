"""Environment factories with friendly dependency errors."""

from __future__ import annotations


def make_pettingzoo_pong(render_mode: str | None = None):
    try:
        from pettingzoo.atari import pong_v3
    except ImportError as exc:
        raise RuntimeError(
            "PettingZoo Atari is not installed. Install dependencies with "
            "`.venv/bin/python -m pip install -r requirements.txt`."
        ) from exc
    return pong_v3.env(render_mode=render_mode)


def make_gym_pong(render_mode: str | None = None):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("Gymnasium is required for pretraining.") from exc
    try:
        import ale_py

        gym.register_envs(ale_py)
    except Exception:
        pass

    candidates = ("ALE/Pong-v5", "PongNoFrameskip-v4")
    last_error: Exception | None = None
    for env_id in candidates:
        try:
            return gym.make(env_id, render_mode=render_mode)
        except Exception as exc:  # Gym raises several registry/ROM errors.
            last_error = exc
    raise RuntimeError(
        "Could not create a Gymnasium Pong environment. Ensure Atari ROMs are "
        "installed, for example with `AutoROM --accept-license`."
    ) from last_error
