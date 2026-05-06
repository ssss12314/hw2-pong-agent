#!/usr/bin/env python3
"""Run two file-based agents against each other in PettingZoo Pong."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pong_framework.actions import clamp_action
from pong_framework.envs import make_pettingzoo_pong
from pong_framework.loader import load_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Python file containing the left-paddle agent")
    parser.add_argument("--right", required=True, help="Python file containing the right-paddle agent")
    parser.add_argument("--left_ckpt", default=None, help="Optional checkpoint for the left agent")
    parser.add_argument("--right_ckpt", default=None, help="Optional checkpoint for the right agent")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", default=None, help="Optional MP4 path for recording gameplay")
    parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for --video output")
    parser.add_argument(
        "--video_speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for --video output, e.g. 4.0 for 4x faster",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_agent = load_agent(args.left, args.left_ckpt)
    right_agent = load_agent(args.right, args.right_ckpt)
    render_mode = "rgb_array" if args.video else "human" if args.render else None
    env = make_pettingzoo_pong(render_mode=render_mode)
    video_writer = None
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if args.video_speed <= 0:
            raise ValueError("--video_speed must be greater than 0")
        video_writer = imageio.get_writer(
            video_path,
            fps=max(1, int(round(args.video_fps * args.video_speed))),
            codec="libx264",
        )

    wins = {"left": 0, "right": 0, "draw": 0}
    try:
        for episode in range(1, args.episodes + 1):
            env.reset(seed=None)
            _append_video_frame(env, video_writer)
            possible = list(getattr(env, "possible_agents", env.agents))
            if len(possible) < 2:
                raise RuntimeError("Expected a two-agent Pong environment")
            agent_for_env = {possible[0]: left_agent, possible[1]: right_agent}
            side_for_env = {possible[0]: "left", possible[1]: "right"}
            scores = defaultdict(float)

            for step_index, env_agent in enumerate(env.agent_iter(max_iter=args.max_steps)):
                observation, reward, termination, truncation, info = env.last()
                scores[side_for_env[env_agent]] += float(reward)
                if termination or truncation:
                    env.step(None)
                    continue
                player = agent_for_env[env_agent]
                raw_action = player.action(observation, reward, termination, truncation, info)
                n = env.action_space(env_agent).n
                env.step(clamp_action(raw_action, n))
                _append_video_frame(env, video_writer)

            left_score = scores["left"]
            right_score = scores["right"]
            winner = "draw"
            if left_score > right_score:
                winner = "left"
            elif right_score > left_score:
                winner = "right"
            wins[winner] += 1
            print(
                f"episode={episode} left_reward={left_score:.1f} "
                f"right_reward={right_score:.1f} winner={winner}"
            )
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    print(f"summary left_wins={wins['left']} right_wins={wins['right']} draws={wins['draw']}")
    if args.video:
        print(f"video={args.video}")


def _append_video_frame(env, video_writer) -> None:
    if video_writer is None:
        return
    frame = env.render()
    if frame is None:
        return
    if isinstance(frame, list):
        frame = frame[0]
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        video_writer.append_data(array[..., :3])


if __name__ == "__main__":
    main()
