#!/usr/bin/env python3
"""Pretrain a right-paddle DQN baseline in Gymnasium Pong."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.dqn_agent import DQNAgent
from pong_framework.actions import clamp_action
from pong_framework.envs import make_gym_pong


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--save_path", default="checkpoints/pretrained_right.pt")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = DQNAgent()
    agent.train_mode()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_gym_pong(render_mode="human" if args.render else None)
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            agent.reset()
            state = agent.encode_observation(observation)
            total_reward = 0.0

            for _ in range(args.max_steps):
                action = agent.select_action(state, explore=True)
                next_observation, reward, termination, truncation, _ = env.step(
                    clamp_action(action, env.action_space.n)
                )
                done = bool(termination or truncation)
                next_state = agent.encode_observation(next_observation)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.optimize()
                state = next_state
                total_reward += float(reward)
                if done:
                    break

            agent.stats["episodes"] = int(agent.stats.get("episodes", 0)) + 1
            print(
                f"episode={episode} reward={total_reward:.2f} "
                f"epsilon={agent.epsilon:.3f} loss={loss if loss is not None else 'n/a'}"
            )
            if episode % args.save_interval == 0:
                agent.save(save_path)
                print(f"saved={save_path}")
    finally:
        env.close()
    agent.save(save_path)
    print(f"saved={save_path}")


if __name__ == "__main__":
    main()

