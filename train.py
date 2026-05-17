#!/usr/bin/env python3
"""Train one PettingZoo Pong paddle against a frozen baseline opponent."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from agents.random_agent import RandomAgent
from pong_framework.actions import clamp_action
from pong_framework.envs import make_pettingzoo_pong
from pong_framework.loader import load_agent


@dataclass
class TrainingResult:
    reward: float
    env_steps: int
    optimizes: int
    mean_loss: float | None
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--left", help="Python file containing the left-paddle trainable agent")
    side.add_argument("--right", help="Python file containing the right-paddle trainable agent")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint for the trainable agent")
    parser.add_argument("--baseline", default="agents/dqn_agent.py", help="Python file for the frozen baseline")
    parser.add_argument("--baseline_ckpt", default="checkpoints/pretrained_right.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--optimize_steps",
        type=int,
        default=1,
        help="Gradient updates after each collected train transition. Increase to use more GPU.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override train agent batch size.")
    parser.add_argument("--min_replay_size", type=int, default=None, help="Override warmup transitions before updates.")
    parser.add_argument(
        "--heuristic_train_probability",
        type=float,
        default=None,
        help="For hybrid agents, probability of using the visual heuristic while training.",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="PyTorch CPU threads. Use 0 to auto-pick a sensible value for the host CPU.",
    )
    parser.add_argument("--latest_interval", type=int, default=25, help="Save latest checkpoint every N episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_threads = _resolve_torch_threads(args.torch_threads)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, torch_threads)))
    except RuntimeError:
        pass
    print(f"torch_threads={torch.get_num_threads()}")
    train_side = "left" if args.left else "right"
    train_path = args.left or args.right
    train_agent = load_agent(train_path, args.ckpt)
    _require_trainable(train_agent)
    _configure_agent(train_agent, args)

    baseline_agent = _load_baseline(args.baseline, args.baseline_ckpt)
    if hasattr(baseline_agent, "eval_mode"):
        baseline_agent.eval_mode()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_reward = float(getattr(train_agent, "stats", {}).get("best_eval_reward", -math.inf))

    for episode in range(1, args.episodes + 1):
        result = run_training_episode(
            train_agent,
            baseline_agent,
            train_side,
            args.max_steps,
            args.render,
            max(1, args.optimize_steps),
        )
        steps_per_second = result.env_steps / max(result.elapsed_seconds, 1e-9)
        loss_text = "n/a" if result.mean_loss is None else f"{result.mean_loss:.5f}"
        print(
            f"episode={episode} train_side={train_side} reward={result.reward:.2f} "
            f"epsilon={_epsilon(train_agent):.3f} steps={result.env_steps} "
            f"ups={result.optimizes} fps={steps_per_second:.1f} loss={loss_text}"
        )
        if args.latest_interval > 0 and episode % args.latest_interval == 0:
            latest_path = save_dir / f"{train_side}_latest.pt"
            train_agent.save(latest_path)
            print(f"saved_latest={latest_path}")

        if episode % args.eval_interval == 0:
            eval_reward = evaluate(train_agent, baseline_agent, train_side, args.eval_episodes, args.max_steps)
            print(f"eval episode={episode} mean_reward={eval_reward:.2f} best={best_reward:.2f}")
            if eval_reward > best_reward:
                best_reward = eval_reward
                train_agent.stats["best_eval_reward"] = best_reward
                path = save_dir / f"{train_side}_best.pt"
                train_agent.save(path)
                print(f"saved={path}")


def run_training_episode(
    train_agent,
    baseline_agent,
    train_side: str,
    max_steps: int,
    render: bool = False,
    optimize_steps: int = 1,
) -> TrainingResult:
    env = make_pettingzoo_pong(render_mode="human" if render else None)
    total_reward = 0.0
    env_steps = 0
    optimizes = 0
    losses = []
    started_at = time.perf_counter()
    try:
        env.reset()
        env_agents = list(getattr(env, "possible_agents", env.agents))
        train_env_agent = env_agents[0] if train_side == "left" else env_agents[1]
        player_for_env = {
            train_env_agent: train_agent,
            env_agents[1] if train_side == "left" else env_agents[0]: baseline_agent,
        }
        last_state = None
        last_action = None
        if hasattr(train_agent, "train_mode"):
            train_agent.train_mode()
        if hasattr(train_agent, "reset"):
            train_agent.reset()
        if hasattr(baseline_agent, "reset"):
            baseline_agent.reset()

        for env_agent in env.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = env.last()
            done = bool(termination or truncation)
            player = player_for_env[env_agent]
            if env_agent == train_env_agent:
                env_steps += 1
                total_reward += float(reward)
                if observation is not None:
                    state = train_agent.encode_observation(observation)
                else:
                    state = train_agent.current_state()
                if last_state is not None and last_action is not None:
                    train_agent.remember(last_state, last_action, reward, state, done)
                    for _ in range(optimize_steps):
                        loss = train_agent.optimize()
                        if loss is not None:
                            losses.append(loss)
                            optimizes += 1
                if done:
                    env.step(None)
                    last_state = None
                    last_action = None
                    continue
                action = train_agent.select_action(state, explore=True)
                last_state = state
                last_action = action
                env.step(clamp_action(action, env.action_space(env_agent).n))
            else:
                if done:
                    env.step(None)
                    continue
                action = player.action(observation, reward, termination, truncation, info)
                env.step(clamp_action(action, env.action_space(env_agent).n))
    finally:
        env.close()
    mean_loss = sum(losses) / len(losses) if losses else None
    return TrainingResult(
        reward=total_reward,
        env_steps=env_steps,
        optimizes=optimizes,
        mean_loss=mean_loss,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def evaluate(train_agent, baseline_agent, train_side: str, episodes: int, max_steps: int) -> float:
    rewards = []
    if hasattr(train_agent, "eval_mode"):
        train_agent.eval_mode()
    for _ in range(episodes):
        rewards.append(run_eval_episode(train_agent, baseline_agent, train_side, max_steps))
    if hasattr(train_agent, "train_mode"):
        train_agent.train_mode()
    return sum(rewards) / max(1, len(rewards))


def run_eval_episode(train_agent, baseline_agent, train_side: str, max_steps: int) -> float:
    env = make_pettingzoo_pong(render_mode=None)
    total_reward = 0.0
    try:
        env.reset()
        env_agents = list(getattr(env, "possible_agents", env.agents))
        train_env_agent = env_agents[0] if train_side == "left" else env_agents[1]
        player_for_env = {
            train_env_agent: train_agent,
            env_agents[1] if train_side == "left" else env_agents[0]: baseline_agent,
        }
        for player in player_for_env.values():
            if hasattr(player, "reset"):
                player.reset()
        for env_agent in env.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = env.last()
            if env_agent == train_env_agent:
                total_reward += float(reward)
            if termination or truncation:
                env.step(None)
                continue
            action = player_for_env[env_agent].action(observation, reward, termination, truncation, info)
            env.step(clamp_action(action, env.action_space(env_agent).n))
    finally:
        env.close()
    return total_reward


def _require_trainable(agent) -> None:
    required = ("encode_observation", "select_action", "remember", "optimize", "current_state", "save")
    missing = [name for name in required if not callable(getattr(agent, name, None))]
    if missing:
        raise TypeError(f"Trainable agent is missing required methods: {', '.join(missing)}")


def _load_baseline(agent_path: str, ckpt_path: str):
    if Path(ckpt_path).exists():
        return load_agent(agent_path, ckpt_path)
    print(f"warning: baseline checkpoint {ckpt_path} not found; using RandomAgent fallback")
    return RandomAgent()


def _epsilon(agent) -> float:
    return float(getattr(agent, "epsilon", 0.0))


def _resolve_torch_threads(value: int) -> int:
    if value > 0:
        return value
    cpu_count = os.cpu_count() or 1
    return max(1, min(6, cpu_count))


def _configure_agent(agent, args: argparse.Namespace) -> None:
    config = getattr(agent, "config", None)
    if config is None:
        return
    overrides = {
        "batch_size": args.batch_size,
        "min_replay_size": args.min_replay_size,
        "heuristic_train_probability": args.heuristic_train_probability,
    }
    changed = []
    for name, value in overrides.items():
        if value is None or not hasattr(config, name):
            continue
        setattr(config, name, value)
        changed.append(f"{name}={value}")
    if changed:
        print("agent_config_overrides " + " ".join(changed))


if __name__ == "__main__":
    main()
