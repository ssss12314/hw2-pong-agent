#!/usr/bin/env python3
"""Parallel Pong training with CPU environment workers and one learner."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.my_agent import Transition
from agents.random_agent import RandomAgent
from pong_framework.actions import clamp_action
from pong_framework.envs import make_pettingzoo_pong
from pong_framework.loader import load_agent
from train import _configure_agent, _epsilon, _load_baseline, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--left", help="Python file containing the left-paddle trainable agent")
    side.add_argument("--right", help="Python file containing the right-paddle trainable agent")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint for the learner")
    parser.add_argument("--baseline", default="agents/dqn_agent.py")
    parser.add_argument("--baseline_ckpt", default="checkpoints/pretrained_right.pt")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--optimize_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--min_replay_size", type=int, default=2_000)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--latest_interval", type=int, default=50)
    parser.add_argument("--sync_interval", type=int, default=1_000, help="Learner updates between worker weight syncs.")
    parser.add_argument("--flush_interval", type=int, default=32, help="Transitions per worker result message.")
    parser.add_argument("--torch_threads", type=int, default=6)
    parser.add_argument("--worker_torch_threads", type=int, default=1)
    parser.add_argument("--heuristic_train_probability", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_side = "left" if args.left else "right"
    train_path = args.left or args.right
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    except RuntimeError:
        pass

    learner = load_agent(train_path, args.ckpt)
    _configure_agent(learner, args)
    if hasattr(learner, "train_mode"):
        learner.train_mode()
    best_reward = float(getattr(learner, "stats", {}).get("best_eval_reward", -math.inf))

    baseline_agent = _load_baseline(args.baseline, args.baseline_ckpt)
    if hasattr(baseline_agent, "eval_mode"):
        baseline_agent.eval_mode()

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=max(128, args.num_workers * 8))
    stop_event = ctx.Event()
    weight_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_workers)]
    workers = [
        ctx.Process(
            target=_worker_main,
            args=(worker_id, train_path, args, train_side, weight_queues[worker_id], result_queue, stop_event),
            daemon=True,
        )
        for worker_id in range(args.num_workers)
    ]
    for worker in workers:
        worker.start()
    _broadcast_weights(learner, weight_queues)

    episodes_done = 0
    transitions_seen = 0
    updates = 0
    last_sync_updates = 0
    losses: list[float] = []
    started_at = time.perf_counter()

    try:
        while episodes_done < args.episodes:
            try:
                message = result_queue.get(timeout=5.0)
            except queue.Empty:
                _raise_if_worker_died(workers)
                continue

            kind = message[0]
            if kind == "transitions":
                batch = message[1]
                for state, action, reward, next_state, done in batch:
                    learner.replay.push(
                        Transition(
                            torch.from_numpy(state),
                            int(action),
                            float(reward),
                            torch.from_numpy(next_state),
                            bool(done),
                        )
                    )
                    transitions_seen += 1
                    for _ in range(max(1, args.optimize_steps)):
                        loss = learner.optimize()
                        if loss is not None:
                            losses.append(loss)
                            updates += 1
                if updates - last_sync_updates >= args.sync_interval:
                    _broadcast_weights(learner, weight_queues)
                    last_sync_updates = updates
            elif kind == "episode":
                _, worker_id, reward, env_steps = message
                episodes_done += 1
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                mean_loss = sum(losses[-200:]) / min(len(losses), 200) if losses else None
                loss_text = "n/a" if mean_loss is None else f"{mean_loss:.5f}"
                print(
                    f"episode={episodes_done}/{args.episodes} worker={worker_id} train_side={train_side} "
                    f"reward={reward:.2f} epsilon={_epsilon(learner):.3f} env_steps={env_steps} "
                    f"transitions={transitions_seen} updates={updates} rate={transitions_seen / elapsed:.1f}/s "
                    f"loss={loss_text}",
                    flush=True,
                )

                if args.latest_interval > 0 and episodes_done % args.latest_interval == 0:
                    latest_path = save_dir / f"{train_side}_latest.pt"
                    learner.save(latest_path)
                    print(f"saved_latest={latest_path}", flush=True)

                if args.eval_interval > 0 and episodes_done % args.eval_interval == 0:
                    eval_reward = evaluate(learner, baseline_agent, train_side, args.eval_episodes, args.max_steps)
                    print(f"eval episode={episodes_done} mean_reward={eval_reward:.2f} best={best_reward:.2f}", flush=True)
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        learner.stats["best_eval_reward"] = best_reward
                        best_path = save_dir / f"{train_side}_best.pt"
                        learner.save(best_path)
                        print(f"saved={best_path}", flush=True)
                    if hasattr(learner, "train_mode"):
                        learner.train_mode()
            _raise_if_worker_died(workers)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=5)
        learner.save(save_dir / f"{train_side}_latest.pt")
        print(f"saved_latest={save_dir / f'{train_side}_latest.pt'}", flush=True)


def _worker_main(
    worker_id: int,
    train_path: str,
    args: argparse.Namespace,
    train_side: str,
    weight_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    torch.set_num_threads(max(1, args.worker_torch_threads))
    agent = load_agent(train_path, args.ckpt)
    _configure_agent(agent, args)
    _force_cpu(agent)
    baseline_agent = _load_worker_baseline(args.baseline, args.baseline_ckpt)
    _force_cpu(baseline_agent)
    pending: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    while not stop_event.is_set():
        _drain_weights(agent, weight_queue)
        reward, env_steps, transitions = _run_worker_episode(
            agent,
            baseline_agent,
            train_side,
            args.max_steps,
        )
        pending.extend(transitions)
        while len(pending) >= args.flush_interval:
            result_queue.put(("transitions", pending[: args.flush_interval]))
            del pending[: args.flush_interval]
        if pending:
            result_queue.put(("transitions", pending))
            pending = []
        result_queue.put(("episode", worker_id, reward, env_steps))


def _run_worker_episode(agent, baseline_agent, train_side: str, max_steps: int):
    env = make_pettingzoo_pong(render_mode=None)
    total_reward = 0.0
    env_steps = 0
    transitions = []
    try:
        env.reset()
        env_agents = list(getattr(env, "possible_agents", env.agents))
        train_env_agent = env_agents[0] if train_side == "left" else env_agents[1]
        player_for_env = {
            train_env_agent: agent,
            env_agents[1] if train_side == "left" else env_agents[0]: baseline_agent,
        }
        last_state = None
        last_action = None
        if hasattr(agent, "train_mode"):
            agent.train_mode()
        for player in player_for_env.values():
            if hasattr(player, "reset"):
                player.reset()

        for env_agent in env.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = env.last()
            done = bool(termination or truncation)
            if env_agent == train_env_agent:
                env_steps += 1
                total_reward += float(reward)
                state = agent.encode_observation(observation) if observation is not None else agent.current_state()
                if last_state is not None and last_action is not None:
                    transitions.append((_pack_state(last_state), last_action, float(reward), _pack_state(state), done))
                if done:
                    env.step(None)
                    last_state = None
                    last_action = None
                    continue
                action = agent.select_action(state, explore=True)
                last_state = state
                last_action = int(action)
                env.step(clamp_action(action, env.action_space(env_agent).n))
            else:
                if done:
                    env.step(None)
                    continue
                action = player_for_env[env_agent].action(observation, reward, termination, truncation, info)
                env.step(clamp_action(action, env.action_space(env_agent).n))
    finally:
        env.close()
    return total_reward, env_steps, transitions


def _pack_state(state: torch.Tensor) -> np.ndarray:
    return state.detach().clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()


def _force_cpu(agent) -> None:
    if hasattr(agent, "device"):
        agent.device = torch.device("cpu")
    for name in ("policy_net", "target_net"):
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "to"):
            module.to("cpu")


def _load_worker_baseline(agent_path: str, ckpt_path: str):
    if Path(ckpt_path).exists():
        return load_agent(agent_path, ckpt_path)
    return RandomAgent()


def _broadcast_weights(agent, weight_queues: list[mp.Queue]) -> None:
    policy_net = getattr(agent, "policy_net", None)
    if policy_net is None:
        return
    state = {key: value.detach().cpu() for key, value in policy_net.state_dict().items()}
    for weight_queue in weight_queues:
        try:
            while True:
                weight_queue.get_nowait()
        except queue.Empty:
            pass
        weight_queue.put(state)


def _drain_weights(agent, weight_queue: mp.Queue) -> None:
    latest: dict[str, Any] | None = None
    try:
        while True:
            latest = weight_queue.get_nowait()
    except queue.Empty:
        pass
    if latest is None:
        return
    policy_net = getattr(agent, "policy_net", None)
    target_net = getattr(agent, "target_net", None)
    if policy_net is not None:
        policy_net.load_state_dict(latest)
    if target_net is not None:
        target_net.load_state_dict(latest)


def _raise_if_worker_died(workers: list[mp.Process]) -> None:
    for worker in workers:
        if worker.exitcode not in (None, 0):
            raise RuntimeError(f"worker pid={worker.pid} exited with code {worker.exitcode}")


if __name__ == "__main__":
    main()
