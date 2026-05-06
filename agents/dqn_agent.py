"""Trainable DQN agent for Pong image observations."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pong_framework.actions import ActionMetadata, clamp_action
from pong_framework.checkpoints import load_checkpoint, save_checkpoint
from pong_framework.preprocessing import FRAME_SIZE, FRAME_STACK, make_frame_stack, preprocess_frame, stack_frames


class Transition(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._items: Deque[Transition] = deque(maxlen=self.capacity)

    def push(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._items, batch_size)

    def __len__(self) -> int:
        return len(self._items)


class PongDQN(nn.Module):
    def __init__(self, input_channels: int = FRAME_STACK, num_actions: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_dim = self.features(torch.zeros(1, input_channels, FRAME_SIZE, FRAME_SIZE)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@dataclass
class DQNConfig:
    num_actions: int = 6
    frame_stack: int = FRAME_STACK
    replay_capacity: int = 50_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_update_interval: int = 1_000
    min_replay_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "DQNConfig":
        cfg = cls()
        if not values:
            return cfg
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: value for key, value in values.items() if key in valid}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class DQNAgent:
    def __init__(self, ckpt_path: str | Path | None = None):
        payload = load_checkpoint(ckpt_path) if ckpt_path else {}
        self.config = DQNConfig.from_dict(payload.get("config"))
        self.action_metadata = ActionMetadata(actions=tuple(range(self.config.num_actions)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PongDQN(self.config.frame_stack, self.config.num_actions).to(self.device)
        self.target_net = PongDQN(self.config.frame_stack, self.config.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.frames = make_frame_stack(self.config.frame_stack)
        self.training = True
        self.steps_done = 0
        self.stats: dict[str, Any] = {"episodes": 0, "best_eval_reward": -math.inf}

        model_state = payload.get("model_state_dict") or payload.get("model")
        if model_state:
            self.policy_net.load_state_dict(model_state)
        target_state = payload.get("target_state_dict")
        if target_state:
            self.target_net.load_state_dict(target_state)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        self.steps_done = int(payload.get("steps_done", 0))
        self.stats.update(payload.get("stats", {}))
        self.target_net.eval()

    def reset(self) -> None:
        self.frames = make_frame_stack(self.config.frame_stack)

    def train_mode(self) -> None:
        self.training = True
        self.policy_net.train()

    def eval_mode(self) -> None:
        self.training = False
        self.policy_net.eval()

    def action(self, observation, reward, termination, truncation, info) -> int:
        if termination or truncation:
            self.reset()
            return self.action_metadata.default_action
        if observation is None:
            return self.action_metadata.default_action
        state = self.encode_observation(observation)
        return self.select_action(state, explore=self.training)

    def encode_observation(self, observation) -> torch.Tensor:
        frame = preprocess_frame(observation)
        self.frames.append(frame)
        return stack_frames(self.frames)

    def current_state(self) -> torch.Tensor:
        return stack_frames(self.frames)

    def select_action(self, state: torch.Tensor, explore: bool = True) -> int:
        epsilon = self.epsilon if explore else 0.0
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.config.num_actions)
        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0).to(self.device))
        return int(q_values.argmax(dim=1).item())

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.steps_done / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * (1.0 - progress)

    def remember(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        self.replay.push(
            Transition(
                state.detach().cpu(),
                clamp_action(action, self.config.num_actions),
                float(reward),
                next_state.detach().cpu(),
                bool(done),
            )
        )

    def optimize(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.min_replay_size):
            return None
        transitions = self.replay.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target = rewards + (1.0 - dones) * self.config.gamma * next_q
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.steps_done % self.config.target_update_interval == 0:
            self.update_target()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path) -> None:
        save_checkpoint(
            path,
            {
                "agent_type": self.__class__.__name__,
                "config": self.config.to_dict(),
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "action_metadata": self.action_metadata.to_dict(),
                "steps_done": self.steps_done,
                "stats": self.stats,
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
            },
        )


Agent = DQNAgent
