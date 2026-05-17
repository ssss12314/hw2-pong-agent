"""Improved DQN agent for Atari Pong.

This agent keeps the public homework interface while adding a visual Pong
tracker, a dueling network, Double DQN targets, cropped preprocessing, and
CUDA/CPU device selection.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Literal, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from pong_framework.actions import ActionMetadata, clamp_action
from pong_framework.checkpoints import load_checkpoint, save_checkpoint
from pong_framework.preprocessing import FRAME_SIZE, FRAME_STACK


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


@dataclass
class VisionObservation:
    ball: tuple[float, float] | None
    left_paddle_y: float | None
    right_paddle_y: float | None
    width: int
    height: int


class VisionTracker:
    """Small Pong-specific tracker used as a policy prior during evaluation."""

    def __init__(self, move_down_action: int = 2, move_up_action: int = 3):
        self.move_down_action = move_down_action
        self.move_up_action = move_up_action
        self.previous: VisionObservation | None = None
        self.previous_action: int | None = None
        self.control_scores = {"left": 0.0, "right": 0.0}
        self.controlled_side: Literal["left", "right"] | None = None

    def reset(self) -> None:
        self.previous = None
        self.previous_action = None
        self.control_scores = {"left": 0.0, "right": 0.0}
        self.controlled_side = None

    def observe(self, observation) -> VisionObservation:
        current = self._detect(observation)
        self._update_controlled_side(current)
        return current

    def choose_action(self, current: VisionObservation) -> int | None:
        if current.ball is None:
            return None

        side = self.controlled_side or self._side_ball_is_approaching(current)
        paddle_y = current.left_paddle_y if side == "left" else current.right_paddle_y
        if paddle_y is None:
            paddle_y = current.right_paddle_y if current.right_paddle_y is not None else current.left_paddle_y
        if paddle_y is None:
            return None

        target_y = self._intercept_y(current, side)
        deadzone = 4.0
        if paddle_y < target_y - deadzone:
            return self.move_down_action
        if paddle_y > target_y + deadzone:
            return self.move_up_action
        return 0

    def remember_action(self, action: int, current: VisionObservation) -> None:
        self.previous = current
        self.previous_action = action

    def _update_controlled_side(self, current: VisionObservation) -> None:
        if self.previous is None or self.previous_action is None:
            return
        expected = self._action_direction(self.previous_action)
        if expected == 0:
            return
        for side, attr in (("left", "left_paddle_y"), ("right", "right_paddle_y")):
            before = getattr(self.previous, attr)
            after = getattr(current, attr)
            if before is None or after is None:
                continue
            delta = after - before
            if abs(delta) < 0.5:
                continue
            self.control_scores[side] += expected * np.sign(delta)
        if abs(self.control_scores["left"] - self.control_scores["right"]) >= 2.0:
            self.controlled_side = "left" if self.control_scores["left"] > self.control_scores["right"] else "right"

    def _intercept_y(self, current: VisionObservation, side: Literal["left", "right"]) -> float:
        ball_x, ball_y = current.ball if current.ball is not None else (current.width / 2.0, current.height / 2.0)
        if self.previous is None or self.previous.ball is None:
            return ball_y
        prev_x, prev_y = self.previous.ball
        vx = ball_x - prev_x
        vy = ball_y - prev_y
        target_x = 12.0 if side == "left" else current.width - 12.0
        if abs(vx) < 0.1 or (side == "left" and vx > 0) or (side == "right" and vx < 0):
            return ball_y
        projected_y = ball_y + vy * ((target_x - ball_x) / vx)
        return self._reflect(projected_y, current.height - 1)

    @staticmethod
    def _reflect(y: float, bottom: int) -> float:
        if bottom <= 0:
            return y
        period = 2.0 * bottom
        folded = y % period
        if folded > bottom:
            folded = period - folded
        return float(np.clip(folded, 0.0, bottom))

    def _side_ball_is_approaching(self, current: VisionObservation) -> Literal["left", "right"]:
        if self.previous is not None and self.previous.ball is not None and current.ball is not None:
            return "left" if current.ball[0] < self.previous.ball[0] else "right"
        return "right"

    def _action_direction(self, action: int) -> int:
        if action in (self.move_down_action, 4):
            return 1
        if action in (self.move_up_action, 5):
            return -1
        return 0

    def _detect(self, observation) -> VisionObservation:
        gray = MyAgent._observation_to_grayscale(observation)
        crop = gray[34:194, :] if gray.shape[0] >= 194 else gray
        mask = crop > 90
        height, width = crop.shape
        left_y = self._detect_paddle_y(mask, 0, max(1, width // 4))
        right_y = self._detect_paddle_y(mask, min(width - 1, 3 * width // 4), width)
        ball = self._detect_ball(mask)
        return VisionObservation(ball=ball, left_paddle_y=left_y, right_paddle_y=right_y, width=width, height=height)

    @staticmethod
    def _detect_paddle_y(mask: np.ndarray, start_x: int, end_x: int) -> float | None:
        region = mask[:, start_x:end_x]
        if region.size == 0:
            return None
        column_counts = region.sum(axis=0)
        active_columns = np.flatnonzero(column_counts >= 6)
        if active_columns.size == 0:
            return None
        ys, xs = np.nonzero(region[:, active_columns])
        if ys.size == 0:
            return None
        return float(np.median(ys))

    @staticmethod
    def _detect_ball(mask: np.ndarray) -> tuple[float, float] | None:
        height, width = mask.shape
        work = mask.copy()
        work[:, : max(1, width // 8)] = False
        work[:, min(width - 1, 7 * width // 8) :] = False
        visited = np.zeros_like(work, dtype=bool)
        best: tuple[float, float] | None = None
        best_score = float("inf")

        for y0, x0 in zip(*np.nonzero(work)):
            if visited[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            points: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and work[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            area = len(points)
            if area < 2 or area > 80:
                continue
            ys = np.array([p[0] for p in points])
            xs = np.array([p[1] for p in points])
            box_h = int(ys.max() - ys.min() + 1)
            box_w = int(xs.max() - xs.min() + 1)
            if box_h > 10 or box_w > 10:
                continue
            score = abs(box_h - box_w) + area * 0.03
            if score < best_score:
                best_score = score
                best = (float(xs.mean()), float(ys.mean()))
        return best


class DuelingPongDQN(nn.Module):
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
        self.value = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class AgentConfig:
    num_actions: int = 6
    frame_stack: int = FRAME_STACK
    replay_capacity: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_update_interval: int = 2_500
    min_replay_size: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 500_000
    reward_clip: float = 1.0
    heuristic_enabled: bool = True
    heuristic_train_probability: float = 0.25
    move_down_action: int = 2
    move_up_action: int = 3

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "AgentConfig":
        if not values:
            return cls()
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class MyAgent:
    def __init__(self, ckpt_path: str | Path | None = None):
        self.device = self._select_device()
        payload = load_checkpoint(ckpt_path, map_location=self.device) if ckpt_path else {}
        self.config = AgentConfig.from_dict(payload.get("config"))
        self.action_metadata = ActionMetadata(actions=tuple(range(self.config.num_actions)))
        self.policy_net = DuelingPongDQN(self.config.frame_stack, self.config.num_actions).to(self.device)
        self.target_net = DuelingPongDQN(self.config.frame_stack, self.config.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.frames = self._make_frame_stack()
        self.tracker = VisionTracker(self.config.move_down_action, self.config.move_up_action)
        self.training = False
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
        self.frames = self._make_frame_stack()
        self.tracker.reset()

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
        tracked = self.tracker.observe(observation)
        action = self._heuristic_action(tracked)
        if action is None:
            action = self.select_action(state, explore=self.training)
        self.tracker.remember_action(action, tracked)
        return clamp_action(action, self.config.num_actions)

    def encode_observation(self, observation) -> torch.Tensor:
        frame = self._preprocess_frame(observation)
        self.frames.append(frame)
        return self.current_state()

    def current_state(self) -> torch.Tensor:
        return torch.cat(list(self.frames), dim=0)

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
        clipped_reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        self.replay.push(
            Transition(
                self._pack_state(state),
                clamp_action(action, self.config.num_actions),
                clipped_reward,
                self._pack_state(next_state),
                bool(done),
            )
        )

    def optimize(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.min_replay_size):
            return None
        transitions = self.replay.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))

        states = self._unpack_batch(batch.state)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = self._unpack_batch(batch.next_state)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
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

    @staticmethod
    def _pack_state(state: torch.Tensor) -> torch.Tensor:
        return (state.detach().clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu())

    def _unpack_batch(self, states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.stack(states).to(self.device).float().div_(255.0)

    def _heuristic_action(self, tracked: VisionObservation) -> int | None:
        if not self.config.heuristic_enabled:
            return None
        if self.training and random.random() > self.config.heuristic_train_probability:
            return None
        return self.tracker.choose_action(tracked)

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

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _make_frame_stack(self) -> Deque[torch.Tensor]:
        return deque((torch.zeros((1, FRAME_SIZE, FRAME_SIZE), dtype=torch.float32) for _ in range(self.config.frame_stack)), maxlen=self.config.frame_stack)

    @staticmethod
    def _preprocess_frame(observation) -> torch.Tensor:
        gray = MyAgent._observation_to_grayscale(observation)
        crop = gray[34:194, :] if gray.shape[0] >= 194 else gray
        image = Image.fromarray(crop.astype(np.uint8), mode="L")
        image = image.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values).unsqueeze(0)

    @staticmethod
    def _observation_to_grayscale(observation) -> np.ndarray:
        array = np.asarray(observation.detach().cpu().numpy() if isinstance(observation, torch.Tensor) else observation)
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
        return np.asarray(image, dtype=np.uint8)


Agent = MyAgent
