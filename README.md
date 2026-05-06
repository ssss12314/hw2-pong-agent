# Pong Agent Framework

This project hosts two Python agents playing Atari Pong against each other through PettingZoo, plus small DQN training scripts for baseline pretraining and self-play-style evaluation.

## Setup

```bash
.venv/bin/python -m pip install -r requirements.txt
AutoROM --accept-license
```

The repository already expects Python 3.11 in `.venv`. `pettingzoo[atari]` is required for `play.py` and `train.py`; Gymnasium Atari ROMs are required for `pretrain.py`.

## Agent Interface

An agent is a Python file exposing exactly one class, or an `Agent` alias. The class must implement:

```python
class MyAgent:
    def __init__(self, ckpt_path=None):
        ...

    def action(self, observation, reward, termination, truncation, info):
        return 0

    def save(self, path):
        ...
```

`observation` is a single image frame from Pong. Return one discrete Atari Pong action in `[0, 5]`.

## Play

```bash
.venv/bin/python play.py \
  --left agents/random_agent.py \
  --right agents/dqn_agent.py \
  --right_ckpt checkpoints/pretrained_right.pt \
  --episodes 3
```

Add `--render` to watch the game.

Save a match as MP4 with `--video`:

```bash
.venv/bin/python play.py \
  --left agents/dqn_agent.py \
  --left_ckpt checkpoints/left_best.pt \
  --right agents/dqn_agent.py \
  --right_ckpt checkpoints/right_best.pt \
  --episodes 1 \
  --video runs/left_vs_right.mp4 \
  --video_speed 4
```

## Pretrain A Baseline

```bash
.venv/bin/python pretrain.py \
  --episodes 100 \
  --save_path checkpoints/pretrained_right.pt
```

This trains `agents/dqn_agent.py` in Gymnasium Pong and writes a checkpoint compatible with the PettingZoo scripts.

## Train One Side

```bash
.venv/bin/python train.py \
  --left agents/dqn_agent.py \
  --baseline agents/dqn_agent.py \
  --baseline_ckpt checkpoints/pretrained_right.pt \
  --episodes 100 \
  --save_dir checkpoints
```

Use `--right` instead of `--left` to train the other paddle. The trainer periodically evaluates against the frozen baseline and saves `left_best.pt` or `right_best.pt` when mean evaluation reward improves.

## Smoke Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

If `pytest` is installed, this suite is also pytest-compatible:

```bash
.venv/bin/python -m pytest
```
