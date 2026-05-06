"""Dynamic loading for user-supplied agent Python files."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REQUIRED_METHODS = ("action", "save")


def import_module_from_path(path: str | Path) -> ModuleType:
    agent_path = Path(path).resolve()
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")
    module_name = f"pong_agent_{agent_path.stem}_{abs(hash(agent_path))}"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import agent module from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_agent_class(module: ModuleType) -> type[Any]:
    explicit = getattr(module, "Agent", None)
    if inspect.isclass(explicit) and _looks_like_agent(explicit):
        return explicit

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and _looks_like_agent(obj)
    ]
    if len(candidates) != 1:
        names = ", ".join(cls.__name__ for cls in candidates) or "none"
        raise ValueError(
            "Expected exactly one agent class or an `Agent` alias in "
            f"{getattr(module, '__file__', module.__name__)}; found {names}"
        )
    return candidates[0]


def load_agent(path: str | Path, ckpt_path: str | Path | None = None) -> Any:
    module = import_module_from_path(path)
    cls = discover_agent_class(module)
    return cls(ckpt_path=ckpt_path)


def _looks_like_agent(cls: type[Any]) -> bool:
    init = getattr(cls, "__init__", None)
    if init is None:
        return False
    return all(callable(getattr(cls, method, None)) for method in REQUIRED_METHODS)
