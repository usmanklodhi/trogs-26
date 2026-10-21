"""Loading YAML configs into dataclasses.

Every configurable step owns a dataclass whose defaults are the values used in
the thesis. A YAML file and ``--set key=value`` overrides are layered on top.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _parse_value(raw: str) -> Any:
    # Reuse YAML scalar parsing so "0.5", "true", "null" and "[1, 2]" behave as expected.
    return yaml.safe_load(raw)


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must look like key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = _parse_value(raw)
    return overrides


def load_config(cls: type[T], path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> T:
    """Build ``cls`` from its defaults, an optional YAML file and explicit overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            values.update(yaml.safe_load(f) or {})
    values.update(overrides or {})

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} option(s): {', '.join(sorted(unknown))}")
    return cls(**values)


def dump_config(cfg: Any, path: str | Path) -> None:
    """Write the resolved config next to run outputs, for reproducibility."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False, allow_unicode=True)
