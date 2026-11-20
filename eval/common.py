"""Shared bits for the evaluation scripts: which assays, where results go."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foldgate.data import ASSAYS, HELD_OUT_FAMILIES
from foldgate.prepare import ZERO_SHOT, per_assay, pooled, zero_shot_frame

__all__ = ["IN_FAMILY", "RUNS", "ZERO_SHOT", "per_assay", "pooled", "write", "zero_shot_frame"]

RUNS = Path("runs")
IN_FAMILY = [a for a in ASSAYS if a not in HELD_OUT_FAMILIES]


def write(name: str, payload: dict[str, Any]) -> Path:
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "results.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path
