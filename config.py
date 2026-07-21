"""
config.py
=========
Typed schema for config.yaml.  Edit config.yaml to change parameters — not this file.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional

import yaml


@dataclass
class Config:
    """Internal configuration container — all values come from config.yaml."""

    # Recording
    epoch_s: int
    min_recording_hours: float
    analysis_day_start_hour: int

    # Algorithms
    roe_threshold: float
    roe_min_seed_period: str
    nonwear_min_consecutive_zero_min: float

    # Naps
    nap_min_duration_min: int
    nap_max_duration_min: int
    daytime_start: str
    daytime_end: str

    # Night metrics (raw activity)
    active_epoch_threshold: float
    sb_min_consecutive_immobile_epochs: int
    wb_min_consecutive_movement_epochs: int

    # Output / UI
    output_dir: str
    state_filename: str
    log_filename: str
    discard_reasons: List[str]
    diary_nights_of_interest: Optional[List[int]] = None  # null in YAML = all nights

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        try:
            return cls(**filtered)
        except TypeError as exc:
            raise SystemExit(
                f"\n[config error] {exc}\n"
                f"Check that config.yaml contains all required keys.\n"
                f"File loaded: {path}"
            ) from None
