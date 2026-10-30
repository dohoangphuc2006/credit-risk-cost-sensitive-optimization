"""Shared experiment settings, resolved relative to this repository."""

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"
DATA_DIR = PROJECT_ROOT / "data"


def load_config() -> dict:
    """Read the canonical configuration and reject invalid core settings."""
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not 0 < config["split"]["test_size"] < 1:
        raise ValueError("split.test_size must be between 0 and 1.")
    if config["cv"]["n_splits"] < 2 or not config["cv"]["shuffle"]:
        raise ValueError("Use at least two shuffled, stratified CV folds.")
    if config["optimization"]["trials_per_seed"] < 1:
        raise ValueError("optimization.trials_per_seed must be positive.")
    seeds = config["optimization"]["sampler_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide distinct sampler seeds.")
    for name, bounds in config["optimization"]["search_space"].items():
        low, high = bounds["low"], bounds["high"]
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"Invalid search bounds for {name}.")
        if bounds["type"] not in {"float", "int"}:
            raise ValueError(f"Unsupported parameter type for {name}.")
        if bounds["type"] == "int" and not all(isinstance(v, int) for v in (low, high)):
            raise ValueError(f"Integer bounds required for {name}.")
    if any(not math.isfinite(v) or v <= 0 for v in config["cost"].values()):
        raise ValueError("Misclassification costs must be finite and positive.")
    policy = config["policy"]
    if not 0 <= policy["max_rejection_rate"] <= 1:
        raise ValueError("policy.max_rejection_rate must be between 0 and 1.")
    if policy["constraint_scope"] != "pooled_train_oof":
        raise ValueError("This protocol measures feasibility on pooled train OOF.")
    multiplier = policy["penalty_multiplier"]
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("policy.penalty_multiplier must be finite and positive.")
    return config
