"""
Robot config registry.

Adding a robot = adding `omni_spot/configs/<name>.py` exposing make_cfg().
No training code branches on the robot name — selection is import-by-name.
"""

from __future__ import annotations

import importlib
import pkgutil

from .base import ExperimentCfg


def available_robots() -> list[str]:
    """Names of config modules in this package (excluding base)."""
    pkg_path = __path__  # type: ignore[name-defined]
    return sorted(
        m.name for m in pkgutil.iter_modules(pkg_path)
        if m.name != "base" and not m.name.startswith("_")
    )


def get_experiment_cfg(robot: str) -> ExperimentCfg:
    """Load `omni_spot.configs.<robot>` and return a fresh ExperimentCfg."""
    try:
        mod = importlib.import_module(f"{__name__}.{robot}")
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unknown robot config '{robot}'. Available: {available_robots()}"
        ) from e
    if not hasattr(mod, "make_cfg"):
        raise ValueError(
            f"Config module '{robot}' must define make_cfg() -> ExperimentCfg"
        )
    cfg = mod.make_cfg()
    if not isinstance(cfg, ExperimentCfg):
        raise TypeError(
            f"{robot}.make_cfg() returned {type(cfg).__name__}, expected ExperimentCfg"
        )
    return cfg
