"""CPU-only PID-inspired replay state; time is measured in nominal replay windows.

The learned margin is a fixed target. Signed error is integrated using the right
endpoint. Each observation updates state exactly once; ranking never updates it.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

STATE_COLUMNS = ("pid_error", "pid_integral", "pid_derivative")
LOG_COLUMNS = (*STATE_COLUMNS, "pid_p_term", "pid_i_term", "pid_d_term",
               "peak_margin", "current_margin")


def load_profiles(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "configs" / "experiment_protocol.json"
    profiles = json.loads(path.read_text(encoding="utf-8")).get("pid_replay_profiles", {})
    for name, gains in profiles.items():
        if not name.startswith("pid_"):
            raise ValueError(f"PID profile name must start with pid_: {name}")
        if set(gains) != {"kp", "ki", "kd"}:
            raise ValueError(f"expected kp, ki, kd for {name}")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in gains.values()):
            raise ValueError(f"invalid PID gains for {name}")
        if float(gains["kp"]) != 1.0:
            raise ValueError(f"fix kp=1 to remove redundant gain scaling: {name}")
    return profiles


PROFILES = load_profiles()


def is_pid(method):
    return method in PROFILES


def ensure_state(rows):
    """Initialize new or legacy rows without changing their learned anchors."""
    rows = rows.copy()
    defaults = {"pid_error": rows["peak_margin"] - rows["current_margin"],
                "pid_integral": 0.0, "pid_derivative": 0.0}
    for column, default in defaults.items():
        if column not in rows:
            rows[column] = default
        else:
            rows[column] = rows[column].fillna(default)
    return rows


def update_state(rows, margin, dt):
    """Advance only the pairs present in this observation (Series keyed by id)."""
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("PID observation dt must be positive and finite")
    rows = ensure_state(rows)
    observed = rows["id"].map(margin)
    present = observed.notna()
    if not np.isfinite(observed.loc[present].to_numpy(float)).all():
        raise ValueError("PID margins must be finite")
    error = rows.loc[present, "peak_margin"] - observed.loc[present]
    previous = rows.loc[present, "pid_error"].copy()
    rows.loc[present, "pid_integral"] += error * dt
    rows.loc[present, "pid_derivative"] = (error - previous) / dt
    rows.loc[present, "pid_error"] = error
    rows.loc[present, "current_margin"] = observed.loc[present]
    return rows


def components(rows, method=None, gains=None):
    """Return scores and auditable components; this function never mutates state."""
    gains = PROFILES[method] if gains is None else gains
    out = ensure_state(rows)
    # P always uses the current fixed-anchor error, including the first plan.
    out["pid_error"] = out["peak_margin"] - out["current_margin"]
    out["pid_p_term"] = float(gains["kp"]) * out["pid_error"]
    out["pid_i_term"] = float(gains["ki"]) * out["pid_integral"]
    out["pid_d_term"] = float(gains["kd"]) * out["pid_derivative"]
    out["selection_score"] = out[["pid_p_term", "pid_i_term", "pid_d_term"]].sum(axis=1)
    return out
