"""
Centralized semantic similarity thresholds and tunables for PMM.
Values can be overridden via environment variables without code changes.
"""
from __future__ import annotations
import os
from typing import Tuple


def _f(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except Exception:
        return default


# Evidence/Completion detection
COMPLETION_SIM_STRONG = _f("PMM_COMPLETION_SIM_STRONG", 0.70)
COMPLETION_SIM_WITH_EXPLICIT = _f("PMM_COMPLETION_SIM_WITH_EXPLICIT", 0.62)
CLOSURE_EXEMPLAR_SIM_MIN = _f("PMM_CLOSURE_EXEMPLAR_SIM_MIN", 0.60)

# Directive/Commitment candidate detection (semantic pass)
DIRECTIVE_CONF_MIN = _f("PMM_DIRECTIVE_CONF_MIN", 0.10)
DIRECTIVE_C_SIM_MIN = _f("PMM_DIRECTIVE_C_SIM_MIN", 0.42)

# Structural bonuses for directive scoring
STRUCTURE_BONUS_FACTOR = _f("PMM_STRUCTURE_BONUS_FACTOR", 0.04)

# Commitment semantic validation
COMMITMENT_ACCEPT_SIM_MIN = _f("PMM_COMMITMENT_ACCEPT_SIM_MIN", 0.55)
COMMITMENT_VS_PRINCIPLE_MARGIN = _f("PMM_COMMITMENT_VS_PRINCIPLE_MARGIN", 0.05)


def get_directive_thresholds() -> Tuple[float, float, float]:
    """Return (conf_min, c_sim_min, structure_bonus_factor)."""
    return DIRECTIVE_CONF_MIN, DIRECTIVE_C_SIM_MIN, STRUCTURE_BONUS_FACTOR


def get_completion_thresholds() -> Tuple[float, float, float]:
    """Return (strong_sim, explicit_sim, exemplar_sim_min)."""
    return (
        COMPLETION_SIM_STRONG,
        COMPLETION_SIM_WITH_EXPLICIT,
        CLOSURE_EXEMPLAR_SIM_MIN,
    )


def get_commitment_thresholds() -> Tuple[float, float]:
    """Return (accept_sim_min, vs_principle_margin)."""
    return COMMITMENT_ACCEPT_SIM_MIN, COMMITMENT_VS_PRINCIPLE_MARGIN
