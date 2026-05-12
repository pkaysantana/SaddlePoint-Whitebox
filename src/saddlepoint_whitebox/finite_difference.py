"""Finite-difference step-size helpers."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite


def adaptive_step(
    coordinates: Iterable[float],
    base_step: float = 1.0e-5,
    minimum_step: float = 1.0e-7,
    maximum_step: float = 1.0e-3,
) -> list[float]:
    """Return one modestly scale-aware finite-difference step per coordinate.

    The step grows with coordinate magnitude so large coordinates do not use an
    unnecessarily tiny displacement, but it is clamped to keep the stencil local
    and to avoid underflow near stationary points.
    """

    values = [float(value) for value in coordinates]
    if not values:
        raise ValueError("coordinates must contain at least one value")
    if any(not isfinite(value) for value in values):
        raise ValueError("coordinates must be finite numbers")

    _validate_step_bound(base_step, "base_step")
    _validate_step_bound(minimum_step, "minimum_step")
    _validate_step_bound(maximum_step, "maximum_step")
    if minimum_step > maximum_step:
        raise ValueError("minimum_step cannot be greater than maximum_step")

    steps: list[float] = []
    for value in values:
        raw_step = base_step * max(1.0, abs(value))
        steps.append(min(max(raw_step, minimum_step), maximum_step))
    return steps


def _validate_step_bound(value: float, name: str) -> None:
    if value <= 0.0 or not isfinite(value):
        raise ValueError(f"{name} must be a positive finite number")
