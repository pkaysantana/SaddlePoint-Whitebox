"""Finite-difference calculus for potential energy surfaces."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite

from .finite_difference import adaptive_step

ScalarFunction = Callable[[list[float]], float]
StepSpec = float | Sequence[float] | str

DEFAULT_GRADIENT_STEP = 1.0e-5
DEFAULT_HESSIAN_STEP = 1.0e-4


@dataclass(frozen=True)
class PESPoint:
    """Potential energy surface data evaluated at one coordinate vector."""

    coordinates: tuple[float, ...]
    energy: float
    gradient: tuple[float, ...]
    force: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]


def central_difference_gradient(
    energy: ScalarFunction,
    coordinates: Iterable[float],
    step: StepSpec = DEFAULT_GRADIENT_STEP,
) -> list[float]:
    """Return first partial derivatives using O(h^2) central differences.

    For each coordinate i:

        dE/dx_i ~= (E(x_i + h_i) - E(x_i - h_i)) / (2 h_i)

    ``step`` may be a scalar, one positive step per coordinate, or the string
    ``"adaptive"``.
    """

    coords = _as_coordinate_vector(coordinates)
    steps = _as_step_vector(step, coords)

    derivatives: list[float] = []
    for index, delta in enumerate(steps):
        forward = _perturb(coords, ((index, delta),))
        backward = _perturb(coords, ((index, -delta),))
        derivatives.append(
            (_evaluate(energy, forward) - _evaluate(energy, backward)) / (2.0 * delta)
        )
    return derivatives


def central_difference_hessian(
    energy: ScalarFunction,
    coordinates: Iterable[float],
    step: StepSpec = DEFAULT_HESSIAN_STEP,
) -> list[list[float]]:
    """Return the Hessian matrix using central finite differences.

    Diagonal terms use:

        d2E/dx_i2 ~= (E(x_i + h_i) - 2E(x) + E(x_i - h_i)) / h_i^2

    Off-diagonal terms use the mixed central formula:

        d2E/dx_i dx_j ~= (
            E(x_i + h_i, x_j + h_j)
            - E(x_i + h_i, x_j - h_j)
            - E(x_i - h_i, x_j + h_j)
            + E(x_i - h_i, x_j - h_j)
        ) / (4 h_i h_j)
    """

    coords = _as_coordinate_vector(coordinates)
    steps = _as_step_vector(step, coords)
    dimension = len(coords)

    base_energy = _evaluate(energy, coords)
    values = [[0.0 for _ in range(dimension)] for _ in range(dimension)]

    for i, h_i in enumerate(steps):
        forward = _perturb(coords, ((i, h_i),))
        backward = _perturb(coords, ((i, -h_i),))
        values[i][i] = (
            _evaluate(energy, forward) - (2.0 * base_energy) + _evaluate(energy, backward)
        ) / (h_i * h_i)

    for i in range(dimension):
        h_i = steps[i]
        for j in range(i + 1, dimension):
            h_j = steps[j]
            plus_plus = _evaluate(energy, _perturb(coords, ((i, h_i), (j, h_j))))
            plus_minus = _evaluate(energy, _perturb(coords, ((i, h_i), (j, -h_j))))
            minus_plus = _evaluate(energy, _perturb(coords, ((i, -h_i), (j, h_j))))
            minus_minus = _evaluate(energy, _perturb(coords, ((i, -h_i), (j, -h_j))))
            mixed = (plus_plus - plus_minus - minus_plus + minus_minus) / (
                4.0 * h_i * h_j
            )
            values[i][j] = mixed
            values[j][i] = mixed

    return values


def evaluate_pes(
    energy: ScalarFunction,
    coordinates: Iterable[float],
    gradient_step: StepSpec = DEFAULT_GRADIENT_STEP,
    hessian_step: StepSpec = DEFAULT_HESSIAN_STEP,
) -> PESPoint:
    """Evaluate energy, gradient, force, and Hessian at one PES point."""

    coords = _as_coordinate_vector(coordinates)
    energy_value = _evaluate(energy, coords)
    gradient_values = central_difference_gradient(energy, coords, step=gradient_step)
    force_values = [-component for component in gradient_values]
    hessian_values = central_difference_hessian(energy, coords, step=hessian_step)
    return PESPoint(
        coordinates=tuple(coords),
        energy=energy_value,
        gradient=tuple(gradient_values),
        force=tuple(force_values),
        hessian=tuple(tuple(row) for row in hessian_values),
    )


def force(
    energy: ScalarFunction,
    coordinates: Iterable[float],
    step: StepSpec = DEFAULT_GRADIENT_STEP,
) -> list[float]:
    """Return the physical force vector, using F = -grad(E)."""

    return [-component for component in central_difference_gradient(energy, coordinates, step)]


gradient = central_difference_gradient
hessian = central_difference_hessian
curvature = central_difference_hessian


def _as_coordinate_vector(coordinates: Iterable[float]) -> list[float]:
    values = [float(value) for value in coordinates]
    if not values:
        raise ValueError("coordinates must contain at least one value")
    if any(not isfinite(value) for value in values):
        raise ValueError("coordinates must be finite numbers")
    return values


def _as_step_vector(step: StepSpec, coordinates: Sequence[float]) -> list[float]:
    dimension = len(coordinates)
    if isinstance(step, (int, float)):
        steps = [float(step)] * dimension
    elif isinstance(step, str):
        if step != "adaptive":
            raise ValueError('step string must be "adaptive"')
        steps = adaptive_step(coordinates)
    else:
        steps = [float(value) for value in step]
        if len(steps) != dimension:
            raise ValueError("step sequence length must match coordinate dimension")

    if any(value <= 0.0 or not isfinite(value) for value in steps):
        raise ValueError("step values must be positive finite numbers")
    return steps


def _perturb(base: Sequence[float], offsets: Iterable[tuple[int, float]]) -> list[float]:
    perturbed = list(base)
    for index, delta in offsets:
        perturbed[index] += delta
    return perturbed


def _evaluate(energy: ScalarFunction, coordinates: list[float]) -> float:
    value = float(energy(list(coordinates)))
    if not isfinite(value):
        raise ValueError("energy function must return finite scalar values")
    return value
