"""Simple local-minimum optimizers for low-dimensional PES models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite

from .calculus import PESPoint, evaluate_pes
from .classification import classify_pes_point
from .matrix import norm

ScalarFunction = Callable[[list[float]], float]


@dataclass(frozen=True)
class MinimizationStep:
    """One gradient-descent minimization iteration."""

    iteration: int
    coordinates: tuple[float, ...]
    energy: float
    gradient_norm: float
    step: tuple[float, ...]
    step_norm: float


@dataclass(frozen=True)
class MinimizationResult:
    """Final result and trace from a local minimization run."""

    converged: bool
    reason: str
    final_coordinates: tuple[float, ...]
    final_energy: float
    final_gradient: tuple[float, ...]
    final_hessian: tuple[tuple[float, ...], ...]
    final_classification: str
    history: tuple[MinimizationStep, ...]


def gradient_descent_minimize(
    energy: ScalarFunction,
    initial_coordinates: Iterable[float],
    max_iterations: int = 200,
    gradient_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-8,
    learning_rate: float = 0.05,
    finite_difference_step: float = 1.0e-5,
    max_step_norm: float = 0.1,
) -> MinimizationResult:
    """Minimize a low-dimensional energy function by capped gradient descent."""

    _validate_minimizer_settings(
        energy=energy,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        step_tolerance=step_tolerance,
        learning_rate=learning_rate,
        finite_difference_step=finite_difference_step,
        max_step_norm=max_step_norm,
    )

    coordinates = _as_coordinate_tuple(initial_coordinates)
    history: list[MinimizationStep] = []

    for iteration in range(max_iterations):
        point = evaluate_pes(
            energy,
            coordinates,
            gradient_step=finite_difference_step,
            hessian_step=finite_difference_step,
        )
        gradient_norm = norm(point.gradient)

        if gradient_norm < gradient_tolerance:
            zero_step = tuple(0.0 for _ in point.coordinates)
            history.append(
                MinimizationStep(
                    iteration=iteration,
                    coordinates=point.coordinates,
                    energy=point.energy,
                    gradient_norm=gradient_norm,
                    step=zero_step,
                    step_norm=0.0,
                )
            )
            return _result(
                converged=True,
                reason="gradient tolerance reached",
                point=point,
                history=history,
                gradient_tolerance=gradient_tolerance,
            )

        step = [-learning_rate * component for component in point.gradient]
        step_norm = norm(step)
        if step_norm > max_step_norm:
            scale = max_step_norm / step_norm
            step = [scale * component for component in step]
            step_norm = norm(step)

        history.append(
            MinimizationStep(
                iteration=iteration,
                coordinates=point.coordinates,
                energy=point.energy,
                gradient_norm=gradient_norm,
                step=tuple(step),
                step_norm=step_norm,
            )
        )

        if step_norm < step_tolerance:
            return _result(
                converged=False,
                reason="step tolerance reached before gradient convergence",
                point=point,
                history=history,
                gradient_tolerance=gradient_tolerance,
            )

        coordinates = tuple(value + delta for value, delta in zip(coordinates, step))

    final_point = evaluate_pes(
        energy,
        coordinates,
        gradient_step=finite_difference_step,
        hessian_step=finite_difference_step,
    )
    return _result(
        converged=False,
        reason="maximum iterations reached without gradient convergence",
        point=final_point,
        history=history,
        gradient_tolerance=gradient_tolerance,
    )


def _result(
    converged: bool,
    reason: str,
    point: PESPoint,
    history: list[MinimizationStep],
    gradient_tolerance: float,
) -> MinimizationResult:
    classification = classify_pes_point(
        point,
        gradient_tolerance=gradient_tolerance,
    )
    return MinimizationResult(
        converged=converged,
        reason=reason,
        final_coordinates=point.coordinates,
        final_energy=point.energy,
        final_gradient=point.gradient,
        final_hessian=point.hessian,
        final_classification=classification.value,
        history=tuple(history),
    )


def _as_coordinate_tuple(coordinates: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in coordinates)
    if not values:
        raise ValueError("initial_coordinates must contain at least one value")
    if any(not isfinite(value) for value in values):
        raise ValueError("initial_coordinates must be finite numbers")
    return values


def _validate_minimizer_settings(
    energy: ScalarFunction,
    max_iterations: int,
    gradient_tolerance: float,
    step_tolerance: float,
    learning_rate: float,
    finite_difference_step: float,
    max_step_norm: float,
) -> None:
    if not callable(energy):
        raise ValueError("energy must be callable")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    for name, value in (
        ("gradient_tolerance", gradient_tolerance),
        ("step_tolerance", step_tolerance),
        ("learning_rate", learning_rate),
        ("finite_difference_step", finite_difference_step),
        ("max_step_norm", max_step_norm),
    ):
        if value <= 0.0 or not isfinite(value):
            raise ValueError(f"{name} must be a positive finite number")
