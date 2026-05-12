"""Basic transition-state optimizers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite

from .calculus import PESPoint, evaluate_pes
from .classification import StationaryPointType, classify_pes_point
from .evf import EVFOptimizer, EVFSettings
from .matrix import jacobi_eigendecomposition_symmetric, norm

ScalarFunction = Callable[[list[float]], float]

EIGENVALUE_SHIFT = 1.0e-8


@dataclass(frozen=True)
class OptimizationStep:
    """One iteration of an eigenvector-following saddle search."""

    iteration: int
    coordinates: tuple[float, ...]
    energy: float
    gradient_norm: float
    eigenvalues: tuple[float, ...]
    step: tuple[float, ...]
    step_norm: float
    classification: StationaryPointType


@dataclass(frozen=True)
class OptimizationResult:
    """Final result and trace from a saddle search."""

    converged: bool
    reason: str
    final_coordinates: tuple[float, ...]
    final_energy: float
    final_gradient: tuple[float, ...]
    final_hessian: tuple[tuple[float, ...], ...]
    final_eigenvalues: tuple[float, ...]
    final_classification: StationaryPointType
    history: tuple[OptimizationStep, ...]


def eigenvector_following_saddle_search(
    energy: ScalarFunction,
    initial_coordinates: Iterable[float],
    max_iterations: int = 100,
    gradient_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-8,
    trust_radius: float = 0.1,
    finite_difference_step: float = 1.0e-5,
) -> OptimizationResult:
    """Search for a first-order saddle using basic eigenvector following.

    This intentionally small implementation is aimed at simple analytic
    low-dimensional surfaces. At each step the Hessian eigenvectors define local
    modes. The lowest mode is treated as the reaction coordinate and the step is
    reversed relative to a minimizer so the search climbs that mode, while all
    other modes descend toward zero gradient.
    """

    _validate_optimizer_settings(
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        step_tolerance=step_tolerance,
        trust_radius=trust_radius,
        finite_difference_step=finite_difference_step,
    )

    coordinates = _as_coordinate_tuple(initial_coordinates)
    history: list[OptimizationStep] = []
    evf_optimizer = EVFOptimizer(
        EVFSettings(
            target_mode_index=0,
            trust_radius=trust_radius,
            eigenvalue_shift=EIGENVALUE_SHIFT,
            climb_on_target_mode=True,
        )
    )

    for iteration in range(max_iterations):
        point = evaluate_pes(
            energy,
            coordinates,
            gradient_step=finite_difference_step,
            hessian_step=finite_difference_step,
        )
        eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(point.hessian)
        classification = classify_pes_point(
            point,
            gradient_tolerance=gradient_tolerance,
        )
        gradient_norm = norm(point.gradient)

        if (
            gradient_norm <= gradient_tolerance
            and classification == StationaryPointType.FIRST_ORDER_SADDLE
        ):
            zero_step = tuple(0.0 for _ in coordinates)
            history.append(
                OptimizationStep(
                    iteration=iteration,
                    coordinates=point.coordinates,
                    energy=point.energy,
                    gradient_norm=gradient_norm,
                    eigenvalues=tuple(eigenvalues),
                    step=zero_step,
                    step_norm=0.0,
                    classification=classification,
                )
            )
            return _result(
                converged=True,
                reason="gradient tolerance reached at a first-order saddle",
                point=point,
                eigenvalues=eigenvalues,
                classification=classification,
                history=history,
            )

        step_result = evf_optimizer.compute_step(
            point.gradient,
            eigenvalues,
            eigenvectors,
        )
        step = step_result.step
        step_norm = step_result.step_norm
        history.append(
            OptimizationStep(
                iteration=iteration,
                coordinates=point.coordinates,
                energy=point.energy,
                gradient_norm=gradient_norm,
                eigenvalues=tuple(eigenvalues),
                step=tuple(step),
                step_norm=step_norm,
                classification=classification,
            )
        )

        if step_norm <= step_tolerance:
            return _result(
                converged=False,
                reason="step tolerance reached before first-order saddle convergence",
                point=point,
                eigenvalues=eigenvalues,
                classification=classification,
                history=history,
            )

        coordinates = tuple(value + delta for value, delta in zip(coordinates, step))

    final_point = evaluate_pes(
        energy,
        coordinates,
        gradient_step=finite_difference_step,
        hessian_step=finite_difference_step,
    )
    final_eigenvalues, _ = jacobi_eigendecomposition_symmetric(final_point.hessian)
    final_classification = classify_pes_point(
        final_point,
        gradient_tolerance=gradient_tolerance,
    )
    return _result(
        converged=False,
        reason="maximum iterations reached without first-order saddle convergence",
        point=final_point,
        eigenvalues=final_eigenvalues,
        classification=final_classification,
        history=history,
    )


def _eigenvector_following_step(
    gradient: Iterable[float],
    eigenvalues: list[float],
    eigenvectors: list[list[float]],
) -> list[float]:
    result = EVFOptimizer(
        EVFSettings(
            target_mode_index=0,
            trust_radius=1.0e300,
            eigenvalue_shift=EIGENVALUE_SHIFT,
            climb_on_target_mode=True,
        )
    ).compute_step(gradient, eigenvalues, eigenvectors)
    return list(result.step)


def _apply_trust_radius(step: list[float], trust_radius: float) -> list[float]:
    step_norm = norm(step)
    if step_norm <= trust_radius:
        return step
    scale = trust_radius / step_norm
    return [scale * value for value in step]


def _result(
    converged: bool,
    reason: str,
    point: PESPoint,
    eigenvalues: list[float],
    classification: StationaryPointType,
    history: list[OptimizationStep],
) -> OptimizationResult:
    return OptimizationResult(
        converged=converged,
        reason=reason,
        final_coordinates=point.coordinates,
        final_energy=point.energy,
        final_gradient=point.gradient,
        final_hessian=point.hessian,
        final_eigenvalues=tuple(eigenvalues),
        final_classification=classification,
        history=tuple(history),
    )


def _as_coordinate_tuple(coordinates: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in coordinates)
    if not values:
        raise ValueError("initial_coordinates must contain at least one value")
    if any(not isfinite(value) for value in values):
        raise ValueError("initial_coordinates must be finite numbers")
    return values


def _validate_optimizer_settings(
    max_iterations: int,
    gradient_tolerance: float,
    step_tolerance: float,
    trust_radius: float,
    finite_difference_step: float,
) -> None:
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    for name, value in (
        ("gradient_tolerance", gradient_tolerance),
        ("step_tolerance", step_tolerance),
        ("trust_radius", trust_radius),
        ("finite_difference_step", finite_difference_step),
    ):
        if value <= 0.0 or not isfinite(value):
            raise ValueError(f"{name} must be a positive finite number")
