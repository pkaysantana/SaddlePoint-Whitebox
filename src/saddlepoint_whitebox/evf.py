"""Explicit eigenvector-following step calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from .matrix import dot, jacobi_eigendecomposition_symmetric, norm


@dataclass(frozen=True)
class EVFSettings:
    """Configuration for a single eigenvector-following step."""

    target_mode_index: int = 0
    trust_radius: float = 0.1
    eigenvalue_shift: float = 1.0e-8
    max_step_component: float | None = None
    climb_on_target_mode: bool = True


@dataclass(frozen=True)
class EVFStepResult:
    """Detailed result from one eigenvector-following step calculation."""

    step: tuple[float, ...]
    step_norm: float
    mode_gradients: tuple[float, ...]
    mode_step_components: tuple[float, ...]
    scaled: bool
    target_mode_index: int
    target_eigenvalue: float


class EVFOptimizer:
    """Compute whitebox eigenvector-following steps from Hessian modes."""

    def __init__(self, settings: EVFSettings | None = None) -> None:
        self.settings = settings or EVFSettings()
        _validate_settings(self.settings)

    def compute_step(
        self,
        gradient: Iterable[float],
        eigenvalues: Iterable[float],
        eigenvectors: Iterable[Iterable[float]],
    ) -> EVFStepResult:
        """Return one EVF step in Cartesian coordinates.

        The Hessian eigenvectors define local mode coordinates. The Cartesian
        gradient is projected into those modes. A Newton-like step is formed in
        each mode using ``abs(lambda_i) + shift`` so nearly flat modes cannot
        explode. The target mode is treated as the reaction coordinate: by
        default the step climbs that mode while all other modes minimize.
        """

        gradient_values, eigenvalue_values, eigenvector_values = _validate_inputs(
            gradient,
            eigenvalues,
            eigenvectors,
            target_mode_index=self.settings.target_mode_index,
        )

        mode_gradients = tuple(
            dot(gradient_values, eigenvector) for eigenvector in eigenvector_values
        )
        step_components: list[float] = []
        scaled = False

        for mode_index, (mode_gradient, eigenvalue) in enumerate(
            zip(mode_gradients, eigenvalue_values)
        ):
            denominator = abs(eigenvalue) + self.settings.eigenvalue_shift

            # A minimization Newton step moves against the projected gradient.
            # For transition-state searches, the reaction-coordinate mode is
            # inverted so the step climbs toward the index-1 saddle direction.
            if mode_index == self.settings.target_mode_index and self.settings.climb_on_target_mode:
                mode_step = mode_gradient / denominator
            else:
                mode_step = -mode_gradient / denominator

            if self.settings.max_step_component is not None:
                capped = _cap_component(mode_step, self.settings.max_step_component)
                scaled = scaled or capped != mode_step
                mode_step = capped
            step_components.append(mode_step)

        step = [0.0 for _ in gradient_values]
        for mode_step, eigenvector in zip(step_components, eigenvector_values):
            for coordinate_index, vector_component in enumerate(eigenvector):
                step[coordinate_index] += mode_step * vector_component

        step_norm = norm(step)
        if step_norm > self.settings.trust_radius:
            scale = self.settings.trust_radius / step_norm
            step = [scale * value for value in step]
            step_components = [scale * value for value in step_components]
            step_norm = norm(step)
            scaled = True

        return EVFStepResult(
            step=tuple(step),
            step_norm=step_norm,
            mode_gradients=mode_gradients,
            mode_step_components=tuple(step_components),
            scaled=scaled,
            target_mode_index=self.settings.target_mode_index,
            target_eigenvalue=eigenvalue_values[self.settings.target_mode_index],
        )


def lowest_mode(
    eigenvalues: Iterable[float],
    eigenvectors: Iterable[Iterable[float]],
) -> tuple[float, tuple[float, ...]]:
    """Return the lowest eigenvalue and its corresponding eigenvector."""

    eigenvalue_values = _as_finite_vector(eigenvalues, "eigenvalues")
    eigenvector_values = [
        _as_finite_vector(eigenvector, f"eigenvectors[{index}]")
        for index, eigenvector in enumerate(eigenvectors)
    ]
    if len(eigenvector_values) != len(eigenvalue_values):
        raise ValueError("number of eigenvectors must match number of eigenvalues")
    for index, eigenvector in enumerate(eigenvector_values):
        if len(eigenvector) != len(eigenvalue_values):
            raise ValueError(
                f"eigenvectors[{index}] length must match number of eigenvalues"
            )
        if norm(eigenvector) == 0.0:
            raise ValueError(f"eigenvectors[{index}] must be non-zero")

    lowest_index = min(range(len(eigenvalue_values)), key=eigenvalue_values.__getitem__)
    return eigenvalue_values[lowest_index], tuple(eigenvector_values[lowest_index])


def reaction_coordinate_from_hessian(
    hessian: Iterable[Iterable[float]],
) -> tuple[float, tuple[float, ...]]:
    """Return the lowest Hessian mode as a reaction-coordinate candidate."""

    eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(hessian)
    return lowest_mode(eigenvalues, eigenvectors)


def _validate_settings(settings: EVFSettings) -> None:
    if not isinstance(settings.target_mode_index, int) or settings.target_mode_index < 0:
        raise ValueError("target_mode_index must be a non-negative integer")
    if settings.trust_radius <= 0.0 or not isfinite(settings.trust_radius):
        raise ValueError("trust_radius must be a positive finite number")
    if settings.eigenvalue_shift <= 0.0 or not isfinite(settings.eigenvalue_shift):
        raise ValueError("eigenvalue_shift must be a positive finite number")
    if settings.max_step_component is not None and (
        settings.max_step_component <= 0.0 or not isfinite(settings.max_step_component)
    ):
        raise ValueError("max_step_component must be None or a positive finite number")
    if not isinstance(settings.climb_on_target_mode, bool):
        raise ValueError("climb_on_target_mode must be a bool")


def _validate_inputs(
    gradient: Iterable[float],
    eigenvalues: Iterable[float],
    eigenvectors: Iterable[Iterable[float]],
    target_mode_index: int,
) -> tuple[list[float], list[float], list[list[float]]]:
    gradient_values = _as_finite_vector(gradient, "gradient")
    eigenvalue_values = _as_finite_vector(eigenvalues, "eigenvalues")
    eigenvector_values = [
        _as_finite_vector(eigenvector, f"eigenvectors[{index}]")
        for index, eigenvector in enumerate(eigenvectors)
    ]

    dimension = len(gradient_values)
    if len(eigenvalue_values) != dimension:
        raise ValueError("eigenvalues length must match gradient dimension")
    if len(eigenvector_values) != dimension:
        raise ValueError("number of eigenvectors must match gradient dimension")
    if target_mode_index >= dimension:
        raise ValueError("target_mode_index is outside the available eigenmodes")

    for index, eigenvector in enumerate(eigenvector_values):
        if len(eigenvector) != dimension:
            raise ValueError(
                f"eigenvectors[{index}] length must match gradient dimension"
            )
        if norm(eigenvector) == 0.0:
            raise ValueError(f"eigenvectors[{index}] must be non-zero")

    return gradient_values, eigenvalue_values, eigenvector_values


def _as_finite_vector(values: Iterable[float], name: str) -> list[float]:
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError(f"{name} must contain at least one value")
    if any(not isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _cap_component(value: float, maximum: float) -> float:
    if value > maximum:
        return maximum
    if value < -maximum:
        return -maximum
    return value
