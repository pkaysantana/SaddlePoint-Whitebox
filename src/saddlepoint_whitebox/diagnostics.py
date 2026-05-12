"""Diagnostics for saddle candidates and reaction-coordinate scans."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite

from .calculus import evaluate_pes
from .classification import StationaryPointType, classify_pes_point
from .matrix import jacobi_eigendecomposition_symmetric, norm, normalize_vector

ScalarFunction = Callable[[list[float]], float]

GRADIENT_WARNING_TOLERANCE = 1.0e-6
EIGENVALUE_ZERO_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class SaddleDiagnostic:
    """Export-friendly summary of a candidate saddle point."""

    name: str
    coordinates: tuple[float, ...]
    energy: float
    gradient_norm: float
    eigenvalues: tuple[float, ...]
    negative_eigenvalue_count: int
    classification: str
    reaction_coordinate: tuple[float, ...] | None
    is_first_order_saddle: bool
    warning: str | None


def diagnose_saddle_candidate(
    name: str,
    energy: ScalarFunction,
    coordinates: Iterable[float],
    finite_difference_step: float = 1.0e-5,
) -> SaddleDiagnostic:
    """Evaluate saddle diagnostics at one coordinate vector."""

    point = evaluate_pes(
        energy,
        coordinates,
        gradient_step=finite_difference_step,
        hessian_step=finite_difference_step,
    )
    eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(point.hessian)
    negative_count = sum(
        1 for eigenvalue in eigenvalues if eigenvalue < -EIGENVALUE_ZERO_TOLERANCE
    )
    reaction_coordinate = tuple(eigenvectors[0]) if eigenvalues[0] < 0.0 else None
    classification = classify_pes_point(point)
    warnings = _diagnostic_warnings(
        gradient_norm=norm(point.gradient),
        eigenvalues=eigenvalues,
        negative_eigenvalue_count=negative_count,
        classification=classification,
    )

    return SaddleDiagnostic(
        name=name,
        coordinates=point.coordinates,
        energy=point.energy,
        gradient_norm=norm(point.gradient),
        eigenvalues=tuple(eigenvalues),
        negative_eigenvalue_count=negative_count,
        classification=classification.value,
        reaction_coordinate=reaction_coordinate,
        is_first_order_saddle=classification == StationaryPointType.FIRST_ORDER_SADDLE,
        warning="; ".join(warnings) if warnings else None,
    )


def diagnostic_to_dict(diagnostic: SaddleDiagnostic) -> dict[str, object]:
    """Convert a saddle diagnostic to JSON-serializable containers."""

    return {
        "name": diagnostic.name,
        "coordinates": list(diagnostic.coordinates),
        "energy": diagnostic.energy,
        "gradient_norm": diagnostic.gradient_norm,
        "eigenvalues": list(diagnostic.eigenvalues),
        "negative_eigenvalue_count": diagnostic.negative_eigenvalue_count,
        "classification": diagnostic.classification,
        "reaction_coordinate": (
            list(diagnostic.reaction_coordinate)
            if diagnostic.reaction_coordinate is not None
            else None
        ),
        "is_first_order_saddle": diagnostic.is_first_order_saddle,
        "warning": diagnostic.warning,
    }


def scan_along_reaction_coordinate(
    energy: ScalarFunction,
    coordinates: Iterable[float],
    reaction_coordinate: Iterable[float],
    step_size: float = 0.05,
    points_each_side: int = 10,
) -> list[dict[str, object]]:
    """Sample energy along a normalized reaction-coordinate vector."""

    coordinate_values = _as_finite_vector(coordinates, "coordinates")
    direction = normalize_vector(_as_finite_vector(reaction_coordinate, "reaction_coordinate"))
    if len(coordinate_values) != len(direction):
        raise ValueError("coordinates and reaction_coordinate must have the same length")
    if step_size <= 0.0 or not isfinite(step_size):
        raise ValueError("step_size must be a positive finite number")
    if not isinstance(points_each_side, int) or points_each_side < 0:
        raise ValueError("points_each_side must be a non-negative integer")

    scan: list[dict[str, object]] = []
    for index in range(-points_each_side, points_each_side + 1):
        displacement = index * step_size
        displaced = [
            coordinate + displacement * direction_component
            for coordinate, direction_component in zip(coordinate_values, direction)
        ]
        scan.append(
            {
                "displacement": displacement,
                "coordinates": displaced,
                "energy": float(energy(list(displaced))),
            }
        )
    return scan


def _diagnostic_warnings(
    gradient_norm: float,
    eigenvalues: list[float],
    negative_eigenvalue_count: int,
    classification: StationaryPointType,
) -> list[str]:
    warnings: list[str] = []
    if gradient_norm > GRADIENT_WARNING_TOLERANCE and negative_eigenvalue_count == 1:
        warnings.append(
            "gradient norm is not small although Hessian has one negative mode"
        )
    if any(abs(eigenvalue) <= EIGENVALUE_ZERO_TOLERANCE for eigenvalue in eigenvalues):
        warnings.append("one or more Hessian eigenvalues are near zero")
    if classification != StationaryPointType.FIRST_ORDER_SADDLE:
        warnings.append("candidate is not classified as a first-order saddle")
    return warnings


def _as_finite_vector(values: Iterable[float], name: str) -> list[float]:
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError(f"{name} must contain at least one value")
    if any(not isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector
