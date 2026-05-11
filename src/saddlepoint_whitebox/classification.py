"""Stationary point classification for potential energy surfaces."""

from __future__ import annotations

from enum import Enum

from .calculus import PESPoint
from .matrix import jacobi_eigenvalues_symmetric, norm


class StationaryPointType(Enum):
    """Qualitative Hessian classification at a stationary point."""

    MINIMUM = "minimum"
    FIRST_ORDER_SADDLE = "first_order_saddle"
    HIGHER_ORDER_SADDLE = "higher_order_saddle"
    MAXIMUM = "maximum"
    FLAT_OR_INDETERMINATE = "flat_or_indeterminate"


def classify_hessian(
    hessian: list[list[float]] | tuple[tuple[float, ...], ...],
    tolerance: float = 1.0e-8,
) -> StationaryPointType:
    """Classify a stationary point from Hessian eigenvalue signs.

    Eigenvalues close to zero are treated as indeterminate because flat modes
    can mean numerical noise, true symmetry/translation, or a higher-order
    stationary point that needs a more specialized analysis.
    """

    eigenvalues = jacobi_eigenvalues_symmetric(hessian)
    if any(abs(eigenvalue) <= tolerance for eigenvalue in eigenvalues):
        return StationaryPointType.FLAT_OR_INDETERMINATE

    negative_count = sum(1 for eigenvalue in eigenvalues if eigenvalue < -tolerance)
    positive_count = sum(1 for eigenvalue in eigenvalues if eigenvalue > tolerance)

    if positive_count == len(eigenvalues):
        return StationaryPointType.MINIMUM
    if negative_count == 1 and positive_count == len(eigenvalues) - 1:
        return StationaryPointType.FIRST_ORDER_SADDLE
    if negative_count > 1 and positive_count >= 1:
        return StationaryPointType.HIGHER_ORDER_SADDLE
    if negative_count == len(eigenvalues):
        return StationaryPointType.MAXIMUM
    return StationaryPointType.FLAT_OR_INDETERMINATE


def classify_pes_point(
    point: PESPoint,
    gradient_tolerance: float = 1.0e-6,
    eigenvalue_tolerance: float = 1.0e-8,
) -> StationaryPointType:
    """Classify a PES point only if its gradient is small enough."""

    if norm(point.gradient) > gradient_tolerance:
        return StationaryPointType.FLAT_OR_INDETERMINATE
    return classify_hessian(point.hessian, tolerance=eigenvalue_tolerance)
