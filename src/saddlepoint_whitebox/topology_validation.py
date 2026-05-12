"""Validation helpers for the synthetic benzene-electrophile topology model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from .calculus import evaluate_pes
from .classification import StationaryPointType, classify_pes_point
from .matrix import jacobi_eigendecomposition_symmetric, norm
from .minimizers import gradient_descent_minimize
from .optimizers import eigenvector_following_saddle_search
from .topology_models import (
    ElectrophileTopologyParameters,
    benzene_electrophile_starting_points,
    benzene_electrophile_topology_energy,
    default_no_plus_topology_parameters,
)

EIGENVALUE_TOLERANCE = 1.0e-8
VALIDATION_GRADIENT_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class TopologyCandidateValidation:
    """Validation result for one synthetic topology candidate."""

    name: str
    initial_coordinates: tuple[float, ...]
    final_coordinates: tuple[float, ...]
    initial_energy: float
    final_energy: float
    gradient_norm: float
    classification: str
    hessian_eigenvalues: tuple[float, ...]
    negative_eigenvalue_count: int
    reaction_coordinate: tuple[float, ...] | None
    converged: bool
    method: str
    warning: str | None


@dataclass(frozen=True)
class TopologyValidationReport:
    """Summary report for synthetic topology validation."""

    model_name: str
    candidates: tuple[TopologyCandidateValidation, ...]
    passed: bool
    summary: str


def validate_benzene_electrophile_topology(
    parameters: ElectrophileTopologyParameters | None = None,
    finite_difference_step: float = 1.0e-5,
) -> TopologyValidationReport:
    """Validate whether the synthetic topology has the intended local shapes."""

    if finite_difference_step <= 0.0 or not isfinite(finite_difference_step):
        raise ValueError("finite_difference_step must be a positive finite number")

    params = parameters or default_no_plus_topology_parameters()
    energy = lambda coordinates: benzene_electrophile_topology_energy(coordinates, params)
    starting_points = benzene_electrophile_starting_points(params)
    candidates: list[TopologyCandidateValidation] = []

    for name in ("pi_complex_guess", "rim_complex_guess"):
        initial_coordinates = starting_points[name]
        result = gradient_descent_minimize(
            energy,
            initial_coordinates,
            max_iterations=1000,
            gradient_tolerance=VALIDATION_GRADIENT_TOLERANCE,
            step_tolerance=1.0e-8,
            learning_rate=0.001,
            finite_difference_step=finite_difference_step,
            max_step_norm=0.02,
        )
        candidates.append(
            _build_candidate_validation(
                name=name,
                initial_coordinates=initial_coordinates,
                initial_energy=float(energy(initial_coordinates)),
                final_coordinates=result.final_coordinates,
                energy=energy,
                finite_difference_step=finite_difference_step,
                gradient_tolerance=VALIDATION_GRADIENT_TOLERANCE,
                converged=result.converged,
                method="gradient_descent_minimize",
                optimizer_reason=result.reason,
            )
        )

    saddle_initial_coordinates = starting_points["saddle_guess"]
    saddle_result = eigenvector_following_saddle_search(
        energy,
        saddle_initial_coordinates,
        max_iterations=150,
        gradient_tolerance=VALIDATION_GRADIENT_TOLERANCE,
        step_tolerance=1.0e-8,
        trust_radius=0.08,
        finite_difference_step=finite_difference_step,
    )
    candidates.append(
        _build_candidate_validation(
            name="saddle_guess",
            initial_coordinates=saddle_initial_coordinates,
            initial_energy=float(energy(saddle_initial_coordinates)),
            final_coordinates=saddle_result.final_coordinates,
            energy=energy,
            finite_difference_step=finite_difference_step,
            gradient_tolerance=VALIDATION_GRADIENT_TOLERANCE,
            converged=saddle_result.converged,
            method="eigenvector_following_saddle_search",
            optimizer_reason=saddle_result.reason,
        )
    )

    passed = _report_passed(candidates)
    summary = _report_summary(candidates, passed)
    return TopologyValidationReport(
        model_name=params.name,
        candidates=tuple(candidates),
        passed=passed,
        summary=summary,
    )


def topology_validation_report_to_dict(
    report: TopologyValidationReport,
) -> dict[str, object]:
    """Convert a topology validation report to JSON-friendly containers."""

    return {
        "model_name": report.model_name,
        "candidates": [
            {
                "name": candidate.name,
                "initial_coordinates": list(candidate.initial_coordinates),
                "final_coordinates": list(candidate.final_coordinates),
                "initial_energy": candidate.initial_energy,
                "final_energy": candidate.final_energy,
                "gradient_norm": candidate.gradient_norm,
                "classification": candidate.classification,
                "hessian_eigenvalues": list(candidate.hessian_eigenvalues),
                "negative_eigenvalue_count": candidate.negative_eigenvalue_count,
                "reaction_coordinate": (
                    list(candidate.reaction_coordinate)
                    if candidate.reaction_coordinate is not None
                    else None
                ),
                "converged": candidate.converged,
                "method": candidate.method,
                "warning": candidate.warning,
            }
            for candidate in report.candidates
        ],
        "passed": report.passed,
        "summary": report.summary,
    }


def scan_topology_parameter_grid(
    parameter_sets: Iterable[ElectrophileTopologyParameters],
) -> list[TopologyValidationReport]:
    """Validate each synthetic topology parameter set in sequence."""

    return [
        validate_benzene_electrophile_topology(parameters)
        for parameters in parameter_sets
    ]


def rank_topology_reports(
    reports: Iterable[TopologyValidationReport],
) -> list[TopologyValidationReport]:
    """Return reports ordered by transparent topology-validation quality."""

    indexed_reports = list(enumerate(reports))
    indexed_reports.sort(
        key=lambda item: (
            not item[1].passed,
            not _saddle_has_one_negative_mode(item[1]),
            _warning_count(item[1]),
            item[0],
        )
    )
    return [report for _, report in indexed_reports]


def _build_candidate_validation(
    name: str,
    initial_coordinates: tuple[float, ...],
    initial_energy: float,
    final_coordinates: tuple[float, ...],
    energy,
    finite_difference_step: float,
    gradient_tolerance: float,
    converged: bool,
    method: str,
    optimizer_reason: str,
) -> TopologyCandidateValidation:
    point = evaluate_pes(
        energy,
        final_coordinates,
        gradient_step=finite_difference_step,
        hessian_step=finite_difference_step,
    )
    eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(point.hessian)
    negative_count = sum(
        1 for eigenvalue in eigenvalues if eigenvalue < -EIGENVALUE_TOLERANCE
    )
    reaction_coordinate = tuple(eigenvectors[0]) if negative_count > 0 else None
    classification = classify_pes_point(
        point,
        gradient_tolerance=gradient_tolerance,
    ).value
    warning = _candidate_warning(
        name=name,
        classification=classification,
        negative_eigenvalue_count=negative_count,
        converged=converged,
        optimizer_reason=optimizer_reason,
    )

    return TopologyCandidateValidation(
        name=name,
        initial_coordinates=tuple(initial_coordinates),
        final_coordinates=point.coordinates,
        initial_energy=initial_energy,
        final_energy=point.energy,
        gradient_norm=norm(point.gradient),
        classification=classification,
        hessian_eigenvalues=tuple(eigenvalues),
        negative_eigenvalue_count=negative_count,
        reaction_coordinate=reaction_coordinate,
        converged=converged,
        method=method,
        warning=warning,
    )


def _candidate_warning(
    name: str,
    classification: str,
    negative_eigenvalue_count: int,
    converged: bool,
    optimizer_reason: str,
) -> str | None:
    warnings: list[str] = []
    if not converged:
        warnings.append(f"optimization did not converge: {optimizer_reason}")

    if name == "pi_complex_guess":
        if classification != StationaryPointType.MINIMUM.value:
            warnings.append("expected pi complex to be minimum-like")
    elif name == "rim_complex_guess":
        if (
            classification == StationaryPointType.HIGHER_ORDER_SADDLE.value
            or negative_eigenvalue_count > 1
        ):
            warnings.append(
                "expected rim candidate to be minimum-like or not a higher-order saddle"
            )
    elif name == "saddle_guess":
        if classification != StationaryPointType.FIRST_ORDER_SADDLE.value:
            warnings.append("expected saddle guess to optimize to a first-order saddle")

    return "; ".join(warnings) if warnings else None


def _report_passed(candidates: list[TopologyCandidateValidation]) -> bool:
    by_name = {candidate.name: candidate for candidate in candidates}
    pi_candidate = by_name["pi_complex_guess"]
    rim_candidate = by_name["rim_complex_guess"]
    saddle_candidate = by_name["saddle_guess"]

    pi_passed = pi_candidate.classification == StationaryPointType.MINIMUM.value
    rim_passed = (
        rim_candidate.classification != StationaryPointType.HIGHER_ORDER_SADDLE.value
        and rim_candidate.negative_eigenvalue_count <= 1
    )
    saddle_passed = (
        saddle_candidate.classification
        == StationaryPointType.FIRST_ORDER_SADDLE.value
    )
    return pi_passed and rim_passed and saddle_passed


def _report_summary(
    candidates: list[TopologyCandidateValidation],
    passed: bool,
) -> str:
    if passed:
        return (
            "Synthetic topology validation passed: pi and rim candidates are "
            "minimum/intermediate-like and the saddle candidate is first-order."
        )

    failing_names = [
        candidate.name
        for candidate in candidates
        if candidate.warning is not None
    ]
    if not failing_names:
        return "Synthetic topology validation did not pass under the explicit rules."
    return (
        "Synthetic topology validation did not pass for: "
        + ", ".join(failing_names)
    )


def _saddle_has_one_negative_mode(report: TopologyValidationReport) -> bool:
    for candidate in report.candidates:
        if candidate.name == "saddle_guess":
            return candidate.negative_eigenvalue_count == 1
    return False


def _warning_count(report: TopologyValidationReport) -> int:
    count = 0
    for candidate in report.candidates:
        if candidate.warning:
            count += len(candidate.warning.split("; "))
    return count
