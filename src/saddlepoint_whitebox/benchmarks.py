"""Small benchmarks for finite-difference PES and optimizer work."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

from .calculus import evaluate_pes
from .classification import StationaryPointType, classify_pes_point
from .instrumentation import EnergyCallCounter
from .matrix import negative_eigenvalue_count, norm
from .optimizers import eigenvector_following_saddle_search

ScalarFunction = Callable[[list[float]], float]


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary for one finite-difference PES evaluation."""

    name: str
    dimension: int
    energy_calls: int
    elapsed_seconds: float
    gradient_norm: float
    classification: StationaryPointType
    negative_eigenvalues: int


@dataclass(frozen=True)
class OptimizerBenchmarkResult:
    """Summary for one optimizer run."""

    name: str
    dimension: int
    converged: bool
    energy_calls: int
    elapsed_seconds: float
    iterations: int
    final_gradient_norm: float
    final_classification: StationaryPointType


def benchmark_pes_evaluation(
    name: str,
    energy: ScalarFunction,
    coordinates: Iterable[float],
    finite_difference_step: float = 1.0e-5,
) -> BenchmarkResult:
    """Measure energy-call cost for evaluating energy, gradient, and Hessian."""

    coordinate_values = tuple(float(value) for value in coordinates)
    counted_energy = EnergyCallCounter(energy)
    start = perf_counter()
    point = evaluate_pes(
        counted_energy,
        coordinate_values,
        gradient_step=finite_difference_step,
        hessian_step=finite_difference_step,
    )
    elapsed_seconds = perf_counter() - start
    return BenchmarkResult(
        name=name,
        dimension=len(coordinate_values),
        energy_calls=counted_energy.calls,
        elapsed_seconds=elapsed_seconds,
        gradient_norm=norm(point.gradient),
        classification=classify_pes_point(point),
        negative_eigenvalues=negative_eigenvalue_count(point.hessian),
    )


def benchmark_optimizer_run(
    name: str,
    energy: ScalarFunction,
    initial_coordinates: Iterable[float],
    max_iterations: int = 50,
    trust_radius: float = 0.1,
    finite_difference_step: float = 1.0e-5,
) -> OptimizerBenchmarkResult:
    """Measure energy-call cost for one eigenvector-following optimizer run."""

    coordinate_values = tuple(float(value) for value in initial_coordinates)
    counted_energy = EnergyCallCounter(energy)
    start = perf_counter()
    result = eigenvector_following_saddle_search(
        counted_energy,
        coordinate_values,
        max_iterations=max_iterations,
        trust_radius=trust_radius,
        finite_difference_step=finite_difference_step,
    )
    elapsed_seconds = perf_counter() - start
    return OptimizerBenchmarkResult(
        name=name,
        dimension=len(coordinate_values),
        converged=result.converged,
        energy_calls=counted_energy.calls,
        elapsed_seconds=elapsed_seconds,
        iterations=len(result.history),
        final_gradient_norm=norm(result.final_gradient),
        final_classification=result.final_classification,
    )
