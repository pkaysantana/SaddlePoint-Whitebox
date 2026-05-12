"""Synthetic benzene-electrophile topology models.

These reduced-order surfaces are mathematical PES topology models inspired by
electrophilic aromatic substitution. They are not ab initio quantum chemistry.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, isfinite

from .calculus import evaluate_pes
from .classification import classify_pes_point
from .matrix import jacobi_eigendecomposition_symmetric, norm


@dataclass(frozen=True)
class ElectrophileTopologyParameters:
    """Parameters controlling a reduced benzene-electrophile topology model."""

    name: str
    pi_minimum_energy: float
    saddle_energy: float
    wheland_region_energy: float
    center_well_strength: float
    rim_well_strength: float
    barrier_strength: float
    coupling_strength: float
    ring_radius: float = 1.0


def default_no_plus_topology_parameters() -> ElectrophileTopologyParameters:
    """Return synthetic parameters for an NO+ inspired topology model."""

    return ElectrophileTopologyParameters(
        name="NO+ synthetic topology",
        pi_minimum_energy=-36.0,
        saddle_energy=23.0,
        wheland_region_energy=-12.0,
        center_well_strength=1.0,
        rim_well_strength=1.0,
        barrier_strength=1.0,
        coupling_strength=6.0,
        ring_radius=1.0,
    )


def benzene_electrophile_topology_energy(
    coordinates: Iterable[float],
    parameters: ElectrophileTopologyParameters | None = None,
) -> float:
    """Return a synthetic benzene-electrophile reduced-order PES energy.

    Coordinates are ``[rho, z]`` or ``[rho, z, q]``:

    - ``rho`` is lateral displacement from ring center toward the rim.
    - ``z`` is height above the aromatic plane.
    - ``q`` is an optional bond-formation / arenium-character coordinate.
    """

    params = _validated_parameters(parameters)
    rho, z_value, q_value = _parse_coordinates(coordinates)
    ring_radius = params.ring_radius

    # Over-center pi-complex attraction: electrophile above the aromatic center,
    # moderate height, and little arenium/bond-formation character.
    center_basin = _gaussian(
        ((rho - 0.0) / (0.30 * ring_radius)) ** 2
        + ((z_value - 1.35) / 0.32) ** 2
        + ((q_value - 0.0) / 0.45) ** 2
    )
    center_term = (
        params.pi_minimum_energy
        * params.center_well_strength
        * center_basin
    )

    # Over-rim / Wheland-like attraction: lateral displacement near the ring
    # rim, lower height, and larger bond-formation character.
    rim_basin = _gaussian(
        ((rho - ring_radius) / (0.28 * ring_radius)) ** 2
        + ((z_value - 0.55) / 0.25) ** 2
        + ((q_value - 1.0) / 0.35) ** 2
    )
    rim_term = (
        params.wheland_region_energy
        * params.rim_well_strength
        * rim_basin
    )

    # Barrier ridge between center and rim: a positive Gaussian feature located
    # between the two basins, representing a transition-state-like topology.
    barrier_basin = _gaussian(
        ((rho - (0.52 * ring_radius)) / (0.20 * ring_radius)) ** 2
        + ((z_value - 0.90) / 0.22) ** 2
        + ((q_value - 0.45) / 0.24) ** 2
    )
    barrier_term = (
        params.saddle_energy
        * params.barrier_strength
        * barrier_basin
    )

    # Coupling stabilizes simultaneous rim approach and bond formation. This is
    # the reduced-coordinate analogue of pi-complex character mixing with
    # sigma-complex / arenium character.
    height_gate = _gaussian(((z_value - 0.70) / 0.55) ** 2)
    coupling_term = (
        -abs(params.coupling_strength)
        * (rho / ring_radius)
        * q_value
        * height_gate
    )

    # Gentle confinement prevents the synthetic PES from drifting to unbounded
    # coordinates far outside the chemically meaningful reduced domain.
    radial_confinement = 1.8 * (rho / (1.6 * ring_radius)) ** 4
    height_confinement = 1.2 * ((z_value - 0.95) / 1.2) ** 4
    q_confinement = 0.8 * ((q_value - 0.5) / 1.2) ** 4

    return (
        center_term
        + rim_term
        + barrier_term
        + coupling_term
        + radial_confinement
        + height_confinement
        + q_confinement
    )


def benzene_electrophile_starting_points(
    parameters: ElectrophileTopologyParameters | None = None,
) -> dict[str, tuple[float, ...]]:
    """Return named starting guesses for the synthetic topology model."""

    params = _validated_parameters(parameters)
    radius = params.ring_radius
    return {
        "pi_complex_guess": (0.0, 1.35, 0.0),
        "rim_complex_guess": (0.95 * radius, 0.55, 1.0),
        "saddle_guess": (0.52 * radius, 0.90, 0.45),
    }


def evaluate_topology_stationary_candidates(
    parameters: ElectrophileTopologyParameters | None = None,
) -> dict[str, object]:
    """Evaluate topology diagnostics for named reduced-coordinate guesses."""

    params = _validated_parameters(parameters)
    results: dict[str, object] = {}
    for name, coordinates in benzene_electrophile_starting_points(params).items():
        point = evaluate_pes(
            lambda x: benzene_electrophile_topology_energy(x, params),
            coordinates,
        )
        eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(point.hessian)
        lowest_index = min(range(len(eigenvalues)), key=eigenvalues.__getitem__)
        negative_count = sum(1 for eigenvalue in eigenvalues if eigenvalue < -1.0e-8)
        results[name] = {
            "coordinates": list(point.coordinates),
            "energy": point.energy,
            "gradient": list(point.gradient),
            "gradient_norm": norm(point.gradient),
            "classification": classify_pes_point(point).value,
            "hessian_eigenvalues": list(eigenvalues),
            "negative_eigenvalue_count": negative_count,
            "reaction_coordinate_eigenvalue": eigenvalues[lowest_index],
            "reaction_coordinate": list(eigenvectors[lowest_index]),
        }
    return results


def _parse_coordinates(coordinates: Iterable[float]) -> tuple[float, float, float]:
    values = [float(value) for value in coordinates]
    if len(values) not in (2, 3):
        raise ValueError("coordinates must be [rho, z] or [rho, z, q]")
    if any(not isfinite(value) for value in values):
        raise ValueError("coordinates must be finite numbers")
    if len(values) == 2:
        return values[0], values[1], 0.0
    return values[0], values[1], values[2]


def _validated_parameters(
    parameters: ElectrophileTopologyParameters | None,
) -> ElectrophileTopologyParameters:
    params = parameters or default_no_plus_topology_parameters()
    numeric_values = (
        params.pi_minimum_energy,
        params.saddle_energy,
        params.wheland_region_energy,
        params.center_well_strength,
        params.rim_well_strength,
        params.barrier_strength,
        params.coupling_strength,
        params.ring_radius,
    )
    if any(not isfinite(value) for value in numeric_values):
        raise ValueError("topology parameters must be finite numbers")
    if params.ring_radius <= 0.0:
        raise ValueError("ring_radius must be positive")
    if params.center_well_strength <= 0.0:
        raise ValueError("center_well_strength must be positive")
    if params.rim_well_strength <= 0.0:
        raise ValueError("rim_well_strength must be positive")
    if params.barrier_strength <= 0.0:
        raise ValueError("barrier_strength must be positive")
    return params


def _gaussian(squared_scaled_distance: float) -> float:
    return exp(-squared_scaled_distance)
