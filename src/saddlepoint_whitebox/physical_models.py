"""Toy physical energy models for domain-agnostic engine tests."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite, sqrt


def lennard_jones_cluster_energy(
    coordinates: Iterable[float],
    dimensions: int = 2,
    epsilon: float = 1.0,
    sigma: float = 1.0,
) -> float:
    """Return a toy Lennard-Jones pair energy for a flat coordinate vector.

    This is an empirical benchmark potential, not quantum chemistry. The engine
    only sees a scalar ``E(x)`` and therefore treats this exactly like any toy
    mathematical PES or future external energy wrapper.
    """

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if epsilon <= 0.0 or not isfinite(epsilon):
        raise ValueError("epsilon must be a positive finite number")
    if sigma <= 0.0 or not isfinite(sigma):
        raise ValueError("sigma must be a positive finite number")

    values = [float(value) for value in coordinates]
    if not values:
        raise ValueError("coordinates must contain at least two particles")
    if any(not isfinite(value) for value in values):
        raise ValueError("coordinates must be finite numbers")
    if len(values) % dimensions != 0:
        raise ValueError("coordinate length must be divisible by dimensions")

    particle_count = len(values) // dimensions
    if particle_count < 2:
        raise ValueError("coordinates must contain at least two particles")

    particles = [
        values[index * dimensions : (index + 1) * dimensions]
        for index in range(particle_count)
    ]
    minimum_distance = max(sigma * 1.0e-12, 1.0e-12)
    energy = 0.0

    for left_index in range(particle_count):
        for right_index in range(left_index + 1, particle_count):
            squared_distance = sum(
                (left - right) ** 2
                for left, right in zip(particles[left_index], particles[right_index])
            )
            distance = sqrt(squared_distance)
            if distance <= minimum_distance:
                raise ValueError("particle distance is zero or too close to zero")

            reduced_distance = sigma / distance
            reduced_distance_6 = reduced_distance**6
            reduced_distance_12 = reduced_distance_6 * reduced_distance_6
            energy += 4.0 * epsilon * (reduced_distance_12 - reduced_distance_6)

    return energy
