"""Small analytic potential-energy surfaces for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp


def quadratic_minimum(x: Sequence[float]) -> float:
    """Convex quadratic with a minimum at ``[0, 0]``."""

    return (x[0] ** 2) + (x[1] ** 2)


def quadratic_first_order_saddle(x: Sequence[float]) -> float:
    """Two-dimensional quadratic saddle with one uphill and one downhill mode."""

    return (x[0] ** 2) - (x[1] ** 2)


def coupled_quadratic_saddle(x: Sequence[float]) -> float:
    """Rotated quadratic saddle with a non-diagonal Hessian."""

    return (2.0 * x[0] ** 2) + (3.0 * x[0] * x[1]) - (x[1] ** 2)


def muller_brown(x: Sequence[float]) -> float:
    """Return the two-dimensional Muller-Brown benchmark potential.

    The Muller-Brown surface is a mathematical benchmark for testing minimum
    and saddle-search algorithms. It is useful for PES navigation work, but it
    is not an actual molecular potential.
    """

    x_value = x[0]
    y_value = x[1]
    amplitudes = (-200.0, -100.0, -170.0, 15.0)
    a_values = (-1.0, -1.0, -6.5, 0.7)
    b_values = (0.0, 0.0, 11.0, 0.6)
    c_values = (-10.0, -10.0, -6.5, 0.7)
    x_centers = (1.0, 0.0, -0.5, -1.0)
    y_centers = (0.0, 0.5, 1.5, 1.0)

    energy = 0.0
    for amplitude, a_value, b_value, c_value, x_center, y_center in zip(
        amplitudes,
        a_values,
        b_values,
        c_values,
        x_centers,
        y_centers,
    ):
        dx = x_value - x_center
        dy = y_value - y_center
        exponent = (a_value * dx * dx) + (b_value * dx * dy) + (c_value * dy * dy)
        energy += amplitude * exp(exponent)
    return energy
