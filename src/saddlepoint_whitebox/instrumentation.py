"""Instrumentation helpers for energy-function evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable

ScalarFunction = Callable[[list[float]], float]


class EnergyCallCounter:
    """Wrap an energy function and count scalar energy evaluations."""

    def __init__(self, energy: ScalarFunction) -> None:
        self._energy = energy
        self._calls = 0
        self._last_coordinates: tuple[float, ...] | None = None
        self._last_energy: float | None = None

    @property
    def calls(self) -> int:
        """Number of times the wrapped function has been called."""

        return self._calls

    @property
    def last_coordinates(self) -> tuple[float, ...] | None:
        """Most recent coordinates passed to the wrapped function."""

        return self._last_coordinates

    @property
    def last_energy(self) -> float | None:
        """Most recent scalar energy value returned by the wrapped function."""

        return self._last_energy

    def reset(self) -> None:
        """Reset call count and last successful evaluation."""

        self._calls = 0
        self._last_coordinates = None
        self._last_energy = None

    def __call__(self, coordinates: Iterable[float]) -> float:
        """Evaluate the wrapped energy function and record the call."""

        coordinate_values = [float(value) for value in coordinates]
        self._calls += 1
        energy_value = float(self._energy(list(coordinate_values)))
        self._last_coordinates = tuple(coordinate_values)
        self._last_energy = energy_value
        return energy_value
