from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.calculus import gradient, hessian
from saddlepoint_whitebox.finite_difference import adaptive_step


class FiniteDifferenceTests(unittest.TestCase):
    def assertVectorAlmostEqual(
        self,
        actual: list[float],
        expected: list[float],
        places: int = 6,
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def assertMatrixAlmostEqual(
        self,
        actual: list[list[float]],
        expected: list[list[float]],
        places: int = 4,
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            self.assertVectorAlmostEqual(actual_row, expected_row, places=places)

    def test_adaptive_steps_are_positive(self) -> None:
        steps = adaptive_step([0.0, -0.5, 2.0])

        self.assertEqual(len(steps), 3)
        self.assertTrue(all(step > 0.0 for step in steps))

    def test_large_coordinates_scale_and_clamp_steps(self) -> None:
        steps = adaptive_step([1.0, 10_000.0], maximum_step=1.0e-3)

        self.assertAlmostEqual(steps[0], 1.0e-5)
        self.assertAlmostEqual(steps[1], 1.0e-3)

    def test_invalid_step_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_step([1.0], base_step=0.0)

        with self.assertRaises(ValueError):
            adaptive_step([1.0], minimum_step=1.0e-2, maximum_step=1.0e-3)

    def test_adaptive_gradient_matches_quadratic(self) -> None:
        def energy(x: list[float]) -> float:
            return x[0] ** 2 - x[1] ** 2

        self.assertVectorAlmostEqual(
            gradient(energy, [0.25, -0.5], step="adaptive"),
            [0.5, 1.0],
        )

    def test_adaptive_hessian_matches_quadratic_at_stationary_point(self) -> None:
        def energy(x: list[float]) -> float:
            return x[0] ** 2 - x[1] ** 2

        self.assertMatrixAlmostEqual(
            hessian(energy, [0.0, 0.0], step="adaptive"),
            [[2.0, 0.0], [0.0, -2.0]],
        )

    def test_unknown_step_string_raises(self) -> None:
        def energy(x: list[float]) -> float:
            return sum(x)

        with self.assertRaises(ValueError):
            gradient(energy, [1.0], step="scaled")


if __name__ == "__main__":
    unittest.main()
