from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.calculus import evaluate_pes, force, gradient, hessian


class CalculusTests(unittest.TestCase):
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
        places: int = 5,
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            self.assertVectorAlmostEqual(actual_row, expected_row, places=places)

    def test_gradient_matches_quadratic_analytic_value(self) -> None:
        def energy(x: list[float]) -> float:
            return x[0] ** 2 + (3.0 * x[0] * x[1]) - (2.0 * x[2]) + 5.0

        coords = [1.5, -2.0, 0.7]

        self.assertVectorAlmostEqual(
            gradient(energy, coords),
            [2.0 * coords[0] + 3.0 * coords[1], 3.0 * coords[0], -2.0],
        )

    def test_hessian_matches_quadratic_analytic_value(self) -> None:
        def energy(x: list[float]) -> float:
            return (
                (2.0 * x[0] ** 2)
                + (3.0 * x[0] * x[1])
                + (5.0 * x[1] ** 2)
                - (x[0] * x[2])
                + (4.0 * x[2] ** 2)
            )

        self.assertMatrixAlmostEqual(
            hessian(energy, [0.2, -0.3, 1.7]),
            [
                [4.0, 3.0, -1.0],
                [3.0, 10.0, 0.0],
                [-1.0, 0.0, 8.0],
            ],
        )

    def test_force_uses_negative_gradient_convention(self) -> None:
        def energy(x: list[float]) -> float:
            return x[0] ** 2 + x[1] ** 2

        self.assertVectorAlmostEqual(force(energy, [1.0, -2.0]), [-2.0, 4.0])

    def test_evaluate_pes_returns_complete_point_data(self) -> None:
        def energy(x: list[float]) -> float:
            return x[0] ** 2 - x[1] ** 2

        point = evaluate_pes(energy, [0.5, -1.5])

        self.assertEqual(point.coordinates, (0.5, -1.5))
        self.assertAlmostEqual(point.energy, -2.0)
        self.assertVectorAlmostEqual(list(point.gradient), [1.0, 3.0])
        self.assertVectorAlmostEqual(list(point.force), [-1.0, -3.0])
        self.assertMatrixAlmostEqual(
            [list(row) for row in point.hessian],
            [[2.0, 0.0], [0.0, -2.0]],
        )

    def test_rejects_invalid_step(self) -> None:
        def energy(x: list[float]) -> float:
            return sum(x)

        with self.assertRaises(ValueError):
            gradient(energy, [1.0], step=0.0)

        with self.assertRaises(ValueError):
            hessian(energy, [1.0, 2.0], step=[1.0e-4])


if __name__ == "__main__":
    unittest.main()
