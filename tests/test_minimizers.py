from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.classification import StationaryPointType
from saddlepoint_whitebox.matrix import norm
from saddlepoint_whitebox.minimizers import gradient_descent_minimize
from saddlepoint_whitebox.surfaces import quadratic_minimum


class MinimizerTests(unittest.TestCase):
    def test_gradient_descent_converges_to_quadratic_minimum(self) -> None:
        result = gradient_descent_minimize(
            quadratic_minimum,
            [0.5, -0.4],
            max_iterations=250,
        )

        self.assertTrue(result.converged, result.reason)
        self.assertAlmostEqual(result.final_coordinates[0], 0.0, places=5)
        self.assertAlmostEqual(result.final_coordinates[1], 0.0, places=5)
        self.assertEqual(result.final_classification, StationaryPointType.MINIMUM.value)
        self.assertLess(norm(result.final_gradient), 1.0e-6)
        self.assertGreaterEqual(len(result.history), 1)

    def test_gradient_descent_rejects_invalid_learning_rate(self) -> None:
        with self.assertRaises(ValueError):
            gradient_descent_minimize(
                quadratic_minimum,
                [0.5, -0.4],
                learning_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
