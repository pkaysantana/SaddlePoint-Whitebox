from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.classification import StationaryPointType
from saddlepoint_whitebox.matrix import norm
from saddlepoint_whitebox.optimizers import eigenvector_following_saddle_search
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


class OptimizerTests(unittest.TestCase):
    def test_eigenvector_following_converges_to_quadratic_saddle(self) -> None:
        result = eigenvector_following_saddle_search(
            quadratic_first_order_saddle,
            initial_coordinates=[0.05, -0.04],
            max_iterations=25,
            trust_radius=0.1,
        )

        self.assertTrue(result.converged, result.reason)
        self.assertAlmostEqual(result.final_coordinates[0], 0.0, places=5)
        self.assertAlmostEqual(result.final_coordinates[1], 0.0, places=5)
        self.assertEqual(
            result.final_classification,
            StationaryPointType.FIRST_ORDER_SADDLE,
        )
        self.assertLess(norm(result.final_gradient), 1.0e-6)
        self.assertGreaterEqual(len(result.history), 1)


if __name__ == "__main__":
    unittest.main()
