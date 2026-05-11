from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.calculus import hessian
from saddlepoint_whitebox.matrix import jacobi_eigenvalues_symmetric
from saddlepoint_whitebox.surfaces import (
    coupled_quadratic_saddle,
    muller_brown,
    quadratic_first_order_saddle,
    quadratic_minimum,
)


class SurfaceTests(unittest.TestCase):
    def test_quadratic_minimum_hessian_is_positive_definite(self) -> None:
        eigenvalues = jacobi_eigenvalues_symmetric(hessian(quadratic_minimum, [0.0, 0.0]))

        self.assertTrue(all(eigenvalue > 0.0 for eigenvalue in eigenvalues))

    def test_quadratic_saddle_hessian_has_one_negative_mode(self) -> None:
        eigenvalues = jacobi_eigenvalues_symmetric(
            hessian(quadratic_first_order_saddle, [0.0, 0.0])
        )

        self.assertLess(eigenvalues[0], 0.0)
        self.assertGreater(eigenvalues[1], 0.0)

    def test_coupled_quadratic_saddle_hessian_has_mixed_signs(self) -> None:
        eigenvalues = jacobi_eigenvalues_symmetric(
            hessian(coupled_quadratic_saddle, [0.0, 0.0])
        )

        self.assertLess(eigenvalues[0], 0.0)
        self.assertGreater(eigenvalues[1], 0.0)

    def test_muller_brown_returns_finite_energy(self) -> None:
        self.assertTrue(math.isfinite(muller_brown([-0.558, 1.442])))


if __name__ == "__main__":
    unittest.main()
