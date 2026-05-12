from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.classification import StationaryPointType
from saddlepoint_whitebox.evf import (
    EVFOptimizer,
    EVFSettings,
    lowest_mode,
    reaction_coordinate_from_hessian,
)
from saddlepoint_whitebox.matrix import norm
from saddlepoint_whitebox.optimizers import eigenvector_following_saddle_search
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


class EVFTests(unittest.TestCase):
    def test_compute_step_descends_stable_mode(self) -> None:
        result = EVFOptimizer(EVFSettings(trust_radius=1.0)).compute_step(
            gradient=[0.0, 1.0],
            eigenvalues=[-2.0, 4.0],
            eigenvectors=[[1.0, 0.0], [0.0, 1.0]],
        )

        self.assertAlmostEqual(result.step[0], 0.0)
        self.assertAlmostEqual(result.step[1], -0.25)
        self.assertAlmostEqual(result.mode_gradients[0], 0.0)
        self.assertAlmostEqual(result.mode_gradients[1], 1.0)
        self.assertAlmostEqual(result.mode_step_components[0], 0.0)
        self.assertAlmostEqual(result.mode_step_components[1], -0.25)
        self.assertEqual(result.target_mode_index, 0)
        self.assertAlmostEqual(result.target_eigenvalue, -2.0)
        self.assertGreaterEqual(result.step_norm, 0.0)
        self.assertFalse(result.scaled)

    def test_compute_step_climbs_target_mode(self) -> None:
        result = EVFOptimizer(EVFSettings(trust_radius=1.0)).compute_step(
            gradient=[1.0, 0.0],
            eigenvalues=[-2.0, 4.0],
            eigenvectors=[[1.0, 0.0], [0.0, 1.0]],
        )

        self.assertGreater(result.step[0], 0.0)
        self.assertAlmostEqual(result.step[1], 0.0)

    def test_trust_radius_caps_step_norm(self) -> None:
        result = EVFOptimizer(EVFSettings(trust_radius=0.1)).compute_step(
            gradient=[0.0, 10.0],
            eigenvalues=[-2.0, 4.0],
            eigenvectors=[[1.0, 0.0], [0.0, 1.0]],
        )

        self.assertAlmostEqual(result.step_norm, 0.1)
        self.assertLessEqual(result.step_norm, 0.1 + 1.0e-12)
        self.assertTrue(result.scaled)

    def test_max_step_component_caps_mode_step(self) -> None:
        result = EVFOptimizer(
            EVFSettings(trust_radius=1.0, max_step_component=0.05)
        ).compute_step(
            gradient=[1.0, 0.0],
            eigenvalues=[-2.0, 4.0],
            eigenvectors=[[1.0, 0.0], [0.0, 1.0]],
        )

        self.assertAlmostEqual(result.step[0], 0.05)
        self.assertTrue(result.scaled)

    def test_dimension_mismatch_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            EVFOptimizer().compute_step(
                gradient=[1.0, 2.0],
                eigenvalues=[-2.0],
                eigenvectors=[[1.0, 0.0]],
            )

    def test_invalid_target_mode_index_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            EVFOptimizer(EVFSettings(target_mode_index=2)).compute_step(
                gradient=[1.0, 2.0],
                eigenvalues=[-2.0, 4.0],
                eigenvectors=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_lowest_mode_chooses_smallest_eigenvalue(self) -> None:
        eigenvalue, eigenvector = lowest_mode(
            eigenvalues=[4.0, -2.0],
            eigenvectors=[[0.0, 1.0], [1.0, 0.0]],
        )

        self.assertAlmostEqual(eigenvalue, -2.0)
        self.assertEqual(eigenvector, (1.0, 0.0))

    def test_reaction_coordinate_from_hessian_returns_lowest_mode(self) -> None:
        eigenvalue, eigenvector = reaction_coordinate_from_hessian(
            [[-2.0, 0.0], [0.0, 4.0]]
        )

        self.assertAlmostEqual(eigenvalue, -2.0)
        self.assertAlmostEqual(abs(eigenvector[0]), 1.0)
        self.assertAlmostEqual(abs(eigenvector[1]), 0.0)

    def test_existing_saddle_search_still_converges(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
