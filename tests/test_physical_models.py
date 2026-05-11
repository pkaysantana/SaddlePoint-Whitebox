from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.calculus import evaluate_pes
from saddlepoint_whitebox.physical_models import lennard_jones_cluster_energy


class PhysicalModelTests(unittest.TestCase):
    def test_lennard_jones_two_particle_minimum_is_negative_epsilon(self) -> None:
        epsilon = 2.5
        sigma = 1.3
        minimum_distance = (2.0 ** (1.0 / 6.0)) * sigma

        energy = lennard_jones_cluster_energy(
            [0.0, 0.0, minimum_distance, 0.0],
            dimensions=2,
            epsilon=epsilon,
            sigma=sigma,
        )

        self.assertAlmostEqual(energy, -epsilon)

    def test_invalid_coordinate_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            lennard_jones_cluster_energy([0.0, 0.0, 1.0], dimensions=2)

    def test_zero_distance_raises(self) -> None:
        with self.assertRaises(ValueError):
            lennard_jones_cluster_energy([0.0, 0.0, 0.0, 0.0], dimensions=2)

    def test_lennard_jones_runs_through_pes_pipeline(self) -> None:
        point = evaluate_pes(
            lambda x: lennard_jones_cluster_energy(x, dimensions=2),
            [0.0, 0.0, 1.5, 0.0],
        )

        self.assertEqual(len(point.gradient), 4)
        self.assertEqual(len(point.hessian), 4)


if __name__ == "__main__":
    unittest.main()
