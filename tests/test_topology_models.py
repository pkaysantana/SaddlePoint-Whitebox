from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.topology_models import (
    benzene_electrophile_starting_points,
    benzene_electrophile_topology_energy,
    default_no_plus_topology_parameters,
    evaluate_topology_stationary_candidates,
    optimize_topology_saddle_candidate,
)


class TopologyModelTests(unittest.TestCase):
    def test_default_parameters(self) -> None:
        parameters = default_no_plus_topology_parameters()

        self.assertIn("NO+", parameters.name)
        self.assertGreater(parameters.ring_radius, 0.0)

    def test_energy_is_finite_for_starting_guesses(self) -> None:
        for coordinates in benzene_electrophile_starting_points().values():
            energy = benzene_electrophile_topology_energy(coordinates)

            self.assertIsInstance(energy, float)
            self.assertTrue(math.isfinite(energy))

    def test_energy_accepts_two_dimensional_coordinates(self) -> None:
        energy = benzene_electrophile_topology_energy([0.0, 1.35])

        self.assertTrue(math.isfinite(energy))

    def test_invalid_coordinate_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            benzene_electrophile_topology_energy([0.0])

        with self.assertRaises(ValueError):
            benzene_electrophile_topology_energy([0.0, 1.0, 0.0, 1.0])

    def test_starting_points_include_three_dimensional_guesses(self) -> None:
        starting_points = benzene_electrophile_starting_points()

        self.assertIn("pi_complex_guess", starting_points)
        self.assertIn("rim_complex_guess", starting_points)
        self.assertIn("saddle_guess", starting_points)
        for coordinates in starting_points.values():
            self.assertEqual(len(coordinates), 3)

    def test_candidate_diagnostics_include_expected_keys(self) -> None:
        diagnostics = evaluate_topology_stationary_candidates()

        self.assertIn("pi_complex_guess", diagnostics)
        self.assertIn("rim_complex_guess", diagnostics)
        self.assertIn("saddle_guess", diagnostics)
        for name, candidate in diagnostics.items():
            self.assertEqual(candidate["name"], name)
            self.assertTrue(math.isfinite(candidate["energy"]))
            self.assertIn("classification", candidate)
            self.assertIn("hessian_eigenvalues", candidate)
            self.assertIn("gradient_norm", candidate)
            self.assertIn("negative_eigenvalue_count", candidate)
            self.assertIn("lowest_eigenvalue", candidate)
            self.assertIn("reaction_coordinate", candidate)
            self.assertTrue(math.isfinite(candidate["gradient_norm"]))
            self.assertTrue(
                all(
                    math.isfinite(eigenvalue)
                    for eigenvalue in candidate["hessian_eigenvalues"]
                )
            )

    def test_optimizer_helper_smoke_test(self) -> None:
        result = optimize_topology_saddle_candidate(max_iterations=3)

        self.assertTrue(hasattr(result, "final_coordinates"))
        self.assertTrue(hasattr(result, "final_energy"))
        self.assertTrue(hasattr(result, "final_classification"))
        self.assertTrue(hasattr(result, "history"))


if __name__ == "__main__":
    unittest.main()
