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
)


class TopologyModelTests(unittest.TestCase):
    def test_default_parameters_name_contains_no_plus(self) -> None:
        parameters = default_no_plus_topology_parameters()

        self.assertIn("NO+", parameters.name)

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

    def test_candidate_diagnostics_include_expected_keys(self) -> None:
        diagnostics = evaluate_topology_stationary_candidates()

        self.assertIn("pi_complex_guess", diagnostics)
        self.assertIn("rim_complex_guess", diagnostics)
        self.assertIn("saddle_guess", diagnostics)

    def test_candidate_diagnostics_include_finite_curvature(self) -> None:
        diagnostics = evaluate_topology_stationary_candidates()
        eigenvalue_sets = [
            candidate["hessian_eigenvalues"]
            for candidate in diagnostics.values()
        ]

        self.assertTrue(
            any(
                any(abs(eigenvalue) > 1.0e-8 for eigenvalue in eigenvalues)
                for eigenvalues in eigenvalue_sets
            )
        )
        for eigenvalues in eigenvalue_sets:
            self.assertTrue(all(math.isfinite(eigenvalue) for eigenvalue in eigenvalues))

    def test_saddle_guess_has_first_order_saddle_like_curvature(self) -> None:
        diagnostics = evaluate_topology_stationary_candidates()

        self.assertEqual(diagnostics["saddle_guess"]["negative_eigenvalue_count"], 1)


if __name__ == "__main__":
    unittest.main()
