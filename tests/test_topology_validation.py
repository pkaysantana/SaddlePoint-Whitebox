from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.topology_models import default_no_plus_topology_parameters
from saddlepoint_whitebox.topology_validation import (
    TopologyValidationReport,
    rank_topology_reports,
    scan_topology_parameter_grid,
    topology_validation_report_to_dict,
    validate_benzene_electrophile_topology,
)


class TopologyValidationTests(unittest.TestCase):
    def test_default_topology_validation_report_shape(self) -> None:
        report = validate_benzene_electrophile_topology()

        self.assertIsInstance(report, TopologyValidationReport)
        self.assertEqual(len(report.candidates), 3)
        self.assertEqual(
            {candidate.name for candidate in report.candidates},
            {"pi_complex_guess", "rim_complex_guess", "saddle_guess"},
        )

        for candidate in report.candidates:
            self.assertTrue(math.isfinite(candidate.final_energy))
            self.assertTrue(math.isfinite(candidate.gradient_norm))
            self.assertGreater(len(candidate.hessian_eigenvalues), 0)
            self.assertTrue(
                all(math.isfinite(value) for value in candidate.hessian_eigenvalues)
            )

    def test_topology_validation_report_to_dict_is_json_friendly(self) -> None:
        report = validate_benzene_electrophile_topology()
        data = topology_validation_report_to_dict(report)

        self.assertEqual(data["model_name"], report.model_name)
        self.assertIsInstance(data["candidates"], list)
        self.assertIsInstance(data["passed"], bool)
        self.assertIsInstance(data["summary"], str)
        first_candidate = data["candidates"][0]
        self.assertIsInstance(first_candidate["final_coordinates"], list)
        self.assertIsInstance(first_candidate["hessian_eigenvalues"], list)

    def test_scan_topology_parameter_grid_and_ranking(self) -> None:
        reports = scan_topology_parameter_grid([default_no_plus_topology_parameters()])
        ranked = rank_topology_reports(reports)

        self.assertEqual(len(reports), 1)
        self.assertIsInstance(reports[0], TopologyValidationReport)
        self.assertEqual(ranked, reports)


if __name__ == "__main__":
    unittest.main()
