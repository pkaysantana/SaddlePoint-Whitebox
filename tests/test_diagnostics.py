from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.classification import StationaryPointType
from saddlepoint_whitebox.diagnostics import (
    diagnose_saddle_candidate,
    diagnostic_to_dict,
    scan_along_reaction_coordinate,
)
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


class DiagnosticsTests(unittest.TestCase):
    def test_diagnose_quadratic_first_order_saddle(self) -> None:
        diagnostic = diagnose_saddle_candidate(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
        )

        self.assertTrue(diagnostic.is_first_order_saddle)
        self.assertEqual(diagnostic.classification, StationaryPointType.FIRST_ORDER_SADDLE.value)
        self.assertEqual(diagnostic.negative_eigenvalue_count, 1)
        self.assertIsNotNone(diagnostic.reaction_coordinate)
        self.assertIsNone(diagnostic.warning)

        as_dict = diagnostic_to_dict(diagnostic)
        self.assertEqual(as_dict["name"], "quadratic saddle")
        self.assertIsInstance(as_dict["eigenvalues"], list)

    def test_reaction_coordinate_scan_returns_expected_number_of_points(self) -> None:
        diagnostic = diagnose_saddle_candidate(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
        )
        self.assertIsNotNone(diagnostic.reaction_coordinate)

        scan = scan_along_reaction_coordinate(
            quadratic_first_order_saddle,
            diagnostic.coordinates,
            diagnostic.reaction_coordinate,
            step_size=0.05,
            points_each_side=3,
        )

        self.assertEqual(len(scan), 7)
        self.assertAlmostEqual(scan[0]["displacement"], -0.15)
        self.assertAlmostEqual(scan[-1]["displacement"], 0.15)

    def test_reaction_coordinate_scan_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            scan_along_reaction_coordinate(
                quadratic_first_order_saddle,
                [0.0, 0.0],
                [1.0, 0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
