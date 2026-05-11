from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.calculus import evaluate_pes
from saddlepoint_whitebox.classification import (
    StationaryPointType,
    classify_hessian,
    classify_pes_point,
)
from saddlepoint_whitebox.surfaces import (
    quadratic_first_order_saddle,
    quadratic_minimum,
)


class ClassificationTests(unittest.TestCase):
    def test_quadratic_minimum_is_classified_as_minimum(self) -> None:
        point = evaluate_pes(quadratic_minimum, [0.0, 0.0])

        self.assertEqual(classify_pes_point(point), StationaryPointType.MINIMUM)

    def test_quadratic_saddle_is_classified_as_first_order_saddle(self) -> None:
        point = evaluate_pes(quadratic_first_order_saddle, [0.0, 0.0])

        self.assertEqual(
            classify_pes_point(point),
            StationaryPointType.FIRST_ORDER_SADDLE,
        )

    def test_negative_definite_quadratic_is_classified_as_maximum(self) -> None:
        def negative_definite(x: list[float]) -> float:
            return -(x[0] ** 2) - (x[1] ** 2)

        point = evaluate_pes(negative_definite, [0.0, 0.0])

        self.assertEqual(classify_pes_point(point), StationaryPointType.MAXIMUM)

    def test_nonstationary_point_is_indeterminate(self) -> None:
        point = evaluate_pes(quadratic_minimum, [1.0, 0.0])

        self.assertEqual(
            classify_pes_point(point),
            StationaryPointType.FLAT_OR_INDETERMINATE,
        )

    def test_flat_hessian_is_indeterminate(self) -> None:
        self.assertEqual(
            classify_hessian([[2.0, 0.0], [0.0, 0.0]]),
            StationaryPointType.FLAT_OR_INDETERMINATE,
        )


if __name__ == "__main__":
    unittest.main()
