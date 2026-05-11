from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.benchmarks import (
    benchmark_optimizer_run,
    benchmark_pes_evaluation,
)
from saddlepoint_whitebox.classification import StationaryPointType
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


class BenchmarkTests(unittest.TestCase):
    def test_pes_benchmark_counts_calls_and_classifies_saddle(self) -> None:
        result = benchmark_pes_evaluation(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
        )

        self.assertEqual(result.dimension, 2)
        self.assertGreater(result.energy_calls, 0)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertEqual(result.classification, StationaryPointType.FIRST_ORDER_SADDLE)
        self.assertEqual(result.negative_eigenvalues, 1)

    def test_optimizer_benchmark_summarizes_run(self) -> None:
        result = benchmark_optimizer_run(
            "quadratic saddle optimizer",
            quadratic_first_order_saddle,
            [0.05, -0.04],
            max_iterations=25,
        )

        self.assertEqual(result.dimension, 2)
        self.assertTrue(result.converged)
        self.assertGreater(result.energy_calls, 0)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertEqual(
            result.final_classification,
            StationaryPointType.FIRST_ORDER_SADDLE,
        )


if __name__ == "__main__":
    unittest.main()
