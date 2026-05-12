from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.dataset_generation import (
    PerturbationSettings,
    generate_perturbed_labels,
    generate_topology_dataset,
)
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


class DatasetGenerationTests(unittest.TestCase):
    def test_perturbed_labels_are_reproducible_with_seed(self) -> None:
        settings = PerturbationSettings(samples=5, radius=0.1, seed=42)

        first = generate_perturbed_labels(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
            settings=settings,
        )
        second = generate_perturbed_labels(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
            settings=settings,
        )

        self.assertEqual(len(first), 5)
        self.assertEqual([label.coordinates for label in first], [label.coordinates for label in second])
        self.assertEqual(first[0].metadata["sample_index"], "0")
        self.assertEqual(first[0].metadata["model_name"], "quadratic saddle")
        self.assertIn("source_center", first[0].metadata)

    def test_perturbed_labels_reject_non_positive_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            generate_perturbed_labels(
                "quadratic saddle",
                quadratic_first_order_saddle,
                [0.0, 0.0],
                settings=PerturbationSettings(samples=0),
            )

    def test_generate_topology_dataset_exports_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            jsonl_path = Path(temporary_directory) / "topology_labels.jsonl"
            csv_path = Path(temporary_directory) / "topology_labels_summary.csv"

            labels = generate_topology_dataset(
                jsonl_path,
                csv_path,
                samples=9,
                seed=7,
            )

            self.assertEqual(len(labels), 9)
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertGreater(jsonl_path.stat().st_size, 0)
            self.assertGreater(csv_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
