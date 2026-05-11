from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.classification import StationaryPointType
from saddlepoint_whitebox.labels import (
    export_labels_csv_summary,
    export_labels_jsonl,
    generate_pes_label,
    label_to_dict,
)
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


class LabelTests(unittest.TestCase):
    def test_generate_pes_label_for_quadratic_saddle(self) -> None:
        label = generate_pes_label(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
            metadata={"source": "unit-test"},
        )

        self.assertEqual(label.classification, StationaryPointType.FIRST_ORDER_SADDLE.value)
        self.assertEqual(label.negative_eigenvalue_count, 1)
        self.assertIsNotNone(label.reaction_coordinate)
        self.assertEqual(label.metadata, {"source": "unit-test"})

        as_dict = label_to_dict(label)
        self.assertEqual(as_dict["classification"], StationaryPointType.FIRST_ORDER_SADDLE.value)
        self.assertIsInstance(as_dict["coordinates"], list)

    def test_export_jsonl_and_csv_summary(self) -> None:
        label = generate_pes_label(
            "quadratic saddle",
            quadratic_first_order_saddle,
            [0.0, 0.0],
        )

        with TemporaryDirectory() as temporary_directory:
            jsonl_path = Path(temporary_directory) / "labels.jsonl"
            csv_path = Path(temporary_directory) / "labels.csv"

            export_labels_jsonl([label], jsonl_path)
            export_labels_csv_summary([label], csv_path)

            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertGreater(jsonl_path.stat().st_size, 0)
            self.assertGreater(csv_path.stat().st_size, 0)

            first_json_line = jsonl_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                json.loads(first_json_line)["classification"],
                StationaryPointType.FIRST_ORDER_SADDLE.value,
            )


if __name__ == "__main__":
    unittest.main()
