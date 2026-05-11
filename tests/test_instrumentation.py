from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.instrumentation import EnergyCallCounter


class InstrumentationTests(unittest.TestCase):
    def test_energy_call_counter_counts_and_returns_energy(self) -> None:
        def energy(x: list[float]) -> float:
            return x[0] ** 2 + x[1] ** 2

        counter = EnergyCallCounter(energy)

        self.assertEqual(counter([3.0, 4.0]), 25.0)
        self.assertEqual(counter([1.0, 2.0]), 5.0)
        self.assertEqual(counter.calls, 2)
        self.assertEqual(counter.last_coordinates, (1.0, 2.0))
        self.assertEqual(counter.last_energy, 5.0)

    def test_reset_returns_counter_to_zero(self) -> None:
        counter = EnergyCallCounter(lambda x: sum(x))
        counter([1.0, 2.0])

        counter.reset()

        self.assertEqual(counter.calls, 0)
        self.assertIsNone(counter.last_coordinates)
        self.assertIsNone(counter.last_energy)


if __name__ == "__main__":
    unittest.main()
