from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.matrix import (
    add_vectors,
    determinant,
    dot,
    identity,
    is_first_order_saddle,
    is_symmetric,
    jacobi_eigenvalues_symmetric,
    matmul,
    matvec,
    negative_eigenvalue_count,
    norm,
    outer,
    solve_linear_system,
    subtract_matrices,
    transpose,
)


class MatrixTests(unittest.TestCase):
    def assertVectorAlmostEqual(
        self,
        actual: list[float],
        expected: list[float],
        places: int = 7,
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_basic_vector_operations(self) -> None:
        self.assertEqual(add_vectors([1, 2, 3], [4, 5, 6]), [5.0, 7.0, 9.0])
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)
        self.assertAlmostEqual(norm([3, 4]), 5.0)

    def test_basic_matrix_operations(self) -> None:
        self.assertEqual(identity(3), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        self.assertEqual(
            subtract_matrices([[5, 7], [9, 11]], [[1, 2], [3, 4]]),
            [[4.0, 5.0], [6.0, 7.0]],
        )
        self.assertEqual(matvec([[2, 0], [1, 3]], [4, 5]), [8.0, 19.0])
        self.assertEqual(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19.0, 22.0], [43.0, 50.0]])
        self.assertEqual(outer([1, 2], [3, 4, 5]), [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])

    def test_determinant_and_solve(self) -> None:
        coefficients = [[3.0, 2.0], [1.0, 2.0]]
        self.assertAlmostEqual(determinant(coefficients), 4.0)
        self.assertVectorAlmostEqual(solve_linear_system(coefficients, [5.0, 5.0]), [0.0, 2.5])

    def test_symmetric_eigenvalue_tools(self) -> None:
        self.assertTrue(is_symmetric([[2.0, 1.0], [1.0, 2.0]]))
        self.assertFalse(is_symmetric([[2.0, 1.0], [0.0, 2.0]]))
        self.assertVectorAlmostEqual(
            jacobi_eigenvalues_symmetric([[2.0, 1.0], [1.0, 2.0]]),
            [1.0, 3.0],
        )

    def test_first_order_saddle_requires_exactly_one_negative_eigenvalue(self) -> None:
        self.assertEqual(negative_eigenvalue_count([[-1.0, 0.0], [0.0, 2.0]]), 1)
        self.assertTrue(is_first_order_saddle([[-1.0, 0.0], [0.0, 2.0]]))
        self.assertFalse(is_first_order_saddle([[-1.0, 0.0], [0.0, -2.0]]))
        self.assertFalse(is_first_order_saddle([[1.0, 0.0], [0.0, 2.0]]))


if __name__ == "__main__":
    unittest.main()
