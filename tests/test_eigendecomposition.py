from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saddlepoint_whitebox.matrix import (
    jacobi_eigendecomposition_symmetric,
    matvec,
    max_abs_off_diagonal,
    norm,
    normalize_vector,
    project,
    scale_vector,
    vector_distance,
)


class EigendecompositionTests(unittest.TestCase):
    def assertVectorAlmostEqual(
        self,
        actual: list[float],
        expected: list[float],
        places: int = 7,
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_diagonal_matrix_eigenpairs_are_sorted(self) -> None:
        eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(
            [[3.0, 0.0], [0.0, 1.0]]
        )

        self.assertVectorAlmostEqual(eigenvalues, [1.0, 3.0])
        self.assertVectorAlmostEqual(eigenvectors[0], [0.0, 1.0])
        self.assertVectorAlmostEqual(eigenvectors[1], [1.0, 0.0])

    def test_two_by_two_symmetric_matrix_eigenvalues(self) -> None:
        eigenvalues, _ = jacobi_eigendecomposition_symmetric(
            [[2.0, 1.0], [1.0, 2.0]]
        )

        self.assertVectorAlmostEqual(eigenvalues, [1.0, 3.0])

    def test_eigenpair_identity(self) -> None:
        matrix = [[2.0, 1.0], [1.0, 2.0]]
        eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(matrix)

        for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
            left = matvec(matrix, eigenvector)
            right = scale_vector(eigenvector, eigenvalue)
            self.assertVectorAlmostEqual(left, right)
            self.assertAlmostEqual(norm(eigenvector), 1.0)

    def test_vector_helpers(self) -> None:
        self.assertVectorAlmostEqual(normalize_vector([3.0, 4.0]), [0.6, 0.8])
        self.assertVectorAlmostEqual(project([2.0, 2.0], [1.0, 0.0]), [2.0, 0.0])
        self.assertAlmostEqual(vector_distance([1.0, 2.0], [4.0, 6.0]), 5.0)
        self.assertAlmostEqual(max_abs_off_diagonal([[1.0, -3.0], [2.0, 4.0]]), 3.0)


if __name__ == "__main__":
    unittest.main()
