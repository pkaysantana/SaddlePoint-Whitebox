"""Dependency-free vector and matrix operations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import copysign, isclose, isfinite, sqrt

NumberLike = int | float
Vector = list[float]
Matrix = list[list[float]]


def zeros(rows: int, columns: int) -> Matrix:
    """Return a rows-by-columns zero matrix."""

    _require_positive_integer(rows, "rows")
    _require_positive_integer(columns, "columns")
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    """Return an identity matrix."""

    _require_positive_integer(size, "size")
    values = zeros(size, size)
    for index in range(size):
        values[index][index] = 1.0
    return values


def shape(matrix: Iterable[Iterable[NumberLike]]) -> tuple[int, int]:
    """Return ``(rows, columns)`` for a rectangular matrix."""

    values = _as_matrix(matrix, "matrix")
    return len(values), len(values[0])


def add_vectors(left: Iterable[NumberLike], right: Iterable[NumberLike]) -> Vector:
    """Return element-wise vector addition."""

    left_values = _as_vector(left, "left")
    right_values = _as_vector(right, "right")
    _require_same_length(left_values, right_values)
    return [a + b for a, b in zip(left_values, right_values)]


def subtract_vectors(left: Iterable[NumberLike], right: Iterable[NumberLike]) -> Vector:
    """Return element-wise vector subtraction."""

    left_values = _as_vector(left, "left")
    right_values = _as_vector(right, "right")
    _require_same_length(left_values, right_values)
    return [a - b for a, b in zip(left_values, right_values)]


def scale_vector(vector: Iterable[NumberLike], scalar: NumberLike) -> Vector:
    """Return ``scalar * vector``."""

    values = _as_vector(vector, "vector")
    factor = _as_scalar(scalar, "scalar")
    return [factor * value for value in values]


def dot(left: Iterable[NumberLike], right: Iterable[NumberLike]) -> float:
    """Return the Euclidean dot product."""

    left_values = _as_vector(left, "left")
    right_values = _as_vector(right, "right")
    _require_same_length(left_values, right_values)
    return sum(a * b for a, b in zip(left_values, right_values))


def norm(vector: Iterable[NumberLike]) -> float:
    """Return the Euclidean vector norm."""

    return sqrt(dot(vector, vector))


def normalize_vector(
    vector: Iterable[NumberLike],
    tolerance: float = 1.0e-15,
) -> Vector:
    """Return a normalized copy of ``vector``."""

    values = _as_vector(vector, "vector")
    length = norm(values)
    if length <= tolerance:
        raise ValueError("cannot normalize a zero-length vector")
    return [value / length for value in values]


def project(
    vector: Iterable[NumberLike],
    onto: Iterable[NumberLike],
    tolerance: float = 1.0e-15,
) -> Vector:
    """Project ``vector`` onto ``onto``."""

    vector_values = _as_vector(vector, "vector")
    onto_values = normalize_vector(onto, tolerance=tolerance)
    coefficient = dot(vector_values, onto_values)
    return scale_vector(onto_values, coefficient)


def vector_distance(
    left: Iterable[NumberLike],
    right: Iterable[NumberLike],
) -> float:
    """Return the Euclidean distance between two vectors."""

    return norm(subtract_vectors(left, right))


def add_matrices(
    left: Iterable[Iterable[NumberLike]],
    right: Iterable[Iterable[NumberLike]],
) -> Matrix:
    """Return element-wise matrix addition."""

    left_values = _as_matrix(left, "left")
    right_values = _as_matrix(right, "right")
    _require_same_shape(left_values, right_values)
    return [
        [left_value + right_value for left_value, right_value in zip(left_row, right_row)]
        for left_row, right_row in zip(left_values, right_values)
    ]


def subtract_matrices(
    left: Iterable[Iterable[NumberLike]],
    right: Iterable[Iterable[NumberLike]],
) -> Matrix:
    """Return element-wise matrix subtraction."""

    left_values = _as_matrix(left, "left")
    right_values = _as_matrix(right, "right")
    _require_same_shape(left_values, right_values)
    return [
        [left_value - right_value for left_value, right_value in zip(left_row, right_row)]
        for left_row, right_row in zip(left_values, right_values)
    ]


def scale_matrix(matrix: Iterable[Iterable[NumberLike]], scalar: NumberLike) -> Matrix:
    """Return ``scalar * matrix``."""

    values = _as_matrix(matrix, "matrix")
    factor = _as_scalar(scalar, "scalar")
    return [[factor * value for value in row] for row in values]


def transpose(matrix: Iterable[Iterable[NumberLike]]) -> Matrix:
    """Return the matrix transpose."""

    values = _as_matrix(matrix, "matrix")
    return [list(column) for column in zip(*values)]


def matmul(
    left: Iterable[Iterable[NumberLike]],
    right: Iterable[Iterable[NumberLike]],
) -> Matrix:
    """Return matrix multiplication ``left @ right``."""

    left_values = _as_matrix(left, "left")
    right_values = _as_matrix(right, "right")
    left_rows, left_columns = len(left_values), len(left_values[0])
    right_rows, right_columns = len(right_values), len(right_values[0])
    if left_columns != right_rows:
        raise ValueError(
            "matrix dimensions are incompatible: "
            f"{left_rows}x{left_columns} cannot multiply {right_rows}x{right_columns}"
        )

    right_transposed = transpose(right_values)
    return [
        [dot(left_row, right_column) for right_column in right_transposed]
        for left_row in left_values
    ]


def matvec(
    matrix: Iterable[Iterable[NumberLike]],
    vector: Iterable[NumberLike],
) -> Vector:
    """Return matrix-vector multiplication."""

    values = _as_matrix(matrix, "matrix")
    vector_values = _as_vector(vector, "vector")
    if len(values[0]) != len(vector_values):
        raise ValueError(
            "matrix and vector dimensions are incompatible: "
            f"{len(values)}x{len(values[0])} cannot multiply length {len(vector_values)}"
        )
    return [dot(row, vector_values) for row in values]


def outer(left: Iterable[NumberLike], right: Iterable[NumberLike]) -> Matrix:
    """Return the outer product of two vectors."""

    left_values = _as_vector(left, "left")
    right_values = _as_vector(right, "right")
    return [[left_value * right_value for right_value in right_values] for left_value in left_values]


def trace(matrix: Iterable[Iterable[NumberLike]]) -> float:
    """Return the trace of a square matrix."""

    values = _as_square_matrix(matrix, "matrix")
    return sum(values[index][index] for index in range(len(values)))


def determinant(
    matrix: Iterable[Iterable[NumberLike]],
    tolerance: float = 1.0e-15,
) -> float:
    """Return the determinant using Gaussian elimination with pivoting."""

    values = _as_square_matrix(matrix, "matrix")
    size = len(values)
    det = 1.0
    sign = 1.0

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(values[row][column]))
        if abs(values[pivot][column]) <= tolerance:
            return 0.0
        if pivot != column:
            values[column], values[pivot] = values[pivot], values[column]
            sign *= -1.0

        pivot_value = values[column][column]
        det *= pivot_value
        for row in range(column + 1, size):
            factor = values[row][column] / pivot_value
            values[row][column] = 0.0
            for inner_column in range(column + 1, size):
                values[row][inner_column] -= factor * values[column][inner_column]

    return sign * det


def solve_linear_system(
    coefficients: Iterable[Iterable[NumberLike]],
    constants: Iterable[NumberLike],
    tolerance: float = 1.0e-15,
) -> Vector:
    """Solve ``coefficients @ x = constants`` using Gaussian elimination."""

    values = _as_square_matrix(coefficients, "coefficients")
    rhs = _as_vector(constants, "constants")
    size = len(values)
    if len(rhs) != size:
        raise ValueError("constants length must match coefficient matrix size")

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(values[row][column]))
        if abs(values[pivot][column]) <= tolerance:
            raise ValueError("linear system is singular or ill-conditioned")
        if pivot != column:
            values[column], values[pivot] = values[pivot], values[column]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]

        pivot_value = values[column][column]
        for row in range(column + 1, size):
            factor = values[row][column] / pivot_value
            values[row][column] = 0.0
            for inner_column in range(column + 1, size):
                values[row][inner_column] -= factor * values[column][inner_column]
            rhs[row] -= factor * rhs[column]

    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        remaining = sum(values[row][column] * solution[column] for column in range(row + 1, size))
        solution[row] = (rhs[row] - remaining) / values[row][row]
    return solution


def is_symmetric(
    matrix: Iterable[Iterable[NumberLike]],
    tolerance: float = 1.0e-9,
) -> bool:
    """Return whether a square matrix is symmetric within ``tolerance``."""

    values = _as_square_matrix(matrix, "matrix")
    size = len(values)
    for row in range(size):
        for column in range(row + 1, size):
            if not isclose(values[row][column], values[column][row], abs_tol=tolerance):
                return False
    return True


def jacobi_eigenvalues_symmetric(
    matrix: Iterable[Iterable[NumberLike]],
    tolerance: float = 1.0e-12,
    max_iterations: int | None = None,
) -> Vector:
    """Return eigenvalues of a real symmetric matrix with Jacobi rotations."""

    eigenvalues, _ = jacobi_eigendecomposition_symmetric(
        matrix,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return eigenvalues


def jacobi_eigendecomposition_symmetric(
    matrix: Iterable[Iterable[NumberLike]],
    tolerance: float = 1.0e-12,
    max_iterations: int | None = None,
) -> tuple[Vector, list[Vector]]:
    """Return sorted eigenvalues and normalized eigenvectors of a symmetric matrix.

    The Jacobi method repeatedly applies plane rotations that zero the largest
    off-diagonal element. For a real symmetric matrix those rotations converge
    to a diagonal matrix, and the product of all rotations is the eigenvector
    matrix. Eigenvectors are returned as column vectors, but stored as a list of
    vectors so ``eigenvectors[k]`` belongs to ``eigenvalues[k]``.
    """

    values = _as_square_matrix(matrix, "matrix")
    if not is_symmetric(values, tolerance=max(tolerance * 100.0, 1.0e-10)):
        raise ValueError("Jacobi eigenvalue decomposition requires a symmetric matrix")

    size = len(values)
    if size == 1:
        return [values[0][0]], [[1.0]]

    eigenvector_matrix = identity(size)
    iterations = max_iterations if max_iterations is not None else 100 * size * size

    for _ in range(iterations):
        row, column, off_diagonal = _largest_off_diagonal(values)
        if off_diagonal <= tolerance:
            break

        a_pp = values[row][row]
        a_qq = values[column][column]
        a_pq = values[row][column]
        tau = (a_qq - a_pp) / (2.0 * a_pq)
        tangent = copysign(1.0, tau) / (abs(tau) + sqrt(1.0 + tau * tau))
        cosine = 1.0 / sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        values[row][row] = a_pp - tangent * a_pq
        values[column][column] = a_qq + tangent * a_pq
        values[row][column] = 0.0
        values[column][row] = 0.0

        for index in range(size):
            if index in (row, column):
                continue
            a_ip = values[index][row]
            a_iq = values[index][column]
            values[index][row] = cosine * a_ip - sine * a_iq
            values[row][index] = values[index][row]
            values[index][column] = sine * a_ip + cosine * a_iq
            values[column][index] = values[index][column]

        # Accumulate the same rotation into the eigenvector matrix. The matrix
        # stores eigenvectors as columns, so each row contributes two entries
        # that are rotated in the active (row, column) plane.
        for index in range(size):
            v_ip = eigenvector_matrix[index][row]
            v_iq = eigenvector_matrix[index][column]
            eigenvector_matrix[index][row] = cosine * v_ip - sine * v_iq
            eigenvector_matrix[index][column] = sine * v_ip + cosine * v_iq
    else:
        raise ValueError("Jacobi eigenvalue decomposition did not converge")

    pairs: list[tuple[float, Vector]] = []
    for index in range(size):
        eigenvector = [eigenvector_matrix[row][index] for row in range(size)]
        pairs.append((values[index][index], _orient_vector(normalize_vector(eigenvector))))

    pairs.sort(key=lambda pair: pair[0])
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def negative_eigenvalue_count(
    matrix: Iterable[Iterable[NumberLike]],
    tolerance: float = 1.0e-8,
) -> int:
    """Return the number of eigenvalues strictly below ``-tolerance``."""

    return sum(
        1 for eigenvalue in jacobi_eigenvalues_symmetric(matrix) if eigenvalue < -tolerance
    )


def is_first_order_saddle(
    hessian: Iterable[Iterable[NumberLike]],
    tolerance: float = 1.0e-8,
) -> bool:
    """Return whether a Hessian has exactly one negative eigenvalue."""

    return negative_eigenvalue_count(hessian, tolerance=tolerance) == 1


has_first_order_saddle = is_first_order_saddle


def max_abs_off_diagonal(matrix: Iterable[Iterable[NumberLike]]) -> float:
    """Return the largest absolute off-diagonal value in a square matrix."""

    values = _as_square_matrix(matrix, "matrix")
    if len(values) == 1:
        return 0.0
    return _largest_off_diagonal(values)[2]


def _largest_off_diagonal(matrix: Matrix) -> tuple[int, int, float]:
    size = len(matrix)
    best_row = 0
    best_column = 1
    best_value = abs(matrix[best_row][best_column])
    for row in range(size):
        for column in range(row + 1, size):
            value = abs(matrix[row][column])
            if value > best_value:
                best_row = row
                best_column = column
                best_value = value
    return best_row, best_column, best_value


def _orient_vector(vector: Vector) -> Vector:
    """Choose a deterministic sign for an eigenvector.

    Eigenvectors are mathematically unchanged by a sign flip. Picking the sign
    so the largest-magnitude component is positive keeps test expectations and
    user-facing output stable across runs.
    """

    largest_index = max(range(len(vector)), key=lambda index: abs(vector[index]))
    if vector[largest_index] < 0.0:
        return [-value for value in vector]
    return vector


def _as_vector(vector: Iterable[NumberLike], name: str) -> Vector:
    values = [_as_scalar(value, name) for value in vector]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _as_matrix(matrix: Iterable[Iterable[NumberLike]], name: str) -> Matrix:
    values = [[_as_scalar(value, name) for value in row] for row in matrix]
    if not values:
        raise ValueError(f"{name} must contain at least one row")
    columns = len(values[0])
    if columns == 0:
        raise ValueError(f"{name} rows must contain at least one value")
    if any(len(row) != columns for row in values):
        raise ValueError(f"{name} must be rectangular")
    return values


def _as_square_matrix(matrix: Iterable[Iterable[NumberLike]], name: str) -> Matrix:
    values = _as_matrix(matrix, name)
    if len(values) != len(values[0]):
        raise ValueError(f"{name} must be square")
    return values


def _as_scalar(value: NumberLike, name: str) -> float:
    scalar = float(value)
    if not isfinite(scalar):
        raise ValueError(f"{name} values must be finite numbers")
    return scalar


def _require_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_same_length(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError(f"vector lengths differ: {len(left)} != {len(right)}")


def _require_same_shape(left: Matrix, right: Matrix) -> None:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ValueError(
            "matrix shapes differ: "
            f"{len(left)}x{len(left[0])} != {len(right)}x{len(right[0])}"
        )
