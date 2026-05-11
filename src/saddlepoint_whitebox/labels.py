"""Export-friendly labels for ML and benchmark datasets."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .calculus import evaluate_pes
from .classification import classify_pes_point
from .matrix import jacobi_eigendecomposition_symmetric, norm

ScalarFunction = Callable[[list[float]], float]


@dataclass(frozen=True)
class PESLabel:
    """Ground-truth PES quantities at one coordinate vector."""

    name: str
    coordinates: tuple[float, ...]
    energy: float
    gradient: tuple[float, ...]
    gradient_norm: float
    hessian: tuple[tuple[float, ...], ...]
    hessian_eigenvalues: tuple[float, ...]
    negative_eigenvalue_count: int
    classification: str
    reaction_coordinate: tuple[float, ...] | None
    metadata: dict[str, str]


def generate_pes_label(
    name: str,
    energy: ScalarFunction,
    coordinates: Iterable[float],
    metadata: dict[str, str] | None = None,
    finite_difference_step: float = 1.0e-5,
) -> PESLabel:
    """Generate an export-ready label from any scalar energy function."""

    point = evaluate_pes(
        energy,
        coordinates,
        gradient_step=finite_difference_step,
        hessian_step=finite_difference_step,
    )
    eigenvalues, eigenvectors = jacobi_eigendecomposition_symmetric(point.hessian)
    negative_count = sum(1 for eigenvalue in eigenvalues if eigenvalue < -1.0e-8)
    reaction_coordinate = tuple(eigenvectors[0]) if negative_count > 0 else None
    return PESLabel(
        name=name,
        coordinates=point.coordinates,
        energy=point.energy,
        gradient=point.gradient,
        gradient_norm=norm(point.gradient),
        hessian=point.hessian,
        hessian_eigenvalues=tuple(eigenvalues),
        negative_eigenvalue_count=negative_count,
        classification=classify_pes_point(point).value,
        reaction_coordinate=reaction_coordinate,
        metadata=dict(metadata or {}),
    )


def label_to_dict(label: PESLabel) -> dict[str, object]:
    """Convert a label to JSON-serializable built-in containers."""

    return {
        "name": label.name,
        "coordinates": list(label.coordinates),
        "energy": label.energy,
        "gradient": list(label.gradient),
        "gradient_norm": label.gradient_norm,
        "hessian": [list(row) for row in label.hessian],
        "hessian_eigenvalues": list(label.hessian_eigenvalues),
        "negative_eigenvalue_count": label.negative_eigenvalue_count,
        "classification": label.classification,
        "reaction_coordinate": (
            list(label.reaction_coordinate)
            if label.reaction_coordinate is not None
            else None
        ),
        "metadata": dict(label.metadata),
    }


def export_labels_jsonl(labels: Iterable[PESLabel], path: str | Path) -> None:
    """Write labels as newline-delimited JSON."""

    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label_to_dict(label), sort_keys=True))
            handle.write("\n")


def export_labels_csv_summary(labels: Iterable[PESLabel], path: str | Path) -> None:
    """Write a compact CSV summary of PES labels."""

    fieldnames = [
        "name",
        "dimension",
        "energy",
        "gradient_norm",
        "classification",
        "negative_eigenvalue_count",
        "hessian_eigenvalues",
        "metadata",
    ]
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            writer.writerow(
                {
                    "name": label.name,
                    "dimension": len(label.coordinates),
                    "energy": label.energy,
                    "gradient_norm": label.gradient_norm,
                    "classification": label.classification,
                    "negative_eigenvalue_count": label.negative_eigenvalue_count,
                    "hessian_eigenvalues": ";".join(
                        str(value) for value in label.hessian_eigenvalues
                    ),
                    "metadata": json.dumps(label.metadata, sort_keys=True),
                }
            )
