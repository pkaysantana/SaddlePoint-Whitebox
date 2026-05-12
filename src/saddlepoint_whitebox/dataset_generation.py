"""Synthetic PES dataset generation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from random import Random

from .labels import PESLabel, export_labels_csv_summary, export_labels_jsonl, generate_pes_label
from .topology_models import (
    benzene_electrophile_starting_points,
    benzene_electrophile_topology_energy,
)

ScalarFunction = Callable[[list[float]], float]


@dataclass(frozen=True)
class PerturbationSettings:
    """Settings for random coordinate perturbation label generation."""

    samples: int = 1000
    radius: float = 0.25
    seed: int = 123
    finite_difference_step: float = 1.0e-5


def generate_perturbed_labels(
    name: str,
    energy: ScalarFunction,
    center_coordinates: Iterable[float],
    settings: PerturbationSettings | None = None,
    metadata: dict[str, str] | None = None,
) -> list[PESLabel]:
    """Generate finite-difference PES labels around one coordinate center."""

    active_settings = settings or PerturbationSettings()
    _validate_settings(active_settings)
    center = _as_finite_tuple(center_coordinates, "center_coordinates")
    random = Random(active_settings.seed)
    labels: list[PESLabel] = []

    for sample_index in range(active_settings.samples):
        coordinates = tuple(
            value + random.uniform(-active_settings.radius, active_settings.radius)
            for value in center
        )
        label_metadata = dict(metadata or {})
        label_metadata.update(
            {
                "source_center": _format_coordinates(center),
                "perturbation_radius": str(active_settings.radius),
                "sample_index": str(sample_index),
                "model_name": name,
            }
        )
        labels.append(
            generate_pes_label(
                name=name,
                energy=energy,
                coordinates=coordinates,
                metadata=label_metadata,
                finite_difference_step=active_settings.finite_difference_step,
            )
        )
    return labels


def generate_topology_dataset(
    output_jsonl_path: str | Path,
    output_csv_path: str | Path,
    samples: int = 1000,
    seed: int = 123,
) -> list[PESLabel]:
    """Generate and export perturbed labels around topology-model guesses."""

    if not isinstance(samples, int) or samples < 0:
        raise ValueError("samples must be a non-negative integer")

    centers = benzene_electrophile_starting_points()
    center_items = list(centers.items())
    base_count = samples // len(center_items)
    remainder = samples % len(center_items)
    all_labels: list[PESLabel] = []

    for center_index, (center_name, center_coordinates) in enumerate(center_items):
        center_samples = base_count + (1 if center_index < remainder else 0)
        if center_samples == 0:
            continue
        settings = PerturbationSettings(
            samples=center_samples,
            radius=0.25,
            seed=seed + center_index,
            finite_difference_step=1.0e-5,
        )
        all_labels.extend(
            generate_perturbed_labels(
                name="benzene-electrophile synthetic topology",
                energy=benzene_electrophile_topology_energy,
                center_coordinates=center_coordinates,
                settings=settings,
                metadata={"center_name": center_name},
            )
        )

    output_jsonl = Path(output_jsonl_path)
    output_csv = Path(output_csv_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_labels_jsonl(all_labels, output_jsonl)
    export_labels_csv_summary(all_labels, output_csv)
    return all_labels


def _validate_settings(settings: PerturbationSettings) -> None:
    if not isinstance(settings.samples, int) or settings.samples < 0:
        raise ValueError("samples must be a non-negative integer")
    if settings.radius < 0.0 or not isfinite(settings.radius):
        raise ValueError("radius must be a non-negative finite number")
    if not isinstance(settings.seed, int):
        raise ValueError("seed must be an integer")
    if settings.finite_difference_step <= 0.0 or not isfinite(
        settings.finite_difference_step
    ):
        raise ValueError("finite_difference_step must be a positive finite number")


def _as_finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{name} must contain at least one value")
    if any(not isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _format_coordinates(values: tuple[float, ...]) -> str:
    return ",".join(str(value) for value in values)
