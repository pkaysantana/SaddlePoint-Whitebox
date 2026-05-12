"""Print diagnostics for synthetic benzene-electrophile topology guesses."""

from __future__ import annotations

from saddlepoint_whitebox.diagnostics import diagnose_saddle_candidate
from saddlepoint_whitebox.topology_models import (
    benzene_electrophile_starting_points,
    benzene_electrophile_topology_energy,
)


def main() -> None:
    for name, coordinates in benzene_electrophile_starting_points().items():
        diagnostic = diagnose_saddle_candidate(
            name,
            benzene_electrophile_topology_energy,
            coordinates,
        )
        print(f"{name}")
        print(f"  coordinates: {diagnostic.coordinates}")
        print(f"  energy: {diagnostic.energy:.8f}")
        print(f"  gradient norm: {diagnostic.gradient_norm:.8e}")
        print(f"  eigenvalues: {diagnostic.eigenvalues}")
        print(f"  classification: {diagnostic.classification}")
        print(f"  negative eigenvalue count: {diagnostic.negative_eigenvalue_count}")
        print(f"  reaction coordinate: {diagnostic.reaction_coordinate}")
        if diagnostic.warning:
            print(f"  warning: {diagnostic.warning}")
        print()


if __name__ == "__main__":
    main()
