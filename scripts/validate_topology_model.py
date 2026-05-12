"""Validate the default synthetic benzene-electrophile topology model."""

from __future__ import annotations

from saddlepoint_whitebox.topology_validation import (
    validate_benzene_electrophile_topology,
)


def main() -> None:
    report = validate_benzene_electrophile_topology()

    print(f"model: {report.model_name}")
    print()
    for candidate in report.candidates:
        print(candidate.name)
        print(f"  method: {candidate.method}")
        print(f"  converged: {candidate.converged}")
        print(f"  initial energy: {candidate.initial_energy:.8f}")
        print(f"  final energy: {candidate.final_energy:.8f}")
        print(f"  final coordinates: {candidate.final_coordinates}")
        print(f"  gradient norm: {candidate.gradient_norm:.8e}")
        print(f"  classification: {candidate.classification}")
        print(f"  Hessian eigenvalues: {candidate.hessian_eigenvalues}")
        print(f"  negative eigenvalue count: {candidate.negative_eigenvalue_count}")
        if candidate.reaction_coordinate is not None:
            print(f"  reaction coordinate: {candidate.reaction_coordinate}")
        if candidate.warning is not None:
            print(f"  warning: {candidate.warning}")
        print()

    print(f"passed: {report.passed}")
    print(f"summary: {report.summary}")


if __name__ == "__main__":
    main()
