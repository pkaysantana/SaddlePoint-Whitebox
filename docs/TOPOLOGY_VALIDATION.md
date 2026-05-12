# Topology Calibration and Validation

## Purpose

The synthetic benzene-electrophile model is useful only if its local topology
matches the intended educational mechanism. Validation asks whether the model's
own gradients, Hessians, eigenvalues, and optimizer behavior support that
mechanistic shape.

This is a calibration layer for a reduced-order PES, not a comparison to real
electronic-structure data.

## Desired Mechanistic Topology

The target synthetic topology is:

- `pi_complex_guess` relaxes toward a local minimum.
- `rim_complex_guess` relaxes toward an intermediate-like or minimum-like basin.
- `saddle_guess` optimizes toward a first-order saddle with exactly one
  negative Hessian eigenvalue.

Those outcomes make the surface useful for testing saddle diagnostics,
reaction-coordinate extraction, and ML label generation.

## Why Minima and First-Order Saddles Matter

Minima represent locally stable basins on a potential energy surface. First-order
saddles represent transition-state-like topology: one unstable mode and
restoring curvature in the remaining modes.

Energy values alone do not prove either structure. The Hessian eigenvalue signs
are the important local evidence. A first-order saddle should have exactly one
negative Hessian eigenvalue, and the associated eigenvector is the candidate
reaction coordinate.

## How the Validation Layer Works

The validation module runs local minimization from the pi and rim guesses, then
uses eigenvector following from the saddle guess. It evaluates each final point
with the existing finite-difference calculus pipeline, classifies the Hessian,
counts negative eigenvalues, and records a reaction-coordinate eigenvector when
a negative mode is present.

The report keeps warnings instead of failing catastrophically. This makes
parameter tuning explicit: a synthetic model can be inspected even when the
current parameters do not yet satisfy every topology rule.

## Why This Is Still Synthetic

The benzene-electrophile model is a reduced mathematical surface. It does not
include electronic structure, basis sets, correlation methods, solvent models,
thermochemistry, or experimentally validated barriers.

The validation layer checks whether this project's own synthetic PES behaves as
intended. It does not claim agreement with CCSD(T), Gaussian, ORCA, Psi4, xTB,
or experiment.

## Next Step: Adaptive Finite Differences and Then Package-Based Computational Chemistry

The next whitebox improvement is better finite-difference control, especially
adaptive steps for gradients, Hessians, and optimizer validation. After that, a
separate package-based chemistry workflow can connect the same energy-function
interface to tools such as xTB, Psi4, ORCA, PySCF, RDKit, or ASE.

Keeping those package integrations separate preserves this repository as a
small, dependency-free reference layer for gradients, Hessians, eigenvalues,
classifications, and reaction-coordinate labels.
