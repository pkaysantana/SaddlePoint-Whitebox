# Hessian Update Roadmap

## Why Full Finite-Difference Hessians Scale Poorly

Finite-difference Hessians are useful because they only require an energy
function `E(x) -> scalar`, but they become expensive as the coordinate dimension
grows. The diagonal terms need displaced evaluations for every coordinate, and
the off-diagonal mixed terms need paired displacements for every coordinate
pair. For dimension `d`, the number of scalar energy calls grows roughly with
`d^2`.

That cost is acceptable for this whitebox project, small surfaces, and
educational diagnostics. It is not a scalable strategy for large molecular
systems or high-throughput transition-state searches.

## Why BFGS Alone Is Not Enough For Saddles

BFGS-style updates are effective for minimization because they try to maintain a
positive-definite Hessian approximation. That is exactly what a minimum search
wants: all local curvatures positive.

A first-order saddle is different. Its Hessian should have exactly one negative
eigenvalue. If an update aggressively preserves positive definiteness, it can
erase the negative mode that an eigenvector-following transition-state search
needs.

## Why Bofill And Powell-Like Updates Matter

Transition-state searches often use Hessian update schemes that can retain or
recover indefinite curvature. Bofill and Powell-like updates blend information
from gradient changes and coordinate displacements without forcing the Hessian
to remain positive definite. That makes them more appropriate for saddle-search
workflows than a plain minimization-oriented update.

## Data Needed Per Iteration

A future Hessian-update layer should record:

- old coordinates
- new coordinates
- old gradient
- new gradient
- previous Hessian estimate

From these values, the update can compare the actual gradient change with the
change predicted by the current Hessian estimate.

## Implementation Timing

This should come after the EVF and synthetic topology layers have been validated
with clear diagnostics. The current finite-difference Hessian remains the
reference implementation. A later update layer should be tested against that
reference on toy surfaces before being used in saddle-search iterations.
