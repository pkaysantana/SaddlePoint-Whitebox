# Layer C Strategic / Leverage Layer

## A. Algorithmic Defense

The engine only assumes an energy function with the shape `E(x) -> scalar`.
That single interface is enough for the existing calculus, Hessian
classification, label generation, benchmarking, and optimizer code.

Because of that boundary, the same engine can work with toy mathematical
surfaces, empirical potentials, quantum chemistry wrappers, or future learned
potentials. Numerical differentiation is the reason this is domain-agnostic:
the engine does not need analytic gradients or Hessians from the energy model.
The tradeoff is cost, because every gradient and Hessian element requires
additional scalar energy evaluations.

## B. Performance Bottleneck

For a coordinate vector of dimension `d`, finite-difference Hessians scale
poorly. The diagonal terms require displaced evaluations for every coordinate,
and the off-diagonal mixed terms require corner evaluations for every coordinate
pair.

A molecular system of 12-15 atoms in Cartesian coordinates is roughly 36-45
dimensions. That is feasible for a whitebox proof of concept and careful small
experiments, but it is not ideal for large molecules. The bottleneck is scalar
energy evaluations, not Python syntax itself. Pure Python loops are also slow,
but this project is intentionally built for learning, verification, and
small-system experimentation before moving heavy production work elsewhere.

## C. Wheland Intermediate Use Case

A benzene plus electrophile or Wheland intermediate study is a plausible future
demonstration. The immediate project should not pretend to perform ab initio
quantum chemistry by itself.

What the whitebox engine can do now is classify a candidate coordinate vector as
a minimum, first-order saddle, higher-order saddle, maximum, or indeterminate if
it is supplied with a valid energy model. Later, that energy function could be
connected to xTB, ORCA, Psi4, PySCF, or a learned potential in a separate
package-based project.

## D. Future ML Integration

The labels generated here are useful ML supervision targets:

- energy
- gradient
- Hessian eigenvalues
- negative eigenvalue count
- reaction-coordinate eigenvector
- stationary point classification

These can train later models for saddle/minimum classification, approximate
barrier height prediction, optimizer-step prediction, and reaction-coordinate
prediction.

## E. Boundary Conditions / Honesty

This does not replace Gaussian, ORCA, xTB, Psi4, or professional transition
state search methods. It is a transparent educational and research scaffold.
Its value is understanding, label generation, and prototype design.
