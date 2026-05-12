# SaddlePoint Whitebox

Small, inspectable tools for exploring potential energy surfaces.

The initial core is dependency-free and centered on:

- vector and matrix operations in `saddlepoint_whitebox.matrix`
- central-difference numerical differentiation in `saddlepoint_whitebox.calculus`
- Hessian eigendecomposition and stationary point classification
- basic eigenvector-following transition-state searches on small test surfaces

## Example

```python
from saddlepoint_whitebox.calculus import evaluate_pes
from saddlepoint_whitebox.matrix import is_first_order_saddle


def energy(x):
    return x[0] ** 2 - x[1] ** 2 + 0.5 * x[0] * x[1]


coords = [0.25, -0.5]

point = evaluate_pes(energy, coords)

print(point.energy)
print(point.gradient)
print(point.force)  # Physical convention: F = -grad(E)
print(point.hessian)
print(is_first_order_saddle(point.hessian))
```

## Transition-State Search Demo

```python
from saddlepoint_whitebox.optimizers import eigenvector_following_saddle_search
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


result = eigenvector_following_saddle_search(
    quadratic_first_order_saddle,
    initial_coordinates=[0.05, 0.05],
)

print(result.final_coordinates)
print(result.final_energy)
print(result.final_eigenvalues)
print(result.final_classification)
```

## Explicit Eigenvector-Following Step

```python
from saddlepoint_whitebox.evf import EVFOptimizer, reaction_coordinate_from_hessian


gradient = [0.0, 1.0]
eigenvalues = [-2.0, 4.0]
eigenvectors = [[1.0, 0.0], [0.0, 1.0]]

step_result = EVFOptimizer().compute_step(gradient, eigenvalues, eigenvectors)
target_eigenvalue, target_mode = reaction_coordinate_from_hessian(
    [[-2.0, 0.0], [0.0, 4.0]]
)

print(step_result.step)
print(step_result.mode_gradients)
print(step_result.target_eigenvalue)
print(target_eigenvalue, target_mode)
```

## Fixed And Adaptive Finite-Difference Steps

```python
from saddlepoint_whitebox.calculus import gradient, hessian


def energy(x):
    return x[0] ** 2 - x[1] ** 2


print(gradient(energy, [0.0, 0.0], step=1.0e-5))
print(hessian(energy, [0.0, 0.0], step="adaptive"))
```

Fixed steps are useful for reproducible numerical experiments. Adaptive steps
scale modestly with coordinate magnitude and are helpful near stationary
regions, but they do not remove the finite-difference cost of gradient and
Hessian evaluation.

## Layer C: Benchmarking And ML Labels

```python
from saddlepoint_whitebox.benchmarks import benchmark_pes_evaluation
from saddlepoint_whitebox.calculus import evaluate_pes
from saddlepoint_whitebox.labels import export_labels_jsonl, generate_pes_label
from saddlepoint_whitebox.physical_models import lennard_jones_cluster_energy
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


benchmark = benchmark_pes_evaluation(
    "quadratic saddle",
    quadratic_first_order_saddle,
    [0.0, 0.0],
)
print(benchmark.energy_calls, benchmark.classification)

label = generate_pes_label("quadratic saddle", quadratic_first_order_saddle, [0.0, 0.0])
export_labels_jsonl([label], "pes_labels.jsonl")

lj_point = evaluate_pes(
    lambda x: lennard_jones_cluster_energy(x, dimensions=2),
    [0.0, 0.0, 1.2, 0.0],
)
print(lj_point.energy, lj_point.gradient)
```

## Reduced-Order Benzene-Electrophile Topology Model

This synthetic reduced-order model uses coordinates `[rho, z, q]`: `rho` is
lateral displacement from the ring center toward the rim, `z` is height above
the aromatic plane, and `q` is a bond-formation / arenium-character coordinate.

```python
from saddlepoint_whitebox.topology_models import (
    benzene_electrophile_topology_energy,
    benzene_electrophile_starting_points,
    evaluate_topology_stationary_candidates,
)


points = benzene_electrophile_starting_points()
energy = benzene_electrophile_topology_energy(points["saddle_guess"])
diagnostics = evaluate_topology_stationary_candidates()

print(energy)
print(diagnostics["saddle_guess"]["classification"])
print(diagnostics["saddle_guess"]["hessian_eigenvalues"])
```

The model is a topology bridge from toy surfaces toward later package-based
Project 2 workflows with xTB, Psi4, ORCA, or PySCF. It is synthetic and
educational; it is not ab initio chemistry and does not reproduce experimental
or high-level quantum chemistry results.

## Diagnostics And Synthetic ML Dataset Generation

```python
from saddlepoint_whitebox.dataset_generation import generate_topology_dataset
from saddlepoint_whitebox.diagnostics import (
    diagnose_saddle_candidate,
    scan_along_reaction_coordinate,
)
from saddlepoint_whitebox.surfaces import quadratic_first_order_saddle


diagnostic = diagnose_saddle_candidate(
    "quadratic saddle",
    quadratic_first_order_saddle,
    [0.0, 0.0],
)
scan = scan_along_reaction_coordinate(
    quadratic_first_order_saddle,
    diagnostic.coordinates,
    diagnostic.reaction_coordinate,
    step_size=0.05,
    points_each_side=3,
)

labels = generate_topology_dataset(
    "data/topology_labels.jsonl",
    "data/topology_labels_summary.csv",
    samples=30,
)

print(diagnostic.classification, diagnostic.eigenvalues)
print(scan)
print(len(labels))
```

## Topology Calibration And Validation

After creating a synthetic PES, the project should still check whether its
local topology matches the intended mechanistic story. The validation layer
relaxes the pi and rim guesses as local basins, runs eigenvector following from
the saddle guess, and reports gradients, Hessian eigenvalues, classifications,
and reaction-coordinate directions.

Run the default validation report with:

```powershell
$env:UV_CACHE_DIR='.uv-cache'; $env:UV_PYTHON_INSTALL_DIR='.uv-python'; uv run python scripts/validate_topology_model.py
```

The desired synthetic result is:

- `pi_complex_guess`: minimum-like.
- `rim_complex_guess`: intermediate/minimum-like, or at least not a higher-order saddle.
- `saddle_guess`: first-order saddle with exactly one negative Hessian eigenvalue.

This validates only the reduced-order synthetic topology. It does not reproduce
real molecular chemistry, ab initio calculations, or experimental data.

Run tests with:

```powershell
$env:UV_CACHE_DIR='.uv-cache'; $env:UV_PYTHON_INSTALL_DIR='.uv-python'; uv run python -m unittest discover -s tests
```
