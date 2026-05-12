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

## Explicit EVF Step Calculation

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
print(target_eigenvalue, target_mode)
```

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

This is a synthetic reduced-coordinate PES topology model inspired by
benzene-electrophile approach, pi-complex character, and Wheland-like regions.
It is useful for testing whether the engine distinguishes minimum-like and
saddle-like topology. It is not a substitute for ORCA, Gaussian, Psi4, xTB,
CCSD(T), or any professional quantum chemistry transition-state workflow.

Run tests with:

```powershell
$env:UV_CACHE_DIR='.uv-cache'; $env:UV_PYTHON_INSTALL_DIR='.uv-python'; uv run python -m unittest discover -s tests
```
