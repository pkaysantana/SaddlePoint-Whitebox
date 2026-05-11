# SaddlePoint Whitebox

Small, inspectable tools for exploring potential energy surfaces.

The initial core is dependency-free and centered on:

- vector and matrix operations in `saddlepoint_whitebox.matrix`
- central-difference numerical differentiation in `saddlepoint_whitebox.calculus`
- Hessian eigenvalue checks for first-order saddle point classification

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

Run tests with:

```powershell
python -m unittest discover -s tests
```
