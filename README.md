# N-Body Simulator with Validated Numerical Integrators

A from-scratch gravitational N-body simulator in Python/NumPy. The focus is **numerical correctness**:
every integrator is validated against analytical results (Kepler orbits, conservation laws,
published periodic solutions) and its order of accuracy is measured empirically.

## What's implemented
- Vectorized O(N^2) Newtonian gravity (optional softening)
- Integrators written from scratch: **Euler**, **Velocity Verlet** (symplectic), **RK4**
- Diagnostics: total energy, linear and angular momentum
- Initial conditions with known answers: Keplerian two-body orbit, Chenciner-Montgomery figure-eight
- 12 pytest tests covering both the physics and the numerics

## Results
| Method | Order (measured) | Rel. energy error after 20 orbits (200 steps/orbit, e=0.6) |
|---|---|---|
| Euler | ~0.8 (approaches 1 as dt shrinks) | ~1.09 (energy not conserved) |
| Velocity Verlet | 2.00 | ~7e-3 (bounded, does not grow) |
| RK4 | ~4.1 | ~1e-4 (slow secular drift) |

![energy](results/energy_comparison.png)
![convergence](results/convergence.png)

**Takeaways:** Verlet is symplectic, so its energy error stays bounded over long runs. RK4 is far more
accurate per step but drifts slowly over very long runs. Euler is unusable for orbits.
Euler's measured order is below 1 because the tested step sizes are too coarse to be in the asymptotic regime.

## Run it

    pip install -r requirements.txt
    pytest -q
    python experiments/energy_comparison.py
    python experiments/convergence.py

## Roadmap
- [x] Numba acceleration and benchmarks vs. pure NumPy
- [x] Barnes-Hut tree (O(N log N)) with accuracy-vs-speed tradeoff
- [ ] Adaptive time stepping
- [ ] Sun-Earth-Moon validation against NASA JPL ephemeris data
- [ ] Streamlit UI / animations


## Performance (force calculation, 3D, 12 logical cores)
All implementations agree to floating-point roundoff (max difference ~1e-13).

| N | NumPy (s) | Numba serial (s) | Numba parallel (s) | Serial speedup | Parallel speedup |
|---|---|---|---|---|---|
| 200 | 0.00549 | 0.00097 | 0.00053 | 5.7x | 10.4x |
| 400 | 0.02620 | 0.00335 | 0.00108 | 7.8x | 24.2x |
| 800 | 0.06882 | 0.00744 | 0.00243 | 9.2x | 28.3x |
| 1600 | 0.29793 | 0.02975 | 0.00906 | 10.0x | 32.9x |
| 3200 | - | 0.11988 | 0.03581 | - | - |

Speedups are relative to the vectorized NumPy version. Parallel scaling is about 3.3x over serial, not 12x,
limited by physical core count, thread overhead and memory bandwidth. All versions are still O(N^2),
which is what Barnes-Hut addresses next.


## Barnes-Hut tree code
A quadtree/octree implementation (`nbody/barnes_hut.py`) that replaces distant groups of particles with
their centre of mass when (node width) / (distance) < theta. With theta = 0 it reproduces the direct sum
to roundoff (tested in 2D and 3D).

**Accuracy vs. cost** (N = 1000, 3D Gaussian cloud, relative force error vs. direct sum):

| theta | median error | worst-particle error | interactions vs. direct |
|---|---|---|---|
| 0.2 | 2.4e-4 | 1.7e-3 | 75.7% |
| 0.3 | 8.7e-4 | 7.0e-3 | 54.7% |
| 0.5 | 3.5e-3 | 4.3e-2 | 29.2% |
| 0.7 | 8.6e-3 | 1.5e-1 | 17.3% |
| 1.0 | 2.4e-2 | 2.7e-1 | 9.7% |

**Work vs. N** (theta = 0.5): pair interactions grow with an empirical exponent of 1.54 over N = 250 to 2000
(direct sum: 2.00). At N = 2000 the tree does 20% of the direct-sum work. This range is too small to confirm
the asymptotic O(N log N) behaviour; larger N is needed. The tree is pure Python, so its wall-clock time is
not competitive with the compiled Numba direct sum yet. The benchmark counts interactions, which are
hardware-independent.

![barnes-hut](results/barnes_hut.png)


## Does the approximation preserve the physics? (Plummer cluster)
`simulate()` accepts any force function, so the same cluster (N = 200, 3D Plummer sphere in virial
equilibrium, softening 0.05, Velocity Verlet, dt = 0.01, 300 steps) was run with three force models.
Energy is always measured with the exact direct-sum potential.

| Force model | Wall time | Max abs(dE/E) | Max total momentum |
|---|---|---|---|
| Direct sum (NumPy) | 1.4 s | 7.4e-6 | 2.3e-17 |
| Barnes-Hut, theta = 0.5 | 107 s | 2.7e-4 | 1.5e-4 |
| Barnes-Hut, theta = 1.0 | 62 s | 9.4e-4 | 1.9e-3 |

- Direct summation conserves momentum to machine precision. Barnes-Hut does not, which is consistent
  with tree forces not being exactly equal and opposite between particle pairs (Newton's third law).
- Barnes-Hut energy error is about 37x (theta = 0.5) and 127x (theta = 1.0) larger than direct summation,
  though still below 0.1% over this run.
- At N = 200 the pure-Python tree is about 76x slower in wall-clock time than the vectorized direct sum.
  A compiled tree and larger N are needed before the O(N log N) algorithm pays off in practice.

![cluster](results/cluster_comparison.png)
