"""Baseline timing: pure-Python loops vs. vectorized NumPy force calculation."""
import sys; sys.path.insert(0, ".")
import time
import numpy as np
from nbody.forces import accelerations
from nbody.forces_naive import accelerations_naive


def best_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


rng = np.random.default_rng(0)

# 1) correctness: both implementations must agree
pos, m = rng.normal(size=(100, 3)), rng.uniform(0.5, 2, 100)
err = np.max(np.abs(accelerations(pos, m) - accelerations_naive(pos, m)))
print(f"max difference naive vs numpy: {err:.2e}\n")

# 2) speed comparison at small N
print(f"{'N':>6} {'naive (s)':>12} {'numpy (s)':>12} {'speedup':>9}")
for n in (50, 100, 200, 400):
    pos, m = rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)
    t_naive = best_time(lambda: accelerations_naive(pos, m), 2)
    t_np = best_time(lambda: accelerations(pos, m), 5)
    print(f"{n:>6} {t_naive:>12.5f} {t_np:>12.6f} {t_naive / t_np:>8.0f}x")

# 3) how numpy version scales with N (expect slope ~2 on log-log)
print(f"\n{'N':>6} {'numpy (s)':>12}")
Ns, ts = [200, 400, 800, 1600], []
for n in Ns:
    pos, m = rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)
    t = best_time(lambda: accelerations(pos, m), 3)
    ts.append(t)
    print(f"{n:>6} {t:>12.5f}")
print(f"\nmeasured scaling exponent: {np.polyfit(np.log(Ns), np.log(ts), 1)[0]:.2f}  (2.0 = O(N^2))")
