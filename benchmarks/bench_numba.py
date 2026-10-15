"""NumPy vs Numba (serial) vs Numba (parallel) force calculation."""
import sys; sys.path.insert(0, ".")
import time
import numpy as np
from nbody.forces import accelerations
from nbody.forces_numba import accelerations_numba, accelerations_numba_parallel


def best_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


rng = np.random.default_rng(0)

# warm-up: first call triggers JIT compilation, so keep it out of the timings
w_pos, w_m = rng.normal(size=(10, 3)), np.ones(10)
t0 = time.perf_counter(); accelerations_numba(w_pos, w_m); accelerations_numba_parallel(w_pos, w_m)
print(f"JIT warm-up (compile or cache load): {time.perf_counter() - t0:.2f} s\n")

pos, m = rng.normal(size=(500, 3)), rng.uniform(0.5, 2, 500)
ref = accelerations(pos, m)
print(f"max diff numba vs numpy:          {np.max(np.abs(accelerations_numba(pos, m) - ref)):.2e}")
print(f"max diff numba-parallel vs numpy: {np.max(np.abs(accelerations_numba_parallel(pos, m) - ref)):.2e}\n")

print(f"{'N':>6} {'numpy (s)':>11} {'numba (s)':>11} {'parallel (s)':>13} {'numba/numpy':>12} {'par/numpy':>10}")
for n in (200, 400, 800, 1600, 3200):
    pos, m = rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)
    t_ser = best_time(lambda: accelerations_numba(pos, m), 5)
    t_par = best_time(lambda: accelerations_numba_parallel(pos, m), 5)
    if n <= 1600:   # NumPy builds N*N*3 temporaries; skip the largest size to save memory
        t_np = best_time(lambda: accelerations(pos, m), 3)
        print(f"{n:>6} {t_np:>11.5f} {t_ser:>11.5f} {t_par:>13.5f} {t_np / t_ser:>11.1f}x {t_np / t_par:>9.1f}x")
    else:
        print(f"{n:>6} {'-':>11} {t_ser:>11.5f} {t_par:>13.5f} {'-':>12} {'-':>10}")

import os
print(f"\nCPU cores available: {os.cpu_count()}")
