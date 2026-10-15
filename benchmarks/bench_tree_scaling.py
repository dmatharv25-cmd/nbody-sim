"""Wall-clock scaling: Numba direct sum (parallel) vs Numba Barnes-Hut (parallel), theta = 0.5.

Optional argument: maximum N (default 128000), e.g.  python benchmarks/bench_tree_scaling.py 16000
"""
import sys; sys.path.insert(0, ".")
import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbody.forces_numba import accelerations_numba_parallel as direct
from nbody.barnes_hut_numba import accelerations_barnes_hut_numba as bh
from nbody.barnes_hut import accelerations_barnes_hut as bh_python

MAX_N = int(sys.argv[1]) if len(sys.argv) > 1 else 128000
DIRECT_MAX = 32000          # direct sum above this takes too long to be worth timing
THETA = 0.5


def best_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


rng = np.random.default_rng(0)
w = rng.normal(size=(50, 3)); wm = np.ones(50)
direct(w, wm); bh(w, wm, theta=THETA)                       # compile outside the timings

# --- accuracy: compiled tree vs compiled direct sum ---
n = min(2000, MAX_N)
pos, m = rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)
ref = direct(pos, m); a, c = bh(pos, m, theta=THETA, return_stats=True)
rel = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
print(f"accuracy at N={n}, theta={THETA}: median rel error {np.median(rel):.2e}, "
      f"interactions {100 * c / (n * (n - 1)):.1f}% of direct\n")

# --- compiled tree vs pure-Python tree (same algorithm) ---
for k in [x for x in (1000, 2000) if x <= MAX_N]:
    p, mm = rng.normal(size=(k, 3)), np.ones(k)
    t_py = best_time(lambda: bh_python(p, mm, theta=THETA), 1)
    t_nb = best_time(lambda: bh(p, mm, theta=THETA), 3)
    print(f"N={k}: pure-Python tree {t_py:.3f} s | compiled tree {t_nb:.4f} s | {t_py / t_nb:.0f}x faster")

# --- scaling ---
print(f"\n{'N':>8} {'direct (s)':>11} {'tree (s)':>10} {'tree/direct':>12} {'interactions':>13} {'tree time / (N log2 N) [ns]':>28}")
Ns = [n for n in (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000) if n <= MAX_N]
t_dir, t_bh, counts, cross = [], [], [], None
for n in Ns:
    p, mm = rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)
    reps = 3 if n <= 4000 else 2
    tb = best_time(lambda: bh(p, mm, theta=THETA), reps)
    _, c = bh(p, mm, theta=THETA, return_stats=True)
    td = best_time(lambda: direct(p, mm), reps) if n <= DIRECT_MAX else float("nan")
    t_bh.append(tb); t_dir.append(td); counts.append(c)
    if cross is None and td > tb:
        cross = n
    ratio = f"{td / tb:>11.1f}x" if td == td else f"{'-':>12}"
    print(f"{n:>8} {td:>11.4f} {tb:>10.4f} {ratio} {c:>13d} {1e9 * tb / (n * np.log2(n)):>28.1f}")

big = [i for i, n in enumerate(Ns) if n >= 4000]
if len(big) >= 3:
    lx = np.log(np.array(Ns)[big])
    print(f"\nTree wall-time exponent (N >= 4000):      {np.polyfit(lx, np.log(np.array(t_bh)[big]), 1)[0]:.2f}")
    print(f"Tree interaction-count exponent (N >= 4000): {np.polyfit(lx, np.log(np.array(counts)[big]), 1)[0]:.2f}")
    d = [i for i in big if t_dir[i] == t_dir[i]]
    if len(d) >= 3:
        print(f"Direct-sum wall-time exponent (N >= 4000):   {np.polyfit(np.log(np.array(Ns)[d]), np.log(np.array(t_dir)[d]), 1)[0]:.2f}")
print(f"First N where the tree beats direct sum: {cross if cross else 'not reached in this range'}")

Narr = np.array(Ns, dtype=float)
plt.figure(figsize=(6.5, 4.8))
plt.loglog(Ns, t_bh, "o-", label="Barnes-Hut (compiled, parallel)")
plt.loglog(Ns, t_dir, "s-", label="direct sum (compiled, parallel)")
plt.loglog(Narr, t_bh[0] * (Narr * np.log2(Narr)) / (Narr[0] * np.log2(Narr[0])), "k:", lw=1, label="N log N guide")
plt.loglog(Narr, t_dir[0] * (Narr / Narr[0]) ** 2, "k--", lw=1, alpha=.5, label="N^2 guide")
plt.xlabel("N"); plt.ylabel("time per force evaluation (s)"); plt.grid(alpha=.3, which="both")
plt.title("Wall-clock scaling (theta = 0.5)"); plt.legend(); plt.tight_layout()
plt.savefig("results/tree_scaling.png", dpi=140)
