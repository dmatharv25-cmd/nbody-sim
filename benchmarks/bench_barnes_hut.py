"""Barnes-Hut: accuracy vs theta, and how the amount of work scales with N."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbody.forces import accelerations
from nbody.barnes_hut import accelerations_barnes_hut as bh

rng = np.random.default_rng(0)

# --- 1) accuracy vs theta (N = 1000) ---
n = 1000
pos, m = rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)
ref = accelerations(pos, m)
nrm = np.linalg.norm(ref, axis=1)
thetas = [0.2, 0.3, 0.5, 0.7, 1.0]
med, mx, frac = [], [], []
print(f"N={n}: direct sum needs {n * (n - 1)} pair interactions")
print(f"{'theta':>6} {'median rel err':>15} {'max rel err':>12} {'interactions':>13} {'% of direct':>12}")
for th in thetas:
    a, c = bh(pos, m, theta=th, return_stats=True)
    rel = np.linalg.norm(a - ref, axis=1) / nrm
    med.append(np.median(rel)); mx.append(rel.max()); frac.append(100 * c / (n * (n - 1)))
    print(f"{th:>6.1f} {med[-1]:>15.2e} {mx[-1]:>12.2e} {c:>13d} {frac[-1]:>11.1f}%")

# --- 2) work vs N at theta = 0.5 (counts are hardware-independent) ---
Ns = [250, 500, 1000, 2000]
counts = []
print(f"\n{'N':>6} {'BH interactions':>16} {'direct N(N-1)':>14} {'BH per particle':>16}")
for k in Ns:
    p = rng.normal(size=(k, 3))
    _, c = bh(p, np.ones(k), theta=0.5, return_stats=True)
    counts.append(c)
    print(f"{k:>6} {c:>16d} {k * (k - 1):>14d} {c / k:>16.1f}")
slope = np.polyfit(np.log(Ns), np.log(counts), 1)[0]
print(f"\nBarnes-Hut work scaling exponent over this range: {slope:.2f}  (direct sum = 2.00)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].loglog(thetas, med, "o-", label="median"); ax[0].loglog(thetas, mx, "s--", label="worst particle")
ax[0].set(xlabel="opening angle theta", ylabel="relative force error", title=f"Accuracy vs theta (N={n})")
ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].loglog(Ns, counts, "o-", label="Barnes-Hut (theta=0.5)")
ax[1].loglog(Ns, [k * (k - 1) for k in Ns], "s--", label="direct sum")
ax[1].set(xlabel="N", ylabel="pair interactions", title="Work vs N")
ax[1].legend(); ax[1].grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig("results/barnes_hut.png", dpi=140)
