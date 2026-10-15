"""Empirical order of accuracy: position error after one full period vs. step size."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbody import simulate, two_body_orbit

pos, vel, m, T = two_body_orbit(e=0.3)
Ns = np.array([100, 200, 400, 800, 1600, 3200])
plt.figure(figsize=(6, 4.5))
for method in ("euler", "verlet", "rk4"):
    errs = np.array([np.max(np.abs(simulate(pos, vel, m, T / n, n, method)["final_pos"] - pos)) for n in Ns])
    slope = np.polyfit(np.log(T / Ns), np.log(errs), 1)[0]
    print(f"{method:7s} measured order = {slope:.2f}")
    plt.loglog(T / Ns, errs, "o-", label=f"{method} (slope {slope:.2f})")
plt.xlabel("time step dt"); plt.ylabel("position error after 1 period")
plt.title("Convergence of integrators"); plt.legend(); plt.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig("results/convergence.png", dpi=140)
