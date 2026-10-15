"""Plummer star cluster: does the Barnes-Hut approximation preserve the physics?

Same initial conditions, same integrator (Velocity Verlet), three force models.
Energy is always measured with the exact direct-sum potential.
"""
import sys; sys.path.insert(0, ".")
import time
from functools import partial
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbody import simulate, accelerations, plummer_sphere
from nbody.barnes_hut import accelerations_barnes_hut as bh

N, eps, dt, n_steps, rec = 200, 0.05, 0.01, 300, 10
pos, vel, m = plummer_sphere(N, seed=1)

models = {
    "direct sum": partial(accelerations, softening=eps),
    "Barnes-Hut theta=0.5": partial(bh, softening=eps, theta=0.5),
    "Barnes-Hut theta=1.0": partial(bh, softening=eps, theta=1.0),
}
print(f"Plummer cluster, N={N}, softening={eps}, dt={dt}, {n_steps} steps (t = {n_steps * dt:g})\n")
print(f"{'force model':<22} {'time (s)':>9} {'max |dE/E|':>11} {'final |dE/E|':>13} {'max |P|':>10}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, fn in models.items():
    t0 = time.perf_counter()
    res = simulate(pos, vel, m, dt, n_steps, "verlet", softening=eps, record_every=rec, acc_fn=fn)
    wall = time.perf_counter() - t0
    drift = np.abs((res["energy"] - res["energy"][0]) / res["energy"][0])
    print(f"{name:<22} {wall:>9.1f} {drift.max():>11.2e} {drift[-1]:>13.2e} {res['p'].max():>10.2e}")
    ax[0].semilogy(res["t"], drift + 1e-12, label=name)
    ax[1].semilogy(res["t"], res["p"] + 1e-18, label=name)
ax[0].set(xlabel="time", ylabel="|dE / E0|", title="Energy conservation"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set(xlabel="time", ylabel="|total momentum|", title="Momentum conservation"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/cluster_comparison.png", dpi=140)
