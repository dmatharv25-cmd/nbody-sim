"""Energy error of Euler / Verlet / RK4 over 20 orbits of an eccentric two-body system."""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbody import simulate, two_body_orbit

pos, vel, m, T = two_body_orbit(e=0.6)
steps_per_orbit, n_orbits = 200, 20
dt = T / steps_per_orbit

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for method in ("euler", "verlet", "rk4"):
    res = simulate(pos, vel, m, dt, steps_per_orbit * n_orbits, method, record_every=5)
    rel_err = np.abs((res["energy"] - res["energy"][0]) / res["energy"][0])
    ax[0].semilogy(res["t"] / T, rel_err + 1e-18, label=method)
    print(f"{method:7s} final relative energy error: {rel_err[-1]:.3e}")
    if method == "verlet":
        ax[1].plot(res["pos"][:, 0, 0], res["pos"][:, 0, 1], lw=.6, label="body 1")
        ax[1].plot(res["pos"][:, 1, 0], res["pos"][:, 1, 1], lw=.6, label="body 2")
ax[0].set(xlabel="orbits", ylabel="|dE / E0|", title=f"Energy error ({steps_per_orbit} steps/orbit, e=0.6)")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set(title="Verlet trajectory (20 orbits)", aspect="equal"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/energy_comparison.png", dpi=140)
