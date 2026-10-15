import numpy as np
from functools import partial
from .forces import accelerations
from .integrators import INTEGRATORS
from .diagnostics import total_energy, angular_momentum, momentum


def simulate(pos, vel, masses, dt, n_steps, method="verlet", G=1.0, softening=0.0, record_every=1):
    """Run a simulation. Returns dict of time series (t, pos, energy, L, p)."""
    step = INTEGRATORS[method]
    acc_fn = partial(accelerations, G=G, softening=softening)
    pos, vel = pos.copy(), vel.copy()
    ts, traj, E, L, P = [], [], [], [], []

    def record(i):
        ts.append(i * dt)
        traj.append(pos.copy())
        E.append(total_energy(pos, vel, masses, G, softening))
        L.append(angular_momentum(pos, vel, masses))
        P.append(np.linalg.norm(momentum(vel, masses)))

    record(0)
    for i in range(1, n_steps + 1):
        pos, vel = step(pos, vel, masses, dt, acc_fn)
        if i % record_every == 0:
            record(i)
    return {"t": np.array(ts), "pos": np.array(traj), "energy": np.array(E),
            "L": np.array(L), "p": np.array(P), "final_pos": pos, "final_vel": vel}
