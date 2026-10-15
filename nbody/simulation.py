import numpy as np
from functools import partial
from .forces import accelerations
from .integrators import INTEGRATORS
from .diagnostics import total_energy, angular_momentum, momentum


def simulate(pos, vel, masses, dt, n_steps, method="verlet", G=1.0, softening=0.0,
             record_every=1, acc_fn=None):
    """Run a simulation. Returns dict of time series (t, pos, energy, L, p).

    acc_fn: optional force function acc_fn(pos, masses) -> accelerations. Defaults to the
    NumPy direct sum. Pass e.g. a Numba or Barnes-Hut kernel (wrapped with functools.partial
    to fix G, softening, theta). Energy is always measured with the exact direct-sum potential,
    so it is an honest check of whatever force approximation is used.
    """
    step = INTEGRATORS[method]
    if acc_fn is None:
        acc_fn = partial(accelerations, G=G, softening=softening)
    raw_fn = acc_fn

    # Velocity Verlet evaluates the force at the end of one step and again at the start of
    # the next, at the same positions. Cache the last result so each force is computed once.
    cache = {"pos": None, "acc": None}

    def cached_acc(p, m):
        if cache["pos"] is p:
            return cache["acc"]
        a = raw_fn(p, m)
        cache["pos"], cache["acc"] = p, a
        return a

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
        pos, vel = step(pos, vel, masses, dt, cached_acc)
        if i % record_every == 0:
            record(i)
    return {"t": np.array(ts), "pos": np.array(traj), "energy": np.array(E),
            "L": np.array(L), "p": np.array(P), "final_pos": pos, "final_vel": vel}
