"""Time integrators. Each step has the signature
    step(pos, vel, masses, dt, acc_fn) -> (new_pos, new_vel)
where acc_fn(pos, masses) returns accelerations.

Global order of accuracy: Euler 1, Velocity Verlet 2 (symplectic), RK4 4.
"""


def euler_step(pos, vel, masses, dt, acc_fn):
    a = acc_fn(pos, masses)
    return pos + dt * vel, vel + dt * a


def verlet_step(pos, vel, masses, dt, acc_fn):
    a = acc_fn(pos, masses)
    new_pos = pos + dt * vel + 0.5 * dt ** 2 * a
    new_a = acc_fn(new_pos, masses)
    new_vel = vel + 0.5 * dt * (a + new_a)
    return new_pos, new_vel


def rk4_step(pos, vel, masses, dt, acc_fn):
    k1v = acc_fn(pos, masses);                       k1x = vel
    k2v = acc_fn(pos + 0.5 * dt * k1x, masses);      k2x = vel + 0.5 * dt * k1v
    k3v = acc_fn(pos + 0.5 * dt * k2x, masses);      k3x = vel + 0.5 * dt * k2v
    k4v = acc_fn(pos + dt * k3x, masses);            k4x = vel + dt * k3v
    new_pos = pos + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_vel = vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return new_pos, new_vel


INTEGRATORS = {"euler": euler_step, "verlet": verlet_step, "rk4": rk4_step}
