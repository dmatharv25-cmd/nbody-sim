"""Initial conditions with known analytical behaviour."""
import numpy as np


def two_body_orbit(m1=1.0, m2=1.0, a=1.0, e=0.5, G=1.0):
    """Two bodies on a Keplerian ellipse, starting at periapsis, in the
    centre-of-mass frame (2D).

    Returns pos, vel, masses, period.  Analytical results:
      period T = 2*pi*sqrt(a^3 / (G*M)),  energy E = -G*m1*m2 / (2a),
      after exactly one period the bodies return to their initial state.
    """
    M = m1 + m2
    r_p = a * (1 - e)
    v_p = np.sqrt(G * M * (1 + e) / (a * (1 - e)))   # relative speed at periapsis
    masses = np.array([m1, m2])
    pos = np.array([[-m2 / M * r_p, 0.0], [m1 / M * r_p, 0.0]])
    vel = np.array([[0.0, -m2 / M * v_p], [0.0, m1 / M * v_p]])
    period = 2 * np.pi * np.sqrt(a ** 3 / (G * M))
    return pos, vel, masses, period


def figure_eight():
    """Chenciner-Montgomery figure-eight three-body orbit (G=1, equal masses).
    Published initial conditions; period ~ 6.32591398."""
    p1 = np.array([0.97000436, -0.24308753])
    v3 = np.array([-0.93240737, -0.86473146])
    pos = np.array([p1, -p1, [0.0, 0.0]])
    vel = np.array([-v3 / 2, -v3 / 2, v3])
    return pos, vel, np.ones(3), 6.32591398


def plummer_sphere(n, a=1.0, total_mass=1.0, G=1.0, seed=0):
    """Random N-body realisation of a Plummer sphere in approximate virial equilibrium (3D).

    Positions follow the Plummer density profile; speeds are drawn from its isotropic
    distribution function by rejection sampling (Aarseth, Henon & Wielen 1974).
    Centre of mass position and velocity are shifted to zero.
    """
    rng = np.random.default_rng(seed)
    masses = np.full(n, total_mass / n)

    # radii from the inverse cumulative mass function
    X = rng.uniform(1e-6, 1.0 - 1e-6, n)
    r = a / np.sqrt(X ** (-2.0 / 3.0) - 1.0)
    cos_t = rng.uniform(-1, 1, n); phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    pos = r[:, None] * np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)

    # speeds: q = v / v_escape drawn from g(q) = q^2 (1 - q^2)^(7/2)
    q = np.empty(n)
    for i in range(n):
        while True:
            x, y = rng.uniform(0, 1), rng.uniform(0, 0.1)
            if y < x ** 2 * (1 - x ** 2) ** 3.5:
                q[i] = x
                break
    v_esc = np.sqrt(2.0 * G * total_mass) * (r ** 2 + a ** 2) ** -0.25
    speed = q * v_esc
    cos_t = rng.uniform(-1, 1, n); phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    vel = speed[:, None] * np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)

    pos -= (masses[:, None] * pos).sum(axis=0) / masses.sum()
    vel -= (masses[:, None] * vel).sum(axis=0) / masses.sum()
    return pos, vel, masses
