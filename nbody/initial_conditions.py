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
