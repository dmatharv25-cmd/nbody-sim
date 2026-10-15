import numpy as np
import pytest
from nbody import (accelerations, simulate, two_body_orbit, figure_eight, total_energy)


def test_newton_third_law_total_force_zero():
    rng = np.random.default_rng(0)
    pos, m = rng.normal(size=(20, 3)), rng.uniform(0.5, 2, 20)
    net_force = np.sum(m[:, None] * accelerations(pos, m), axis=0)
    assert np.allclose(net_force, 0, atol=1e-12)


def test_two_body_acceleration_matches_analytic():
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    m = np.array([3.0, 5.0])
    a = accelerations(pos, m, G=1.0)
    assert np.allclose(a[0], [5.0 / 4.0, 0.0])     # G*m2/r^2
    assert np.allclose(a[1], [-3.0 / 4.0, 0.0])


def test_two_body_initial_energy_matches_kepler():
    m1, m2, a = 1.0, 2.0, 1.5
    pos, vel, m, _ = two_body_orbit(m1, m2, a, e=0.6)
    assert total_energy(pos, vel, m) == pytest.approx(-m1 * m2 / (2 * a), rel=1e-12)


@pytest.mark.parametrize("method,n_steps,tol", [("rk4", 2000, 1e-6), ("verlet", 20000, 1e-3)])
def test_orbit_returns_after_one_period(method, n_steps, tol):
    pos, vel, m, T = two_body_orbit(e=0.5)
    res = simulate(pos, vel, m, T / n_steps, n_steps, method)
    assert np.max(np.abs(res["final_pos"] - pos)) < tol


def test_verlet_energy_bounded_over_many_orbits():
    pos, vel, m, T = two_body_orbit(e=0.5)
    res = simulate(pos, vel, m, T / 500, 500 * 50, "verlet", record_every=50)
    drift = np.abs((res["energy"] - res["energy"][0]) / res["energy"][0])
    assert drift.max() < 1e-2


def test_euler_energy_drifts_more_than_verlet():
    pos, vel, m, T = two_body_orbit(e=0.5)
    n = 500 * 10
    e_euler = simulate(pos, vel, m, T / 500, n, "euler")["energy"]
    e_verlet = simulate(pos, vel, m, T / 500, n, "verlet")["energy"]
    assert abs(e_euler[-1] - e_euler[0]) > 100 * abs(e_verlet[-1] - e_verlet[0])


def test_momentum_conserved():
    pos, vel, m, T = figure_eight()
    res = simulate(pos, vel, m, T / 2000, 2000, "rk4")
    assert res["p"].max() < 1e-10


def test_figure_eight_is_periodic():
    pos, vel, m, T = figure_eight()
    res = simulate(pos, vel, m, T / 4000, 4000, "rk4")
    assert np.max(np.abs(res["final_pos"] - pos)) < 1e-4


@pytest.mark.parametrize("method,order", [("euler", 1), ("verlet", 2), ("rk4", 4)])
def test_convergence_order(method, order):
    pos, vel, m, T = two_body_orbit(e=0.3)
    errs = []
    for n in (400, 800):
        res = simulate(pos, vel, m, T / n, n, method)
        errs.append(np.max(np.abs(res["final_pos"] - pos)))
    assert np.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.5)
