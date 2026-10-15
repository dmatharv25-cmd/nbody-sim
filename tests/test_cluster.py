import numpy as np
from functools import partial
from nbody import (simulate, accelerations, two_body_orbit, plummer_sphere,
                   kinetic_energy, potential_energy)


def test_plummer_is_near_virial_equilibrium_and_centred():
    pos, vel, m = plummer_sphere(300, seed=1)
    ratio = 2 * kinetic_energy(vel, m) / abs(potential_energy(pos, m))
    assert 0.8 < ratio < 1.2
    assert np.allclose((m[:, None] * pos).sum(axis=0), 0, atol=1e-12)
    assert np.allclose((m[:, None] * vel).sum(axis=0), 0, atol=1e-12)


def test_simulate_accepts_custom_force_function():
    pos, vel, m, T = two_body_orbit(e=0.5)
    default = simulate(pos, vel, m, T / 300, 300, "verlet")
    custom = simulate(pos, vel, m, T / 300, 300, "verlet", acc_fn=partial(accelerations, G=1.0))
    assert np.array_equal(default["final_pos"], custom["final_pos"])


def test_force_cache_calls_force_once_per_verlet_step():
    pos, vel, m, T = two_body_orbit(e=0.5)
    calls = []
    def counting(p, mm):
        calls.append(1)
        return accelerations(p, mm)
    simulate(pos, vel, m, T / 100, 100, "verlet", acc_fn=counting)
    assert len(calls) == 101        # 1 initial + 1 per step
