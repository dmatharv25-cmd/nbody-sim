import numpy as np
from nbody.forces import accelerations
from nbody.barnes_hut import accelerations_barnes_hut


def _system(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.uniform(0.5, 2, n)


def test_theta_zero_equals_direct_sum():
    pos, m = _system()
    ref = accelerations(pos, m, softening=0.02)
    out = accelerations_barnes_hut(pos, m, softening=0.02, theta=0.0)
    assert np.allclose(out, ref, rtol=1e-9, atol=1e-11)


def test_works_in_2d():
    pos, m = _system(d=2)
    assert np.allclose(accelerations_barnes_hut(pos, m, theta=0.0), accelerations(pos, m), rtol=1e-9, atol=1e-11)


def test_error_shrinks_as_theta_shrinks():
    pos, m = _system()
    ref = accelerations(pos, m)
    nrm = np.linalg.norm(ref, axis=1)
    errs = [np.median(np.linalg.norm(accelerations_barnes_hut(pos, m, theta=t) - ref, axis=1) / nrm)
            for t in (1.0, 0.5, 0.25)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] < 0.02


def test_fewer_interactions_than_direct():
    pos, m = _system(400)
    _, count = accelerations_barnes_hut(pos, m, theta=0.5, return_stats=True)
    assert count < 0.5 * 400 * 399


def test_coincident_particles_do_not_crash():
    pos = np.vstack([np.zeros((3, 3)), np.random.default_rng(3).normal(size=(5, 3))])
    out = accelerations_barnes_hut(pos, np.ones(8), softening=0.1, theta=0.5)
    assert np.all(np.isfinite(out))
