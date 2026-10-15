import numpy as np
from nbody.forces import accelerations
from nbody.barnes_hut import accelerations_barnes_hut
from nbody.barnes_hut_numba import accelerations_barnes_hut_numba as bh_nb


def _system(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.uniform(0.5, 2, n)


def test_theta_zero_equals_direct_sum():
    pos, m = _system()
    assert np.allclose(bh_nb(pos, m, softening=0.02, theta=0.0),
                       accelerations(pos, m, softening=0.02), rtol=1e-9, atol=1e-11)


def test_matches_pure_python_tree_exactly_in_work_and_forces():
    pos, m = _system()
    for theta in (0.3, 0.5, 1.0):
        a_py, c_py = accelerations_barnes_hut(pos, m, softening=0.02, theta=theta, return_stats=True)
        a_nb, c_nb = bh_nb(pos, m, softening=0.02, theta=theta, return_stats=True)
        assert c_py == c_nb
        assert np.allclose(a_py, a_nb, rtol=1e-9, atol=1e-11)


def test_coincident_and_tiny_systems():
    rng = np.random.default_rng(3)
    pos = np.vstack([np.zeros((4, 3)), rng.normal(size=(6, 3))])
    out = bh_nb(pos, np.ones(10), softening=0.1, theta=0.5)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, accelerations_barnes_hut(pos, np.ones(10), softening=0.1, theta=0.5))
    p, m = _system(2)
    assert np.allclose(bh_nb(p, m, theta=0.0), accelerations(p, m))
    assert np.all(bh_nb(p[:1], m[:1]) == 0.0)


def test_rejects_2d_input():
    import pytest
    with pytest.raises(ValueError):
        bh_nb(np.zeros((5, 2)), np.ones(5))
