import numpy as np
from nbody.forces import accelerations
from nbody.forces_numba import accelerations_numba, accelerations_numba_parallel


def test_numba_matches_numpy():
    rng = np.random.default_rng(1)
    pos, m = rng.normal(size=(60, 3)), rng.uniform(0.5, 2, 60)
    ref = accelerations(pos, m, G=1.3, softening=0.05)
    assert np.allclose(accelerations_numba(pos, m, 1.3, 0.05), ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(accelerations_numba_parallel(pos, m, 1.3, 0.05), ref, rtol=1e-10, atol=1e-12)


def test_numba_works_in_2d():
    rng = np.random.default_rng(2)
    pos, m = rng.normal(size=(30, 2)), rng.uniform(0.5, 2, 30)
    assert np.allclose(accelerations_numba(pos, m), accelerations(pos, m), rtol=1e-10, atol=1e-12)
