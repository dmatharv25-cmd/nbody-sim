"""Numba-compiled O(N^2) force kernels.

accelerations_numba:          serial compiled loops
accelerations_numba_parallel: outer loop split across CPU cores (prange).
Each i writes only to acc[i], so there are no race conditions.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def accelerations_numba(pos, masses, G=1.0, softening=0.0):
    n, d = pos.shape
    acc = np.zeros((n, d))
    s2 = softening * softening
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r2 = s2
            for k in range(d):
                diff = pos[j, k] - pos[i, k]
                r2 += diff * diff
            f = G * masses[j] / (r2 * np.sqrt(r2))
            for k in range(d):
                acc[i, k] += f * (pos[j, k] - pos[i, k])
    return acc


@njit(parallel=True, cache=True)
def accelerations_numba_parallel(pos, masses, G=1.0, softening=0.0):
    n, d = pos.shape
    acc = np.zeros((n, d))
    s2 = softening * softening
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            r2 = s2
            for k in range(d):
                diff = pos[j, k] - pos[i, k]
                r2 += diff * diff
            f = G * masses[j] / (r2 * np.sqrt(r2))
            for k in range(d):
                acc[i, k] += f * (pos[j, k] - pos[i, k])
    return acc
