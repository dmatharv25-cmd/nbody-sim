"""Gravitational force model (Newtonian, G configurable, optional softening).

Vectorized O(N^2) pairwise computation with NumPy.
Shapes: pos (N, D), masses (N,), returns (N, D).
"""
import numpy as np


def accelerations(pos, masses, G=1.0, softening=0.0):
    # diff[i, j] = pos[j] - pos[i]
    diff = pos[None, :, :] - pos[:, None, :]
    dist2 = np.sum(diff ** 2, axis=-1) + softening ** 2
    np.fill_diagonal(dist2, np.inf)          # no self-force
    inv_d3 = dist2 ** -1.5
    return G * np.einsum("ij,ijk,j->ik", inv_d3, diff, masses)


def potential_energy(pos, masses, G=1.0, softening=0.0):
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + softening ** 2)
    iu = np.triu_indices(len(masses), k=1)
    return -G * np.sum(masses[iu[0]] * masses[iu[1]] / dist[iu])
