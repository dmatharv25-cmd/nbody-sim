"""Conserved quantities used to validate a simulation."""
import numpy as np
from .forces import potential_energy


def kinetic_energy(vel, masses):
    return 0.5 * np.sum(masses[:, None] * vel ** 2)


def total_energy(pos, vel, masses, G=1.0, softening=0.0):
    return kinetic_energy(vel, masses) + potential_energy(pos, masses, G, softening)


def momentum(vel, masses):
    return np.sum(masses[:, None] * vel, axis=0)


def center_of_mass(pos, masses):
    return np.sum(masses[:, None] * pos, axis=0) / np.sum(masses)


def angular_momentum(pos, vel, masses):
    """z-component of total angular momentum (2D systems)."""
    return np.sum(masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
