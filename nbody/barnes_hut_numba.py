"""Numba-compiled Barnes-Hut (3D only).

The tree is stored in flat arrays (no Python objects) so it can be built and traversed in compiled code:
  child[node, octant]  index of child node, or -1
  head[node]           first particle stored in a leaf (linked list via next_body), or -1
  next_body[i]         next particle in the same leaf; only chains at max depth (coincident particles)
Children are always created after their parent, so centre-of-mass moments are computed
bottom-up by looping over nodes in reverse creation order (no recursion).
Force evaluation is parallel over particles; each particle writes only its own result.
"""
import numpy as np
from numba import njit, prange

MAX_DEPTH = 40


@njit(cache=True)
def _octant(x, y, z, cx, cy, cz):
    c = 0
    if x >= cx:
        c += 1
    if y >= cy:
        c += 2
    if z >= cz:
        c += 4
    return c


@njit(cache=True)
def _init_child(center, half, parent, new, c):
    h = half[parent] / 2.0
    half[new] = h
    center[new, 0] = center[parent, 0] + (h if (c & 1) else -h)
    center[new, 1] = center[parent, 1] + (h if (c & 2) else -h)
    center[new, 2] = center[parent, 2] + (h if (c & 4) else -h)


@njit(cache=True)
def _build(pos, cap, root_center, root_half):
    """Returns (center, half, child, head, next_body, has_kids, count). count = -1 means capacity overflow."""
    n = pos.shape[0]
    center = np.zeros((cap, 3))
    half = np.zeros(cap)
    child = np.full((cap, 8), -1, np.int64)
    head = np.full(cap, -1, np.int64)
    has_kids = np.zeros(cap, np.bool_)
    next_body = np.full(n, -1, np.int64)
    center[0, 0] = root_center[0]
    center[0, 1] = root_center[1]
    center[0, 2] = root_center[2]
    half[0] = root_half
    count = 1
    for i in range(n):
        node = 0
        depth = 0
        while True:
            if has_kids[node]:
                c = _octant(pos[i, 0], pos[i, 1], pos[i, 2],
                            center[node, 0], center[node, 1], center[node, 2])
                nxt = child[node, c]
                if nxt == -1:
                    if count >= cap:
                        return center, half, child, head, next_body, has_kids, -1
                    nxt = count
                    count += 1
                    _init_child(center, half, node, nxt, c)
                    head[nxt] = i
                    child[node, c] = nxt
                    break
                node = nxt
                depth += 1
            else:
                if head[node] == -1:
                    head[node] = i
                    break
                if depth >= MAX_DEPTH:             # coincident particles: chain them in this leaf
                    next_body[i] = head[node]
                    head[node] = i
                    break
                b = head[node]                      # split: push the resident particle one level down
                head[node] = -1
                has_kids[node] = True
                c = _octant(pos[b, 0], pos[b, 1], pos[b, 2],
                            center[node, 0], center[node, 1], center[node, 2])
                if count >= cap:
                    return center, half, child, head, next_body, has_kids, -1
                nxt = count
                count += 1
                _init_child(center, half, node, nxt, c)
                head[nxt] = b
                child[node, c] = nxt
    return center, half, child, head, next_body, has_kids, count


@njit(cache=True)
def _moments(pos, masses, child, head, next_body, has_kids, count):
    mass = np.zeros(count)
    com = np.zeros((count, 3))
    for node in range(count - 1, -1, -1):
        if has_kids[node]:
            for c in range(8):
                ch = child[node, c]
                if ch >= 0:
                    mass[node] += mass[ch]
                    for k in range(3):
                        com[node, k] += mass[ch] * com[ch, k]
        else:
            j = head[node]
            while j != -1:
                mass[node] += masses[j]
                for k in range(3):
                    com[node, k] += masses[j] * pos[j, k]
                j = next_body[j]
        if mass[node] > 0.0:
            for k in range(3):
                com[node, k] /= mass[node]
    return mass, com


@njit(parallel=True, cache=True)
def _forces(pos, masses, half, child, head, next_body, has_kids, mass, com, G, s2, theta2):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    counts = np.zeros(n, np.int64)
    for i in prange(n):
        stack = np.empty(8 * (MAX_DEPTH + 2), np.int64)
        stack[0] = 0
        sp = 1
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        ax = 0.0
        ay = 0.0
        az = 0.0
        cnt = 0
        while sp > 0:
            sp -= 1
            node = stack[sp]
            if has_kids[node]:
                dx = com[node, 0] - xi
                dy = com[node, 1] - yi
                dz = com[node, 2] - zi
                r2 = dx * dx + dy * dy + dz * dz
                w = 2.0 * half[node]
                if w * w < theta2 * r2:            # far enough: whole node as one point mass
                    r2 += s2
                    f = G * mass[node] / (r2 * np.sqrt(r2))
                    ax += f * dx
                    ay += f * dy
                    az += f * dz
                    cnt += 1
                else:
                    for c in range(8):
                        ch = child[node, c]
                        if ch >= 0:
                            stack[sp] = ch
                            sp += 1
            else:
                j = head[node]
                while j != -1:
                    if j != i:
                        dx = pos[j, 0] - xi
                        dy = pos[j, 1] - yi
                        dz = pos[j, 2] - zi
                        r2 = dx * dx + dy * dy + dz * dz + s2
                        f = G * masses[j] / (r2 * np.sqrt(r2))
                        ax += f * dx
                        ay += f * dy
                        az += f * dz
                        cnt += 1
                    j = next_body[j]
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
        counts[i] = cnt
    return acc, counts


def accelerations_barnes_hut_numba(pos, masses, G=1.0, softening=0.0, theta=0.5, return_stats=False):
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("accelerations_barnes_hut_numba supports 3D positions only")
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    center = (lo + hi) / 2.0
    half = max(float((hi - lo).max()) / 2.0, 1e-12) * 1.0000001
    cap = 12 * len(pos) + 64
    while True:
        center_a, half_a, child, head, next_body, has_kids, count = _build(pos, cap, center, half)
        if count >= 0:
            break
        cap *= 2                                    # rare: tree needed more nodes than allocated
    mass, com = _moments(pos, masses, child, head, next_body, has_kids, count)
    acc, counts = _forces(pos, masses, half_a, child, head, next_body, has_kids, mass, com,
                          G, softening ** 2, theta * theta)
    if return_stats:
        return acc, int(counts.sum())
    return acc
