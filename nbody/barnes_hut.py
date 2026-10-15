"""Barnes-Hut tree code: O(N log N) approximate gravity.

Particles are stored in a 2^d-tree (quadtree in 2D, octree in 3D). When a node is
far enough away from a particle, i.e. (node width) / (distance to node centre of mass) < theta,
the whole node is replaced by a single point mass at its centre of mass.

theta = 0 never approximates (equals direct summation); larger theta is faster but less accurate.
"""
import numpy as np


class _Node:
    __slots__ = ("width", "mass", "com", "children", "idx")


def _build(pos, masses, idx, center, half, depth, max_depth=40):
    node = _Node()
    node.width = 2.0 * half
    node.idx = idx
    m = masses[idx]
    node.mass = m.sum()
    node.com = (m[:, None] * pos[idx]).sum(axis=0) / node.mass
    node.children = None
    if len(idx) <= 1 or depth >= max_depth:
        return node                                   # leaf
    d = pos.shape[1]
    # child index = bit pattern of (pos >= center) per dimension
    bits = (pos[idx] >= center).astype(np.int64) @ (1 << np.arange(d))
    node.children = []
    for c in range(1 << d):
        sel = idx[bits == c]
        if len(sel) == 0:
            continue
        offset = np.array([(1 if (c >> k) & 1 else -1) for k in range(d)]) * (half / 2.0)
        node.children.append(_build(pos, masses, sel, center + offset, half / 2.0, depth + 1, max_depth))
    node.idx = None
    return node


def build_tree(pos, masses):
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    center = (lo + hi) / 2.0
    half = max(float((hi - lo).max()) / 2.0, 1e-12) * 1.0000001
    return _build(pos, masses, np.arange(len(pos)), center, half, 0)


def accelerations_barnes_hut(pos, masses, G=1.0, softening=0.0, theta=0.5, return_stats=False):
    root = build_tree(pos, masses)
    n, d = pos.shape
    acc = np.zeros((n, d))
    s2 = softening ** 2
    theta2 = theta * theta
    interactions = 0
    for i in range(n):
        p_i = pos[i]
        a = np.zeros(d)
        stack = [root]
        while stack:
            node = stack.pop()
            if node.children is None:                  # leaf (usually one particle)
                for j in node.idx:
                    if j == i:
                        continue
                    diff = pos[j] - p_i
                    r2 = diff @ diff + s2
                    a += G * masses[j] * diff / (r2 * np.sqrt(r2))
                    interactions += 1
                continue
            diff = node.com - p_i
            r2 = diff @ diff
            if node.width * node.width < theta2 * r2:  # far enough: use centre of mass
                r2 += s2
                a += G * node.mass * diff / (r2 * np.sqrt(r2))
                interactions += 1
            else:
                stack.extend(node.children)
        acc[i] = a
    if return_stats:
        return acc, interactions
    return acc
