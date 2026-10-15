"""Reference implementation: plain Python double loop, O(N^2).
Deliberately unoptimized. Used as a correctness reference and a speed baseline."""
import math
import numpy as np


def accelerations_naive(pos, masses, G=1.0, softening=0.0):
    p = pos.tolist()
    m = masses.tolist()
    n, d = len(p), len(p[0])
    acc = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = [p[j][k] - p[i][k] for k in range(d)]
            r2 = sum(x * x for x in diff) + softening ** 2
            f = G * m[j] / (r2 * math.sqrt(r2))
            for k in range(d):
                acc[i][k] += f * diff[k]
    return np.array(acc)
