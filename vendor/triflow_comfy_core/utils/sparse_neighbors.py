"""Coordinate maps for stride-one submanifold convolution, built on CPU.

Only active voxels are retained. No dense 512-cubed volume is allocated.
Coordinates are (batch, x, y, z); offset order matches spconv KRSC weights.
"""
from itertools import product
import numpy as np


def neighbor_pairs(coords, kernel_size, dilation=1):
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Expected N x 4 batched voxel coordinates.")
    if kernel_size < 1 or kernel_size % 2 != 1 or dilation < 1:
        raise ValueError("Portable convolution supports positive odd kernels and dilation.")
    if len(coords) == 0:
        return [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)) for _ in range(kernel_size ** 3)]
    if np.any(coords < 0):
        raise ValueError("Voxel coordinates must be nonnegative.")
    extent = coords.max(axis=0) + 1
    if int(np.prod(extent, dtype=object)) > np.iinfo(np.int64).max:
        raise ValueError("Voxel coordinate extent exceeds int64 indexing.")
    multipliers = np.array([int(np.prod(extent[i + 1:], dtype=object)) for i in range(4)], dtype=np.int64)
    codes = coords @ multipliers
    order = np.argsort(codes)
    sorted_codes = codes[order]
    if np.any(np.diff(sorted_codes) == 0):
        raise ValueError("Submanifold convolution requires unique coordinates.")
    radius = kernel_size // 2
    pairs = []
    for offset in product(range(-radius, radius + 1), repeat=3):
        query = coords.copy()
        query[:, 1:] += np.asarray(offset) * dilation
        valid = np.all((query >= 0) & (query < extent), axis=1)
        destinations = np.flatnonzero(valid)
        keys = query[valid] @ multipliers
        positions = np.searchsorted(sorted_codes, keys)
        hit = positions < len(sorted_codes)
        safe = np.minimum(positions, len(sorted_codes) - 1)
        hit &= sorted_codes[safe] == keys
        pairs.append((destinations[hit], order[positions[hit]]))
    return pairs
