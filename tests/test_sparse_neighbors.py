"""CPU numerical checks independent of optional PyTorch/native GPU packages."""
import importlib.util
from itertools import product
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('neighbors', ROOT / 'vendor/triflow_comfy_core/utils/sparse_neighbors.py')
neighbors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neighbors)


class NeighborTests(unittest.TestCase):
    def test_sparse_convolution_matches_dense_reference_with_holes_and_batches(self):
        rng = np.random.default_rng(17)
        coords = np.array([(b, x, y, z) for b, x, y, z in product(range(2), range(4), range(3), range(5)) if (x + y + z) % 3], dtype=np.int64)
        rng.shuffle(coords)
        features = rng.normal(size=(len(coords), 2))
        weights = rng.normal(size=(3, 3, 3, 3, 2))
        bias = rng.normal(size=3)
        for dilation in (1, 2):
            with self.subTest(dilation=dilation):
                actual = np.broadcast_to(bias, (len(coords), 3)).copy()
                for i, (dst, src) in enumerate(neighbors.neighbor_pairs(coords, 3, dilation)):
                    actual[dst] += features[src] @ weights.reshape(3, 27, 2)[:, i].T
                grid = np.zeros((2, 4, 3, 5, 2))
                for coord, feat in zip(coords, features):
                    grid[tuple(coord)] = feat
                expected = np.broadcast_to(bias, actual.shape).copy()
                for n, (b, x, y, z) in enumerate(coords):
                    for a, c, d in product(range(3), repeat=3):
                        q = (x + (a - 1) * dilation, y + (c - 1) * dilation, z + (d - 1) * dilation)
                        if all(0 <= v < bound for v, bound in zip(q, grid.shape[1:4])):
                            expected[n] += grid[(b, *q)] @ weights[:, a, c, d, :].T
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_pointwise_preserves_input_order(self):
        coords = np.array([[1, 2, 0, 0], [0, 3, 1, 2], [0, 0, 0, 0]])
        dst, src = neighbors.neighbor_pairs(coords, 1)[0]
        np.testing.assert_array_equal(dst, np.arange(3))
        np.testing.assert_array_equal(src, np.arange(3))

    def test_invalid_and_empty_coordinates(self):
        for coords in ([[0, 0, 0, 0], [0, 0, 0, 0]], [[0, -1, 0, 0]], [[0, 1, 2]]):
            with self.assertRaises(ValueError):
                neighbors.neighbor_pairs(coords, 3)
        pairs = neighbors.neighbor_pairs(np.empty((0, 4)), 3)
        self.assertEqual(len(pairs), 27)
        self.assertTrue(all(len(dst) == len(src) == 0 for dst, src in pairs))

    def test_voxel_extent_cannot_overflow(self):
        with self.assertRaises(ValueError):
            neighbors.neighbor_pairs([[0, 2**40, 2**40, 0]], 3)

if __name__ == '__main__':
    unittest.main()
