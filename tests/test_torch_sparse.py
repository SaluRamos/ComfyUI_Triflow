"""Compare the actual portable layers with PyTorch dense convolution on CPU/XPU."""
import os
from pathlib import Path
import sys
import unittest
try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipIf(torch is None, 'PyTorch not installed')
class TorchSparseTests(unittest.TestCase):
    def test_layers_and_checkpoint_layout(self):
        sys.path.insert(0, str(ROOT / 'vendor'))
        os.environ['TRIFLOW_FLOW_SPARSE_BACKEND'] = 'torch'
        os.environ['TRIFLOW_VAE_SPARSE_BACKEND'] = 'torch'
        from triflow_comfy_trellis.modules.sparse import SparseTensor, SparseConv3d
        from triflow_comfy_core.utils.windows_weights import convert_vae_weights
        devices = ['cpu']
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            devices.append('xpu')
        for device in devices:
            for kernel in (1, 3):
                with self.subTest(device=device, kernel=kernel):
                    torch.manual_seed(19)
                    coords = torch.tensor([[b, x, y, z] for b in range(2) for x in range(3) for y in range(3) for z in range(3) if (x + y + z) % 2], dtype=torch.int32, device=device)
                    feats = torch.randn(len(coords), 2, device=device, requires_grad=True)
                    layer = SparseConv3d(2, 3, kernel).to(device)
                    sparse = SparseTensor(feats, coords)
                    result = layer(sparse)
                    dense = sparse.dense()
                    reference = torch.nn.functional.conv3d(dense, layer.conv.weight.permute(0, 4, 1, 2, 3), layer.conv.bias, padding=kernel//2)
                    b, x, y, z = coords.long().unbind(1)
                    torch.testing.assert_close(result.feats, reference[b, :, x, y, z], atol=1e-5, rtol=1e-4)
                    layer(result.replace(feats))  # Reuse the coordinate cache with new features.
                    with torch.inference_mode():
                        inferred = SparseTensor(feats.detach().clone(), coords.clone())
                        inferred_result = layer(inferred)
                        torch.testing.assert_close(inferred_result.feats, result.feats.detach(), atol=1e-5, rtol=1e-4)
                    result.feats.square().sum().backward()
                    self.assertIsNotNone(feats.grad)
                    vae = torch.arange(kernel**3 * 2 * 3).reshape(kernel**3, 2, 3)
                    converted = convert_vae_weights({'example.conv.kernel': vae})['example.conv.weight']
                    for x in range(kernel):
                        for y in range(kernel):
                            for z in range(kernel):
                                torch.testing.assert_close(converted[:, x, y, z, :], vae[(z*kernel+y)*kernel+x].T)

if __name__ == '__main__':
    unittest.main()
