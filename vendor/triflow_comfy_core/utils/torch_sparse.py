"""Portable active-voxel tensors and exact stride-one convolutions.

The published TriFlow models use odd stride-one convolutions plus separate
pooling/subdivision. Features and matrix multiplication stay on CUDA/XPU/CPU;
coordinate maps use NumPy on CPU. Checkpoints retain spconv KRSC weight names.
"""
import math
import torch
from torch import nn
from .sparse_neighbors import neighbor_pairs


class SparseTensorData:
    def __init__(self, features, indices, spatial_shape, batch_size,
                 grid=None, voxel_num=None, indice_dict=None, **kwargs):
        self._features, self.indices = features, indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = batch_size
        self.grid, self.voxel_num = grid, voxel_num
        self.indice_dict = {} if indice_dict is None else indice_dict
        self.benchmark = False
        self.benchmark_record = {}
        self.thrust_allocator = self._timer = self.force_algo = self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def dense(self):
        features = self.features.reshape(len(self.indices), -1)
        output = features.new_zeros((self.batch_size, features.shape[1], *self.spatial_shape))
        b, x, y, z = self.indices.long().unbind(1)
        output[b, :, x, y, z] = features
        return output


class ConvKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        bound = 1 / math.sqrt(in_channels * kernel_size ** 3)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)


def make_layers(SparseTensor):
    class SparseConv3d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, padding=0, bias=True, indice_key=None):
            super().__init__()
            if stride not in (1, (1, 1, 1)) or not isinstance(kernel_size, int) or kernel_size % 2 != 1:
                raise NotImplementedError("Portable TriFlow supports odd, stride-one submanifold convolutions only.")
            if not isinstance(dilation, int) or dilation < 1:
                raise ValueError("Dilation must be a positive integer.")
            self.kernel_size, self.dilation = kernel_size, dilation
            self.conv = ConvKernel(in_channels, out_channels, kernel_size, bias)

        def forward(self, x):
            feats = x.feats
            if feats.ndim != 2:
                raise ValueError("Convolution features must have shape N x C.")
            # The cache follows coordinate storage and survives feature-only replace().
            try:
                version = x.coords._version
            except RuntimeError:  # ComfyUI may create inference-mode tensors.
                version = None
            key = ("torch_conv", x.coords.data_ptr(), version, tuple(x.coords.shape), self.kernel_size, self.dilation, str(feats.device))
            cached = x.data.indice_dict.get(key)
            pairs = None if cached is None else cached[1]
            if pairs is None:
                cpu_pairs = neighbor_pairs(x.coords.detach().cpu().numpy(), self.kernel_size, self.dilation)
                pairs = [(torch.as_tensor(dst, device=feats.device), torch.as_tensor(src, device=feats.device))
                         for dst, src in cpu_pairs]
                # Retain coordinate storage so a reused allocation cannot hit stale maps.
                x.data.indice_dict[key] = (x.coords.detach(), pairs)
            weights = self.conv.weight.reshape(self.conv.weight.shape[0], -1, self.conv.weight.shape[-1])
            # Match AMP matmul dtype, while bounding the gather workspace.
            output_dtype = torch.get_autocast_dtype(feats.device.type) if torch.is_autocast_enabled(feats.device.type) else feats.dtype
            result = torch.zeros((len(feats), weights.shape[0]), device=feats.device, dtype=output_dtype)
            for offset, (dst, src) in enumerate(pairs):
                for start in range(0, len(src), 4096):
                    target = dst[start:start + 4096]
                    value = feats[src[start:start + 4096]] @ weights[:, offset, :].T
                    # Targets are unique for each offset; no atomic scatter is required.
                    result[target] = result[target] + value.to(output_dtype)
            if self.conv.bias is not None:
                result = result + self.conv.bias.to(output_dtype)
            return x.replace(result)

    class SparseInverseConv3d(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise NotImplementedError("TriFlow's portable backend uses pooling/subdivision, not inverse convolutions.")

    return SparseConv3d, SparseInverseConv3d
