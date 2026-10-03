from .full_attn import *
from .serialized_attn import *
from .windowed_attn import *
from .modules import *
def __getattr__(name):
    if name == "SpatialSparseAttention":
        from .spatial_sparse_attention.module.spatial_sparse_attention import SpatialSparseAttention
        return SpatialSparseAttention
    raise AttributeError(name)
