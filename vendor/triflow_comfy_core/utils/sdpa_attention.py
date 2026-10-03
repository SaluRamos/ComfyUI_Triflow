"""Packed attention interface implemented with native PyTorch SDPA on Windows."""
from collections import defaultdict
import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def flash_attn_qkvpacked_func(qkv):
    return flash_attn_func(*qkv.unbind(2))


def flash_attn_kvpacked_func(q, kv):
    return flash_attn_func(q, *kv.unbind(2))


def flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k):
    q_offsets, k_offsets = cu_q.tolist(), cu_k.tolist()
    groups = defaultdict(list)
    for index in range(len(q_offsets) - 1):
        nq = q_offsets[index + 1] - q_offsets[index]
        nk = k_offsets[index + 1] - k_offsets[index]
        groups[((nq + 31) // 32 * 32, (nk + 31) // 32 * 32)].append(index)
    output = torch.empty((len(q), q.shape[1], v.shape[-1]), device=q.device, dtype=q.dtype)
    for (nq_pad, nk_pad), indices in groups.items():
        # Bound padded workspaces and batch variable windows without crossing boundaries.
        workspace = max(nq_pad * q.shape[1] * q.shape[2], nk_pad * k.shape[1] * k.shape[2],
                        nk_pad * v.shape[1] * v.shape[2])
        batch_size = max(1, 16777216 // workspace)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            qs = q.new_zeros((len(batch), nq_pad, *q.shape[1:]))
            ks = k.new_zeros((len(batch), nk_pad, *k.shape[1:]))
            vs = v.new_zeros((len(batch), nk_pad, *v.shape[1:]))
            mask = torch.zeros((len(batch), 1, 1, nk_pad), device=q.device, dtype=torch.bool)
            for row, index in enumerate(batch):
                a, b = q_offsets[index:index + 2]
                c, d = k_offsets[index:index + 2]
                qs[row, :b-a] = q[a:b]
                ks[row, :d-c] = k[c:d]
                vs[row, :d-c] = v[c:d]
                mask[row, :, :, :d-c] = True
            result = F.scaled_dot_product_attention(qs.transpose(1, 2), ks.transpose(1, 2),
                vs.transpose(1, 2), attn_mask=mask).transpose(1, 2)
            for row, index in enumerate(batch):
                a, b = q_offsets[index:index + 2]
                output[a:b] = result[row, :b-a]
    return output


def flash_attn_varlen_qkvpacked_func(qkv, cu, maximum):
    return flash_attn_varlen_func(*qkv.unbind(1), cu, cu, maximum, maximum)


def flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, max_q, max_k):
    return flash_attn_varlen_func(q, *kv.unbind(1), cu_q, cu_k, max_q, max_k)
