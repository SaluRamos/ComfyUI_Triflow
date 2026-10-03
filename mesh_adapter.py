"""Convert native ComfyUI MESH batches to/from geometry-only trimesh objects."""
import numpy as np
import torch
import trimesh


def unpack_mesh(mesh):
    vertices, faces = mesh.vertices, mesh.faces
    if not isinstance(vertices, torch.Tensor) or not isinstance(faces, torch.Tensor):
        raise TypeError("TriFlow expects ComfyUI's native tensor-backed MESH. Convert TRIMESH to MESH first.")
    if vertices.ndim == faces.ndim == 2:
        vertices, faces = vertices.unsqueeze(0), faces.unsqueeze(0)
    if (vertices.ndim != 3 or faces.ndim != 3 or vertices.shape[-1] != 3
            or faces.shape[-1] != 3 or vertices.shape[0] != faces.shape[0]
            or vertices.shape[0] == 0):
        raise ValueError("MESH must contain vertices (B, N, 3) and triangular faces (B, F, 3).")
    if faces.dtype not in (torch.int32, torch.int64):
        raise ValueError("MESH face indices must be integer tensors.")
    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)
    if (vertex_counts is None) != (face_counts is None):
        raise ValueError("MESH must supply vertex_counts and face_counts together.")
    if vertex_counts is not None and (len(vertex_counts) != len(vertices) or len(face_counts) != len(faces)):
        raise ValueError("MESH counts must have one entry per batch item.")
    result = []
    for index in range(len(vertices)):
        nv = int(vertex_counts[index]) if vertex_counts is not None else vertices.shape[1]
        nf = int(face_counts[index]) if face_counts is not None else faces.shape[1]
        if not 0 < nv <= vertices.shape[1] or not 0 < nf <= faces.shape[1]:
            raise ValueError(f"MESH item {index} has invalid or empty geometry counts.")
        v = vertices[index, :nv].detach().cpu().float().numpy().copy()
        f = faces[index, :nf].detach().cpu().long().numpy().copy()
        if not np.isfinite(v).all() or f.min() < 0 or f.max() >= nv:
            raise ValueError(f"MESH item {index} has non-finite vertices or invalid face indices.")
        if np.ptp(v, axis=0).max() <= 0:
            raise ValueError(f"MESH item {index} has zero spatial extent.")
        result.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
    return result


def pack_mesh(meshes):
    # Current ComfyUI location, with backward compatibility for Hunyuan3D releases.
    try:
        from comfy_api.latest._util import MESH
    except ImportError:
        from comfy_extras.nodes_hunyuan3d import MESH
    if not meshes or any(len(m.vertices) == 0 or len(m.faces) == 0 for m in meshes):
        raise RuntimeError("TriFlow returned an empty mesh.")
    vertices = [torch.tensor(np.asarray(m.vertices).copy(), dtype=torch.float32) for m in meshes]
    faces = [torch.tensor(np.asarray(m.faces).copy(), dtype=torch.int64) for m in meshes]
    if len({(len(v), len(f)) for v, f in zip(vertices, faces)}) == 1:
        return MESH(torch.stack(vertices), torch.stack(faces))
    try:
        from comfy_extras.nodes_save_3d import pack_variable_mesh_batch
    except ImportError as exc:
        raise RuntimeError("Update ComfyUI to support variable-size mesh batches, or process one mesh at a time.") from exc
    return pack_variable_mesh_batch(vertices, faces)
