import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("isolated_triflow_node", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from isolated_triflow_node.mesh_adapter import unpack_mesh, pack_mesh


class Mesh:
    def __init__(self, vertices, faces, vertex_counts=None, face_counts=None):
        self.vertices, self.faces = vertices, faces
        self.vertex_counts, self.face_counts = vertex_counts, face_counts


class MeshAdapterTests(unittest.TestCase):
    def input_mesh(self):
        m = trimesh.creation.box()
        m.apply_translation([10, -3, 7])
        return Mesh(torch.tensor(m.vertices.copy()).unsqueeze(0), torch.tensor(m.faces.copy()).unsqueeze(0))

    def test_input_coordinates_and_source_are_preserved(self):
        source = self.input_mesh()
        vertices = source.vertices.clone()
        result = unpack_mesh(source)[0]
        np.testing.assert_allclose(result.vertices, vertices[0])
        result.apply_translation([1, 1, 1])
        torch.testing.assert_close(source.vertices, vertices)

    def test_padding_is_excluded(self):
        m = self.input_mesh()
        m.vertices = torch.cat([m.vertices, torch.full((1, 3, 3), float("nan"))], dim=1)
        m.faces = torch.cat([m.faces, torch.full((1, 2, 3), -1)], dim=1)
        m.vertex_counts, m.face_counts = torch.tensor([8]), torch.tensor([12])
        result = unpack_mesh(m)[0]
        self.assertEqual((len(result.vertices), len(result.faces)), (8, 12))

    def test_invalid_indices_are_rejected(self):
        m = self.input_mesh()
        m.faces[0, 0, 0] = 100
        with self.assertRaisesRegex(ValueError, "invalid face indices"):
            unpack_mesh(m)

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            unpack_mesh(Mesh(torch.zeros(0, 1, 3), torch.zeros(0, 1, 3, dtype=torch.int64)))

    def test_pair_of_counts_is_required(self):
        m = self.input_mesh()
        m.vertex_counts = torch.tensor([8])
        with self.assertRaisesRegex(ValueError, "together"):
            unpack_mesh(m)

    def test_single_output_uses_batched_native_constructor(self):
        native = types.ModuleType("comfy_api.latest._util")
        native.MESH = Mesh
        geometry = unpack_mesh(self.input_mesh())
        with patch.dict(sys.modules, {"comfy_api.latest._util": native}):
            output = pack_mesh(geometry)
        self.assertEqual(output.vertices.shape, (1, 8, 3))
        self.assertEqual(output.faces.dtype, torch.int64)

    def test_variable_batch_uses_native_packing_helper(self):
        native = types.ModuleType("comfy_api.latest._util")
        native.MESH = Mesh
        helper = types.ModuleType("comfy_extras.nodes_save_3d")
        calls = []
        def pack(vertices, faces):
            calls.append(([len(v) for v in vertices], [len(f) for f in faces]))
            return "native packed batch"
        helper.pack_variable_mesh_batch = pack
        with patch.dict(sys.modules, {"comfy_api.latest._util": native, "comfy_extras.nodes_save_3d": helper}):
            output = pack_mesh([trimesh.creation.box(), trimesh.creation.icosphere(subdivisions=1)])
        self.assertEqual(output, "native packed batch")
        self.assertEqual(calls[0][0], [8, 42])

    def test_invalid_parameters_fail_before_loading_cuda(self):
        with self.assertRaises(ValueError):
            package.NODE_CLASS_MAPPINGS["TriFlowRemesh"]().remesh(None, 12, 320, float("nan"))


if __name__ == "__main__":
    unittest.main()
