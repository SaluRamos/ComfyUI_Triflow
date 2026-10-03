# Bundled source provenance

TriFlow inference components were copied from the workspace's TriFlow project,
under the Automotive Development Public Non-Commercial License v1.0 (`LICENSE`).

| Component | Upstream | Source revision | License file |
| --- | --- | --- | --- |
| TRELLIS inference subset | https://github.com/microsoft/TRELLIS | `f17fdf12d8f17a6a09225f01756d141285dc848f` | `vendor/triflow_comfy_trellis/LICENSE` |
| Direct3D-S2 inference subset | https://github.com/DreamTechAI/Direct3D-S2 | `a1cf235b2881cff04a91900060a9546b40e7ee5d` | `vendor/triflow_comfy_direct3d/LICENSE.txt` |
| Customized pyfqmr | TriFlow's customized fork | `c93a7671ba4a2163dd3c7ea6ab6b52ad0167cc72` | `third_party/pyfqmr-triflow/LICENSE` |

These are regular vendored files, not Git submodules. The workspace copies may
include local changes made before this conversion. Package identifiers were
renamed to `triflow_comfy_*`; initialization was reduced to inference imports.
The dataset retains its batching methods, with visualization helpers removed.
Hydra configuration and checkpoint paths are resolved from this project.

Direct3D-S2's spatial attention kernels retain their original Apache-2.0
copyright headers and attribution to Xunhao Lai and Shuang Wu. See
`vendor/triflow_comfy_direct3d/modules/sparse/attention/spatial_sparse_attention/LICENSE-APACHE-2.0.txt`
for the license text, copied from https://www.apache.org/licenses/LICENSE-2.0.txt.

PyTorch, TorchSparse, spconv, FlashAttention, Triton, MeshLib, Open3D, meshiki,
and other installed packages are external dependencies and are not bundled.
