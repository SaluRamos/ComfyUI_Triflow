# Intel Arc / Arc Pro B70: portable sparse backend

Intel support uses native PyTorch XPU, SDPA and a portable submanifold convolution implementation. It does not install spconv, TorchSparse, CUDA FlashAttention or CUDA Triton for Intel inference. NVIDIA continues using its native sparse backends.

## Installation

Install the Intel graphics/compute driver appropriate for Windows or Linux. From the node folder, using ComfyUI's Python:

```sh
python install.py --device xpu
```

The installer creates a private Python 3.12 environment in `.venv-xpu`, installs the current official PyTorch XPU wheel and the runtime dependencies, and builds the existing QEM extension. Windows needs the C++ build environment described in README.md if no matching QEM wheel exists; Linux needs a C++ compiler and the extension build prerequisites. This preserves ComfyUI's own PyTorch/NumPy versions. CUDA Windows uses the existing `.venv` separately.

Verify GPU visibility with `.venv-xpu/Scripts/python.exe` on Windows or `.venv-xpu/bin/python` on Linux:

```sh
python -c "import torch; assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
```

Use the appropriate interpreter in place of `python` in the command above. Restart ComfyUI and select `xpu` in TriFlowRemesh, or `auto` when ComfyUI selects Intel. The worker runs in the private environment on both Windows and Linux, preserving the selected GPU index and releasing its allocations on exit. Existing workflows without the optional device input keep using auto.

For standalone inference using the private XPU interpreter, set `TRIFLOW_DEVICE=xpu` (or `xpu:1`) before invoking inference.py. Example on Linux:

```sh
TRIFLOW_DEVICE=xpu .venv-xpu/bin/python inference.py -i input_meshes -o output_meshes
```

The three checkpoints still belong in the documented checkpoint directory; no model formats are changed.

## Sparse backend and limitations

The portable backend retains active coordinates and builds cached neighbor maps on CPU. It computes feature matrix products on the GPU in bounded chunks, using the original spconv KRSC weight layout. Original TorchSparse VAE kernels are converted using the existing conversion used by Windows CUDA. It does not allocate a dense high-resolution volume for convolution.

Only positive odd, stride-one submanifold convolutions are implemented, matching the published TriFlow configuration. Downsampling/subdivision remain separate operators. Unsupported strided/inverse convolutions fail explicitly rather than silently changing the model. Sparse full/window attention uses SDPA. Arbitrary alternate model configurations using serialized CUDA voxel encoders or native spatial sparse attention are outside this support path.

The coordinate-map builder is tested numerically against an independent dense NumPy reference with holes, multiple batches, asymmetric kernels, dilation and invalid coordinates. Tests for the actual PyTorch layers, gradients and VAE weight orientation are included and exercise CPU plus XPU when available. Python compilation and the NumPy tests passed; PyTorch tests were skipped locally because PyTorch is unavailable. Full remeshing, model output equivalence, runtime and VRAM on Arc Pro B70 have not been validated in this environment. Treat the backend as experimental until those checks are performed.

The portable implementation will generally be slower than specialized sparse CUDA kernels, especially when building large coordinate maps. Start with a small mesh. On the XPU runtime run:

```sh
python -m unittest discover -s tests -p test_sparse_neighbors.py -v
python -m unittest discover -s tests -p test_torch_sparse.py -v
```

For production validation, compare vertex coordinates and mesh quality against the CUDA path with the same input, checkpoint and settings. Floating-point and AMP differences can change the resulting topology.
