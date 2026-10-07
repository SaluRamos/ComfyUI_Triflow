"""Select the inference device before importing the vendored sparse modules."""
import os
import sys
import torch


def resolve_device(requested=None):
    requested = requested or os.environ.get("TRIFLOW_DEVICE", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "xpu"
    device = torch.device(requested)
    if device.type not in ("cuda", "xpu"):
        raise ValueError("TriFlow inference requires cuda or xpu.")
    backend = getattr(torch, device.type, None)
    if backend is None or not backend.is_available():
        raise RuntimeError(f"TriFlow: {device.type.upper()} unavailable; install the corresponding PyTorch build and GPU driver.")
    index = device.index if device.index is not None else backend.current_device()
    if index >= backend.device_count():
        raise RuntimeError(f"TriFlow: {device.type}:{index} is not available.")
    return torch.device(f"{device.type}:{index}")


def configure_backends(device):
    if device.type != "xpu":
        return
    for name in ("triflow_comfy_trellis.modules.sparse", "triflow_comfy_direct3d.modules.sparse"):
        module = sys.modules.get(name)
        if module is not None and module.BACKEND != "torch":
            raise RuntimeError("TriFlow: changing sparse backend requires a fresh inference process.")
    os.environ["TRIFLOW_FLOW_SPARSE_BACKEND"] = "torch"
    os.environ["TRIFLOW_VAE_SPARSE_BACKEND"] = "torch"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["ATTN_BACKEND"] = "sdpa"


class InferenceContext:
    """Minimal device/AMP interface consumed by the inference callbacks."""
    def __init__(self, device):
        self.device = device

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.float16)
