import gc
import logging
import math
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen
from threading import Lock

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_LOCK = Lock()
CHECKPOINT_NAMES = ("flow_model.safetensors", "sdf_vae.safetensors", "nvv_vae.safetensors")


def checkpoint_directory():
    desktop_models = Path(os.environ.get("LOCALAPPDATA", "")) / "Comfy-Desktop" / "ComfyUI-Shared" / "models"
    if os.name == "nt" and desktop_models.is_dir():
        directory = desktop_models / "triflow"
    else:
        import folder_paths
        directory = Path(folder_paths.models_dir) / "triflow"
    directory.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_NAMES:
        target = directory / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        LOGGER.info("TriFlow: downloading %s to %s", name, directory)
        url = f"https://huggingface.co/lihcxr/TriFlow/resolve/main/{name}"
        temporary = None
        try:
            with urlopen(url, timeout=60) as response, NamedTemporaryFile(
                    dir=directory, prefix=name + ".", suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
            size = temporary.stat().st_size
            if size == 0 or (expected is not None and size != int(expected)):
                raise OSError(f"Incomplete TriFlow download: {name}")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return directory


class TriFlowRemesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH",),
            "qem_threshold": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 1000000.0, "step": 0.1,
                                       "tooltip": "Maximum QEM error allowed during simplification."}),
            "face_count": ("INT", {"default": 4000, "min": 0, "max": 1000000, "step": 1,
                                    "tooltip": "Target triangle count. 0 uses the preprocessed input count; the target is approximate."}),
            "quad_ratio": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Model conditioning for quad-like topology. Output remains triangular; 0 is a valid ratio."}),
        }, "optional": {
            "device": (["auto", "cuda", "xpu"], {"default": "auto", "tooltip": "auto follows ComfyUI; xpu uses Intel Arc with the portable sparse backend."}),
        }}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "remesh"
    CATEGORY = "3d/TriFlow"
    DESCRIPTION = "Generates new triangle topology with TriFlow. Preserves input coordinates; UVs and textures require rebaking. Supports NVIDIA CUDA and Intel XPU (portable sparse backend)."

    def remesh(self, mesh, qem_threshold, face_count, quad_ratio, device="auto"):
        if (not math.isfinite(qem_threshold) or qem_threshold < 0
                or not math.isfinite(quad_ratio) or not 0 <= quad_ratio <= 1
                or not isinstance(face_count, int) or face_count < 0):
            raise ValueError("Use qem_threshold >= 0, face_count >= 0 and quad_ratio in [0, 1].")
        from .mesh_adapter import unpack_mesh, pack_mesh
        import torch
        import trimesh
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        if device not in ("auto", "cuda", "xpu"):
            raise ValueError("Select auto, cuda or xpu.")
        selected = mm.get_torch_device() if device == "auto" else torch.device(device)
        if selected.type not in ("cuda", "xpu"):
            raise RuntimeError("TriFlow requires a CUDA or Intel XPU GPU.")
        inputs = unpack_mesh(mesh)
        with INFERENCE_LOCK:
            mm.throw_exception_if_processing_interrupted()
            mm.unload_all_models()
            mm.soft_empty_cache()
            runtime = None
            try:
                if os.name == "nt" or selected.type == "xpu":
                    from .isolated_runtime import remesh_isolated
                    with TemporaryDirectory(prefix="comfy-triflow-") as temporary:
                        paths = remesh_isolated(inputs, Path(temporary), checkpoint_directory(),
                            qem_threshold, face_count, quad_ratio,
                            mm.throw_exception_if_processing_interrupted, device=str(selected))
                        return (pack_mesh([trimesh.load(path, force="mesh", process=False) for path in paths]),)
                # Namespaced local sources avoid collisions with other TRELLIS/Direct3D nodes.
                os.environ["TRIFLOW_DEVICE"] = str(selected)
                from .inference import load_inference_runtime, run_inference
                runtime = load_inference_runtime(checkpoint_directory())
                outputs = []
                progress = ProgressBar(len(inputs))
                with TemporaryDirectory(prefix="comfy-triflow-") as temporary:
                    folder = Path(temporary)
                    for index, item in enumerate(inputs):
                        mm.throw_exception_if_processing_interrupted()
                        input_file = folder / f"mesh_{index}.obj"
                        item.export(input_file, file_type="obj")
                        paths = run_inference(
                            str(folder), str(folder / "output"), rank=0, world_size=1,
                            face_count=face_count or None, qem_threshold=qem_threshold,
                            quad_ratio=quad_ratio, runtime=runtime, mesh_paths=[input_file],
                        )
                        outputs.append(trimesh.load(paths[0], force="mesh", process=False))
                        progress.update(1)
                        mm.throw_exception_if_processing_interrupted()
                return (pack_mesh(outputs),)
            except ImportError as exc:
                raise RuntimeError(
                    f"TriFlow dependency unavailable: {exc}. See README.md and run install.py "
                    "with ComfyUI's Python. Install with --device xpu for Intel; CUDA dependencies must match PyTorch for NVIDIA."
                ) from exc
            finally:
                runtime = None
                gc.collect()
                mm.soft_empty_cache()
