"""Validate imports/checkpoints, optionally exercise real CUDA remeshing."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remesh", action="store_true", help="Run real inference on an offset, scaled icosphere")
    args = parser.parse_args()
    import inference
    print(f"Python {sys.version.split()[0]}, PyTorch {inference.torch.__version__}, CUDA {inference.torch.version.cuda}", flush=True)
    runtime = inference.load_inference_runtime()
    print("Standalone CUDA runtime and all three checkpoints loaded.", flush=True)
    import pyfqmr_triflow
    print(f"QEM extension: {pyfqmr_triflow.__file__}", flush=True)
    if args.remesh:
        import numpy as np
        import trimesh
        with TemporaryDirectory(prefix="triflow-check-") as temporary:
            folder = Path(temporary)
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
            mesh.apply_translation([10.0, -4.0, 3.0])
            path = folder / "input.obj"
            mesh.export(path)
            outputs = inference.run_inference(
                str(folder), str(folder / "output"), rank=0, world_size=1,
                face_count=320, qem_threshold=12.0, quad_ratio=0.95,
                runtime=runtime, mesh_paths=[path],
            )
            result = trimesh.load(outputs[0], force="mesh", process=False)
            assert len(result.vertices) and len(result.faces)
            assert np.isfinite(result.vertices).all()
            assert np.allclose(result.bounds, mesh.bounds, atol=0.25), (result.bounds, mesh.bounds)
            print(f"CUDA inference passed: {len(result.vertices)} vertices, {len(result.faces)} faces; source coordinates preserved.")
