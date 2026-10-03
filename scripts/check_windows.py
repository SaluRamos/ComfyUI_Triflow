"""Exercise the private Windows runtime with a translated, scaled mesh."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isolated_runtime import remesh_isolated
from nodes import checkpoint_directory

if __name__ == '__main__':
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    mesh.apply_translation([10.0, -4.0, 3.0])
    # Prefer local weights; models/triflow also works when invoked from ComfyUI.
    checkpoints = ROOT / 'checkpoints'
    with TemporaryDirectory(prefix='triflow-windows-check-') as temporary:
        paths = remesh_isolated([mesh], Path(temporary), checkpoints, 12.0, 320, 0.95, lambda: None)
        output = trimesh.load(paths[0], force='mesh', process=False)
        assert len(output.vertices) and len(output.faces)
        assert np.isfinite(output.vertices).all()
        assert np.allclose(output.bounds, mesh.bounds, atol=0.25), output.bounds
        print(f'Windows CUDA OK: {len(output.vertices)} vertices, {len(output.faces)} faces; coordinates preserved.')
