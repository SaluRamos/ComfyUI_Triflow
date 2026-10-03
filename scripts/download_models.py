"""Download model weights into this project, independent of the working directory."""
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    for name in ("flow_model.safetensors", "sdf_vae.safetensors", "nvv_vae.safetensors"):
        print(hf_hub_download("lihcxr/TriFlow", name, local_dir=args.output_dir))
