"""Test a relocated copy and compile its QEM without using the parent project."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remesh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="triflow-standalone-") as temporary:
        destination = Path(temporary) / "node"
        shutil.copytree(root, destination, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".cache", "*.egg-info", "build", "*.so", "*.pyd"
        ))
        target = Path(temporary) / "installed"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "--target", str(target), str(destination / "third_party/pyfqmr-triflow"),
        ], cwd=temporary, env=environment, check=True)
        # Prioritize the newly compiled extension over any original editable install.
        environment["PYTHONPATH"] = str(target)
        command = [sys.executable, str(destination / "scripts/check_runtime.py")]
        if args.remesh:
            command.append("--remesh")
        subprocess.run(command, cwd=temporary, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(destination / "tests"), "-v"],
                       cwd=temporary, env=environment, check=True)
        print("Relocated standalone project and locally compiled QEM passed.")
