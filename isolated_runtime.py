"""Run TriFlow in its private Windows interpreter, exchanging geometry files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def remesh_isolated(inputs, folder, checkpoints, qem_threshold, face_count, quad_ratio, interrupted, device="cuda"):
    root = Path(__file__).resolve().parent
    is_xpu = device.split(":")[0] == "xpu"
    runtime_name = '.venv-xpu' if is_xpu else '.venv'
    python = root / runtime_name / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        raise RuntimeError('TriFlow: execute install.py com o Python do ComfyUI para criar o ambiente isolado (--device xpu para Intel).')
    paths = []
    for index, mesh in enumerate(inputs):
        path = folder / f'mesh_{index}.obj'
        mesh.export(path, file_type='obj')
        paths.append(str(path))
    request = folder / 'request.json'
    response = folder / 'response.json'
    request.write_text(json.dumps(dict(inputs=paths, output=str(folder / 'output'),
        checkpoints=str(checkpoints), face_count=face_count or None,
        qem_threshold=qem_threshold, quad_ratio=quad_ratio, response=str(response), device=device)), encoding='utf-8')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'SPARSE_BACKEND', 'ATTN_BACKEND', 'SPARSE_ATTN_BACKEND'):
        env.pop(key, None)
    env['PYTHONUNBUFFERED'] = '1'
    env['TRIFLOW_DEVICE'] = device
    for key in ('TRIFLOW_FLOW_SPARSE_BACKEND', 'TRIFLOW_VAE_SPARSE_BACKEND'):
        env.pop(key, None)
    log_path = folder / 'runtime.log'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(python), '-X', 'faulthandler', str(root / 'windows_worker.py'), str(request)],
            cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                interrupted()
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if process.returncode:
        raise RuntimeError(f'TriFlow no ambiente isolado (exit {process.returncode}):\n' + log_path.read_text(encoding='utf-8', errors='replace')[-6000:])
    return json.loads(response.read_text(encoding='utf-8'))
