"""Run TriFlow in its private Windows interpreter, exchanging geometry files."""
import json
import os
from pathlib import Path
import subprocess
import time


def remesh_isolated(inputs, folder, checkpoints, qem_threshold, face_count, quad_ratio, interrupted):
    root = Path(__file__).resolve().parent
    python = root / '.venv' / 'Scripts' / 'python.exe'
    if not python.is_file():
        raise RuntimeError('TriFlow: execute install.py com o Python do ComfyUI para criar o ambiente Windows isolado.')
    paths = []
    for index, mesh in enumerate(inputs):
        path = folder / f'mesh_{index}.obj'
        mesh.export(path, file_type='obj')
        paths.append(str(path))
    request = folder / 'request.json'
    response = folder / 'response.json'
    request.write_text(json.dumps(dict(inputs=paths, output=str(folder / 'output'),
        checkpoints=str(checkpoints), face_count=face_count or None,
        qem_threshold=qem_threshold, quad_ratio=quad_ratio, response=str(response))), encoding='utf-8')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'SPARSE_BACKEND', 'ATTN_BACKEND', 'SPARSE_ATTN_BACKEND'):
        env.pop(key, None)
    env['PYTHONUNBUFFERED'] = '1'
    log_path = folder / 'runtime.log'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(python), '-X', 'faulthandler', str(root / 'windows_worker.py'), str(request)],
            cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                interrupted()
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if process.returncode:
        raise RuntimeError(f'TriFlow no ambiente Windows isolado (exit {process.returncode}):\n' + log_path.read_text(encoding='utf-8', errors='replace')[-6000:])
    return json.loads(response.read_text(encoding='utf-8'))
