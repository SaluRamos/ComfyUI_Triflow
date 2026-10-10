"""Run TriFlow in its private Windows interpreter, exchanging geometry files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
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
    print(f"[TriFlow] Iniciando worker isolado em {device}; log: {log_path}", flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(python), '-X', 'faulthandler', str(root / 'windows_worker.py'), str(request)],
            cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                if os.name == "nt" else 0)

        def forward_output():
            pending = ""
            decoder = __import__('codecs').getincrementaldecoder('utf-8')(errors='replace')
            last_progress = 0.0
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk.decode('utf-8', errors='replace'))
                log.flush()
                pending += decoder.decode(chunk)
                while '\n' in pending or '\r' in pending:
                    newline = pending.find('\n')
                    carriage = pending.find('\r')
                    positions = [p for p in (newline, carriage) if p >= 0]
                    split_at = min(positions)
                    line, pending = pending[:split_at], pending[split_at + 1:]
                    if pending.startswith('\n') and split_at == newline:
                        pending = pending[1:]
                    if line.strip():
                        now = time.monotonic()
                        # tqdm refreshes can be frequent; cap progress chatter.
                        if split_at == carriage and now - last_progress < 1.0:
                            continue
                        print(f"[TriFlow] {line.strip()}", flush=True)
                        if split_at == carriage:
                            last_progress = now
            pending += decoder.decode(b'', final=True)
            if pending.strip():
                print(f"[TriFlow] {pending.strip()}", flush=True)

        output_thread = threading.Thread(target=forward_output, name='triflow-output', daemon=True)
        output_thread.start()
        started = time.monotonic()
        next_heartbeat = started + 20
        try:
            while process.poll() is None:
                interrupted()
                if time.monotonic() >= next_heartbeat:
                    print(f"[TriFlow] Worker ainda ativo ({int(time.monotonic() - started)} s).", flush=True)
                    next_heartbeat = time.monotonic() + 20
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    # torch/XPU may leave helper processes inheriting stdout.
                    # Killing only Python can leave runtime.log locked, which
                    # then makes TemporaryDirectory cleanup fail on Windows.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        check=False, timeout=15,
                    )
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            output_thread.join(timeout=10)
    if process.returncode:
        raise RuntimeError(f'TriFlow no ambiente isolado (exit {process.returncode}):\n' + log_path.read_text(encoding='utf-8', errors='replace')[-6000:])
    return json.loads(response.read_text(encoding='utf-8'))
