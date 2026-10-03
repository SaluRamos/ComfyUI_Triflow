"""Install TriFlow; on Windows all packages stay in a private interpreter."""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    if os.name == 'nt':
        tools = root / '.installer'
        uv = tools / 'bin' / 'uv.exe'
        if not uv.is_file():
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(tools), 'uv==0.12.23'], check=True)
        env = os.environ.copy()
        env['UV_PYTHON_INSTALL_DIR'] = str(root / '.python')
        env['UV_CACHE_DIR'] = str(root / '.installer' / 'cache')
        env['UV_LINK_MODE'] = 'copy'
        python = root / '.venv' / 'Scripts' / 'python.exe'
        if not python.is_file():
            subprocess.run([str(uv), 'venv', '--python', '3.11', str(root / '.venv')], env=env, check=True)
        install = [str(uv), 'pip', 'install', '--python', str(python)]
        subprocess.run(install + ['torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'], env=env, check=True)
        subprocess.run(install + ['-r', str(root / 'requirements-runtime.txt'), 'spconv-cu120==2.3.6'], env=env, check=True)
        wheels = list((root / 'wheels').glob('*cp311*win_amd64.whl'))
        if wheels:
            qem = wheels[0]
        else:
            from windows_build import prepare_build_env
            env = prepare_build_env(root, env)
            qem = root / 'third_party/pyfqmr-triflow'
        subprocess.run(install + [str(qem)], env=env, check=True)
        print('TriFlow Windows instalado em .venv. Reinicie o ComfyUI.')
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(root / 'requirements-runtime.txt')], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', str(root / 'third_party/pyfqmr-triflow')], check=True)
        print('Configure spconv, torchsparse e flash-attn conforme README.md.')
