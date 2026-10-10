"""Install TriFlow; on Windows all packages stay in a private interpreter."""
import os
import argparse
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('auto', 'cuda', 'xpu'), default='auto')
    args = parser.parse_args()
    device = args.device
    if device == 'auto':
        try:
            import torch
            device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() and not torch.cuda.is_available() else 'cuda'
        except ImportError:
            device = 'cuda'
    is_xpu = device == 'xpu'
    root = Path(__file__).resolve().parent
    if os.name == 'nt' or is_xpu:
        tools = root / '.installer'
        uv = tools / 'bin' / ('uv.exe' if os.name == 'nt' else 'uv')
        if not uv.is_file():
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(tools), 'uv==0.12.23'], check=True)
        env = os.environ.copy()
        env['UV_PYTHON_INSTALL_DIR'] = str(root / '.python')
        env['UV_CACHE_DIR'] = str(root / '.installer' / 'cache')
        env['UV_LINK_MODE'] = 'copy'
        runtime = root / ('.venv-xpu' if is_xpu else '.venv')
        python = runtime / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        python_version = '3.12' if is_xpu else '3.11'
        if not python.is_file() or subprocess.run(
                [str(python), '-c', 'pass'], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL).returncode:
            subprocess.run([str(uv), 'venv', '--allow-existing', '--relocatable',
                            '--python', python_version, str(runtime)], env=env, check=True)
        install = [str(uv), 'pip', 'install', '--python', str(python)]
        torch_packages = ['torch', '--index-url', 'https://download.pytorch.org/whl/xpu'] if is_xpu else ['torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124']
        subprocess.run(install + torch_packages, env=env, check=True)
        native_packages = [] if is_xpu else ['spconv-cu120==2.3.6']
        subprocess.run(install + ['-r', str(root / 'requirements-runtime.txt')] + native_packages, env=env, check=True)
        wheel_tag = 'cp312' if is_xpu else 'cp311'
        wheels = list((root / 'wheels').glob(f'*{wheel_tag}*win_amd64.whl')) if os.name == 'nt' else []
        if wheels:
            qem = wheels[0]
        else:
            if os.name == 'nt':
                from windows_build import prepare_build_env
                env = prepare_build_env(root, env)
            qem = root / 'third_party/pyfqmr-triflow'
        subprocess.run(install + [str(qem)], env=env, check=True)
        print(f'TriFlow {device} instalado em {runtime.name}. Reinicie o ComfyUI.')
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(root / 'requirements-runtime.txt')], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', str(root / 'third_party/pyfqmr-triflow')], check=True)
        print('Configure spconv, torchsparse e flash-attn conforme README.md.')
