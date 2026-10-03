"""Load MSVC and provide a private Windows SDK when the system has none."""
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile


def prepare_build_env(root, env):
    installer = root / '.installer'
    installer.mkdir(exist_ok=True)
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    visual_studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not visual_studio:
        raise RuntimeError('Instale as ferramentas C++ do Visual Studio para compilar o QEM do TriFlow.')
    batch = installer / 'msvc-env.cmd'
    vcvars = Path(visual_studio) / 'VC/Auxiliary/Build/vcvars64.bat'
    batch.write_text(f'@echo off\ncall "{vcvars}" >nul\nset\n', encoding='utf-8')
    output = subprocess.check_output(['cmd.exe', '/d', '/c', str(batch)], env=env, text=True, errors='replace')
    for line in output.splitlines():
        if '=' in line and not line.startswith('='):
            key, value = line.split('=', 1)
            env[key.upper()] = value
    env['DISTUTILS_USE_SDK'] = '1'
    env['MSSdk'] = '1'
    sdk = installer / 'sdk'
    version = '10.0.28000.2705'
    includes = sdk / 'microsoft.windows.sdk.cpp/c/Include/10.0.28000.0'
    if not any((Path(path) / 'io.h').is_file() for path in env.get('INCLUDE', '').split(';')):
        for package in ('microsoft.windows.sdk.cpp', 'microsoft.windows.sdk.cpp.x64'):
            target = sdk / package
            if not target.is_dir():
                sdk.mkdir(exist_ok=True)
                archive = sdk / f'{package}.nupkg'
                print(f'Downloading private Windows SDK: {package}', flush=True)
                urllib.request.urlretrieve(f'https://api.nuget.org/v3-flatcontainer/{package}/{version}/{package}.{version}.nupkg', archive)
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(target)
        env['INCLUDE'] = ';'.join(str(includes / part) for part in ('ucrt', 'shared', 'um')) + ';' + env.get('INCLUDE', '')
        libs = sdk / 'microsoft.windows.sdk.cpp.x64/c'
        env['LIB'] = ';'.join(str(libs / part / 'x64') for part in ('ucrt', 'um')) + ';' + env.get('LIB', '')
        env['PATH'] = str(sdk / 'microsoft.windows.sdk.cpp/c/bin/10.0.28000.0/x64') + ';' + env.get('PATH', '')
    return env
