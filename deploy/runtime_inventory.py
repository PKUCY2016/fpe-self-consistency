"""Produce a compact version/hash receipt without credentials or environment dumps."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


def inventory(root: Path) -> dict:
    versions = {}
    for name in ('fpe-self-consistency', 'jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
                 'jax-cuda13-plugin', 'optax', 'numpy', 'scipy', 'matplotlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hashes = {}
    for path in sorted([*root.glob('src/fpe_solver/*.py'), *root.glob('deploy/*.py'),
                        root / 'pyproject.toml', root / 'uv.lock']):
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'observed_unix': time.time(), 'host': platform.node(), 'python': platform.python_version(),
              'executable': os.sys.executable, 'platform': platform.platform(), 'versions': versions,
              'source_sha256': hashes, 'filesystem': shutil.disk_usage(root)._asdict(),
              'filesystem_is_quota': False}
    if shutil.which('nvidia-smi'):
        result['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used',
                                       '--format=csv,noheader'], capture_output=True, text=True,
                                      timeout=10, check=True).stdout.strip()
        result['compute_processes'] = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                                      '--format=csv,noheader'], capture_output=True,
                                                     text=True, timeout=10, check=True).stdout.strip()
    # Device initialization is deliberately separate, only inside the supervisor's GPU lock.
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/mnt/workspace/fpe-self-consistency'))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    value = json.dumps(inventory(a.root), indent=2, sort_keys=True)
    if a.output:
        a.output.write_text(value + '\n')
    else:
        print(value)
