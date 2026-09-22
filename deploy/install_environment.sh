#!/bin/sh
# Operates only inside the independently deployed FPE directory; no model/data downloads.
set -eu
FPE_ROOT=${1:-/mnt/workspace/fpe-self-consistency}
FPE_DEVICE=${2:-cpu}
if [ "$FPE_ROOT" != /mnt/workspace/fpe-self-consistency ]; then
    printf '%s\n' 'Refusing an unapproved remote root' >&2
    exit 2
fi
case "$FPE_DEVICE" in cpu|gpu) ;; *) exit 2;; esac
cd "$FPE_ROOT"
mkdir -p receipts
export UV_CACHE_DIR="$FPE_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$FPE_ROOT/.python"
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
finish() {
    FPE_EXIT=$?
    python3 - "$FPE_EXIT" <<'PYRESULT'
import json, sys, time
from pathlib import Path
Path("receipts/install-result.json").write_text(json.dumps({"exit_code": int(sys.argv[1]), "finished_unix": time.time()}))
PYRESULT
}
trap finish EXIT
FPE_PACKAGE_INDEX=${FPE_PACKAGE_INDEX:-https://pypi.org/simple}
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
fi
uv export --frozen --all-groups --no-emit-project --output-file receipts/cpu-requirements.lock
uv pip install --python .venv/bin/python --index "$FPE_PACKAGE_INDEX" --default-index https://pypi.org/simple --index-strategy unsafe-best-match \
    --require-hashes -r receipts/cpu-requirements.lock
uv pip install --python .venv/bin/python --index "$FPE_PACKAGE_INDEX" --default-index https://pypi.org/simple --index-strategy unsafe-best-match --no-deps -e .
if [ "$FPE_DEVICE" = gpu ]; then
    FPE_JAX_VERSION=$(.venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("jax"))')
    # Extra CUDA libraries live only in this virtualenv; CPU package versions remain frozen.
    if [ -f deploy/locks/cuda-requirements.lock ]; then
        cp deploy/locks/cuda-requirements.lock receipts/cuda-requirements.lock
    else
        uv pip compile --python .venv/bin/python --index "$FPE_PACKAGE_INDEX" --default-index https://pypi.org/simple --index-strategy unsafe-best-match --generate-hashes \
            --constraint receipts/cpu-requirements.lock --output-file receipts/cuda-requirements.lock - <<EOF
jax[cuda12]==$FPE_JAX_VERSION
nvidia-nccl-cu12==${FPE_NCCL_VERSION:-2.31.2}
EOF
    fi
    uv pip install --python .venv/bin/python --index "$FPE_PACKAGE_INDEX" --default-index https://pypi.org/simple --index-strategy unsafe-best-match --require-hashes --constraint receipts/cpu-requirements.lock -r receipts/cuda-requirements.lock
fi
uv pip freeze --python .venv/bin/python > receipts/installed.txt
.venv/bin/python deploy/runtime_inventory.py --root "$FPE_ROOT" --output receipts/runtime.json
