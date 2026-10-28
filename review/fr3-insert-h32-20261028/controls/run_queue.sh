#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/path/to/storage/SJ/Outer/EasyWAM/.venv/bin/python
export NCCL_CUMEM_HOST_ENABLE=0
exec env -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 \
    NCCL_CUMEM_HOST_ENABLE="$NCCL_CUMEM_HOST_ENABLE" "$PYTHON" "$ROOT/queue_successor.py" "$@"
