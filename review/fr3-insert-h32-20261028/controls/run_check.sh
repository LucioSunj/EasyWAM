#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/path/to/storage/SJ/Outer/EasyWAM/.venv/bin/python
exec env -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/queue_successor.py" --check "$@"
