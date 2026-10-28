#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/path/to/storage/SJ/Outer/EasyWAM/.venv/bin/python
exec "$PYTHON" "$ROOT/run_cpu_tests.py"
