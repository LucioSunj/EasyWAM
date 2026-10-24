#!/usr/bin/env bash
# Convert FR3 recordings, prepare model assets/cache, then start native training.
set -euo pipefail
EASYWAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EASYWAM_ROOT}"
EASYWAM_PYTHON="${EASYWAM_PYTHON:-${EASYWAM_ROOT}/.venv/bin/python}"
export PATH="$(dirname -- "${EASYWAM_PYTHON}"):${PATH}"
export PYTHONPATH="${EASYWAM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${EASYWAM_PYTHON}" scripts/run_real_world_fr3.py "$@"
