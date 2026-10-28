"""Run the successor's CPU-only tests and seal their exact control hashes."""
import json
import os
from pathlib import Path
import subprocess
import sys

from insert_common import ROOT, input_hashes, write_new


def main():
    result_path = ROOT / 'CPU_TEST_RESULTS.json'
    if result_path.exists():
        raise RuntimeError(f'CPU test result already exists; manual review required: {result_path}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='')
    command = [sys.executable, '-m', 'unittest', 'discover', '-v', '-s', str(ROOT), '-p', 'test_*.py']
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True)
    log_path = ROOT / 'cpu-test.log'
    log_path.write_text(completed.stdout + completed.stderr)
    result = {
        'passed': completed.returncode == 0,
        'returncode': completed.returncode,
        'command': command,
        'log_path': str(log_path),
    }
    # The result itself is deliberately excluded from this snapshot: all
    # other controls must be present and immutable before the seal is made.
    result['input_hashes'] = input_hashes()
    write_new(result_path, result)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
