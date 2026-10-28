"""Independent read-only Insert monitor; never starts or repairs training."""
import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from insert_common import FORMAL_OUTPUT, ROOT, live_quota, now, read


def maybe(path):
    return read(path) if Path(path).is_file() else None


def process(pid):
    if not pid:
        return None
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start_ticks': fields[19],
                'cmdline': (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


def update_summary(rank):
    path = ROOT / f'updates-formal-rank-{rank}.jsonl'
    if not path.is_file():
        return None
    rows = []
    for line in deque(path.open(), maxlen=250):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None
    last = rows[-1]
    return {'rank': rank, 'last': last, 'process': process(last['pid']),
            'recent_count': len(rows),
            'recent_contiguous': [r['step'] for r in rows] == list(range(rows[0]['step'], last['step'] + 1)),
            'recent_finite': all(r['loss_finite'] is True and math.isfinite(r['loss'])
                                 and math.isfinite(r['global_grad_norm']) for r in rows)}


def snapshot():
    queue = maybe(ROOT / 'QUEUE_STATUS.json')
    formal = maybe(ROOT / 'STATUS-formal.json')
    result = {
        'observed_at': now(), 'monitor_pid': os.getpid(),
        'insert_queue': queue, 'queue_process': process(queue.get('queue_pid')) if queue else None,
        'fresh': maybe(ROOT / 'B/STATUS-fresh.json'),
        'resume': maybe(ROOT / 'B/STATUS-resume.json'), 'formal': formal,
        'formal_launcher': process(formal.get('training_pid')) if formal else None,
        'preflight_acceptance': maybe(ROOT / 'PREFLIGHT_ACCEPTANCE.json'),
        'completion': maybe(ROOT / 'COMPLETION_AUDIT.json'),
        'early_stop_handoff_used': maybe(ROOT / 'EARLY_STOP_HANDOFF_USED.json'),
        'formal_output_exists': FORMAL_OUTPUT.exists(),
        'updates': [update_summary(rank) for rank in range(4)],
    }
    try:
        result['quota'] = live_quota()
        result['gpu'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True)
    except Exception as exc:
        result['observation_error'] = repr(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if not args.loop:
        print(json.dumps(snapshot(), indent=2, ensure_ascii=True))
        return
    with (ROOT / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                item = snapshot()
            except Exception as exc:
                item = {'observed_at': now(), 'monitor_pid': os.getpid(), 'monitor_error': repr(exc)}
            with (ROOT / 'monitor.jsonl').open('a') as output:
                output.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + '\n')
                output.flush()
            time.sleep(120)


if __name__ == '__main__':
    main()
