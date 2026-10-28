"""One-shot fresh Insert after the user's verified Drawer early-stop handoff."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from early_stop_handoff import HANDOFF, verify_early_stop

from insert_common import (ARTIFACT_INTEGRITY, ARM_DIR, AUTHORIZATION, BASE,
                           CONFIG, DRAWER_BASELINE_SHA256, DRAWER_RECOVERY,
                           FORMAL_OUTPUT, GPU_LOCK, MIN_FORMAL_QUOTA_BYTES,
                           MIN_HEADROOM_BYTES, MIN_SAVE_QUOTA_BYTES,
                           MIN_PRELAUNCH_QUOTA_BYTES, MIN_RESUME_QUOTA_BYTES,
                           NCCL_CUMEM_HOST_ENABLE, PYTHON, ROOT, RUN_NAME,
                           SOURCE, PREPARATION, REFERENCE, SEALED, atomic,
                           checked_identity, checkpoint, check_quota, digest, finite_log,
                           identity, live_quota, now, read, require,
                           require_nccl_environment, validate_config, validate_preflight_report,
                           verify_sealed_inputs, write_new)

RECOVERY_OUTPUT = (BASE / 'Outer/EasyWAM/runs/real_world_fr3_fastwam_idm_wan22/'
                   'fr3_drawer_fastwam_idm_fullfix_h32_4gpu_b4a4_z1_20261027_recovery2000_ncclfix')
PORT = 29661


def queue_status(value, **fields):
    atomic(ROOT / 'QUEUE_STATUS.json', {
        'status': value, 'utc': now(), 'queue_pid': os.getpid(),
        'automatic_restart': False, **fields,
    })


def verify_preparation():
    preparation = read(PREPARATION)
    require(preparation['prepared'] is True, 'Successor preparation is not sealed')
    require(preparation.get('runtime_environment') ==
            {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
            'Successor preparation is missing the mandatory NCCL environment')
    for path, expected in preparation['control_hashes'].items():
        require(digest(path) == expected, f'Successor control changed: {path}')
    authorization = read(AUTHORIZATION)
    require(authorization['approved'] is True and
            authorization['automatic_retry'] is False and
            authorization['initialization'] == 'WanRobot fresh' and
            authorization['resume'] is None and
            authorization.get('runtime_environment') ==
            {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
            'Successor authorization is not explicit one-shot fresh training')
    baseline = read(REFERENCE)
    validate_config(baseline, read(CONFIG))
    require(digest(CONFIG) == authorization['config_sha256'] and
            digest(SEALED) == authorization['sealed_inputs_sha256'],
            'Successor authorization/config/sealed-input binding changed')
    sealed = verify_sealed_inputs()
    require(sealed['old_baseline_config_sha256'] == authorization['old_insert_baseline_sha256'],
            'Old Insert baseline identity changed')
    return preparation, authorization, sealed


def predecessor_ready():
    return verify_early_stop()


def wait_for_predecessor():
    # This root accepts an explicit early-stop handoff, not a failed old queue.
    return predecessor_ready()


def gpu_idle():
    memory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True)
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True)
    values = [line.strip() for line in memory.splitlines() if line.strip()]
    require(len(values) == 4, 'Expected exactly four GPUs')
    return all(int(value) < 512 for value in values) and not processes.strip()


def wait_for_gpu_release_and_idle():
    """Wait for the predecessor's finally block and then for all GPUs to quiesce."""
    while True:
        candidate = GPU_LOCK.open('a')
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close()
            queue_status('waiting_drawer_gpu_release',
                         required='Authorized Drawer early stop; waiting for GPU lock release')
            time.sleep(15)
            continue
        try:
            if gpu_idle():
                return candidate
        except BaseException:
            candidate.close()
            raise
        candidate.close()
        queue_status('waiting_gpu_idle', required='GPU lock free but GPUs still have active work')
        time.sleep(15)


def _stage_paths(mode):
    if mode in ('fresh', 'resume'):
        return ARM_DIR / mode, ARM_DIR / f'STATUS-{mode}.json', ROOT / 'preflight_four_gpu.py'
    if mode == 'formal':
        return FORMAL_OUTPUT, ROOT / 'STATUS-formal.json', ROOT / 'formal_train.py'
    raise RuntimeError(f'Unknown Insert stage: {mode}')


def minimum_quota(mode):
    if mode == 'fresh':
        return MIN_PRELAUNCH_QUOTA_BYTES
    if mode == 'resume':
        return MIN_RESUME_QUOTA_BYTES
    if mode == 'formal':
        return MIN_FORMAL_QUOTA_BYTES
    raise RuntimeError(f'Unknown Insert stage: {mode}')


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def launch(mode, gpu_lock):
    require(gpu_lock is not None and not gpu_lock.closed, 'GPU reservation is not held')
    held, actual = os.fstat(gpu_lock.fileno()), GPU_LOCK.stat()
    require((held.st_dev, held.st_ino) == (actual.st_dev, actual.st_ino),
            'GPU reservation inode changed')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_preparation()
    require(predecessor_ready() is not None, 'Authorized early-stop handoff was lost')
    check_quota(live_quota(), minimum_quota(mode))
    output, status_path, script = _stage_paths(mode)
    require(not status_path.exists() and not output.exists(),
            f'{mode} stage/output already exists; no retry or reuse')
    with socket.socket() as port:
        port.bind(('127.0.0.1', PORT))
    command = [str(PYTHON), '-m', 'accelerate.commands.launch', '--config_file',
               str(SOURCE / 'scripts/accelerate_configs/accelerate_zero1_ds.yaml'),
               '--num_processes', '4', '--num_machines', '1', '--machine_rank', '0',
               '--main_process_ip', '127.0.0.1', '--main_process_port', str(PORT), str(script)]
    env = os.environ.copy()
    env.update(INSERT_ARM='B', INSERT_MODE=mode, EASYWAM_ROOT=str(SOURCE),
               PYTHONPATH=f'{SOURCE}/src:{SOURCE}:{ROOT}',
               CUDA_VISIBLE_DEVICES='0,1,2,3', NCCL_CUMEM_HOST_ENABLE=NCCL_CUMEM_HOST_ENABLE,
               PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONUNOPTIMIZED='1', PYTHONOPTIMIZE='',
               PYTHONPYCACHEPREFIX='/tmp/insert-after-drawer-pycache')
    for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'):
        env.pop(key, None)
    state = {'status': 'starting', 'mode': mode, 'arm': 'B', 'utc': now(),
             'output_dir': str(output), 'config_sha256': digest(CONFIG),
             'command': command, 'automatic_restart': False, 'launcher_pid': os.getpid(),
             'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE}}
    write_new(status_path, state)
    output.mkdir(parents=True)
    process = None
    try:
        with (output / 'train.log').open('x') as log:
            process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', training_pid=process.pid)
            atomic(status_path, state)
            queue_status(f'running_{mode}', mode=mode, training_pid=process.pid,
                         output_dir=str(output))
            code = process.wait()
        state.update(status='exited', exit_code=code, ended_at=now())
        atomic(status_path, state)
        require(code == 0, f'{mode} process exited {code}; no automatic retry')
    except BaseException as exc:
        if process is not None:
            terminate(process)
        state.update(status='failed', error=repr(exc), ended_at=now(),
                     exit_code=None if process is None else process.returncode)
        try:
            atomic(status_path, state)
        except BaseException:
            pass
        raise


def validate_formal_runtime():
    """Require one complete, fresh-start runtime proof from each final rank."""
    expected_names = [f'formal-runtime-rank-{rank}.json' for rank in range(4)]
    actual_names = sorted(path.name for path in FORMAL_OUTPUT.glob('formal-runtime-rank-*.json'))
    require(actual_names == expected_names,
            'Formal runtime proof set is not exactly the four final rank records')
    authorization = read(AUTHORIZATION)
    config_hash = digest(CONFIG)
    records = []
    pids = set()
    expected_measurements = list(range(1, 20001))
    expected_saves = list(range(2000, 20001, 2000))
    for rank in range(4):
        path = FORMAL_OUTPUT / f'formal-runtime-rank-{rank}.json'
        item = read(path)
        require(item['stage'] == 'formal' and item['rank'] == rank and
                isinstance(item['pid'], int) and item['pid'] > 0 and item['pid'] not in pids and
                item['config_sha256'] == config_hash and
                item['authorization_sha256'] == digest(AUTHORIZATION) and
                item['output_dir'] == str(FORMAL_OUTPUT) and
                item.get('runtime_env') == {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE} and
                item['start_step'] == 0 and item['final_step'] == 20000 and
                item['trainer_initial_step'] == item['engine_initial_step'] == 0 and
                item['engine_step'] == item['trainer_step'] == 20000 and
                item['world_size'] == 4 and item['batch_size'] == 4 and
                item['gradient_accumulation_steps'] == 4 and
                item['global_batch'] == 64 and item['zero_stage'] == 1 and
                item['grad_accum_dtype'] == 'torch.bfloat16' and
                item['scheduler_total_steps'] == 20000,
                f'Formal rank {rank} runtime recipe/start/lineage proof mismatch')
        pids.add(item['pid'])
        measurements = item['measurements']
        require([entry['step'] for entry in measurements] == expected_measurements and
                all(entry['engine_step'] == entry['step'] and
                    entry['loss_finite'] is True and math.isfinite(entry['loss']) and
                    math.isfinite(entry['global_grad_norm']) and
                    entry['microbatches'] == 4 and
                    int(entry['headroom_bytes']) >= MIN_HEADROOM_BYTES and
                    int(entry['total']) - int(entry['peak_allocated']) >= MIN_HEADROOM_BYTES
                    for entry in measurements),
                f'Formal rank {rank} update/headroom proof is incomplete')
        saves = item['save_peaks']
        require([entry['step'] for entry in saves] == expected_saves and
                all(int(entry['headroom_bytes']) >= MIN_HEADROOM_BYTES and
                    int(entry['total']) - int(entry['peak_allocated']) >= MIN_HEADROOM_BYTES
                    for entry in saves),
                f'Formal rank {rank} save/headroom proof is incomplete')
        eval_peaks = item['evaluation_peaks']
        require(isinstance(eval_peaks, list), f'Formal rank {rank} evaluation proof missing')
        if item['evaluation_expected']:
            require(eval_peaks and all(
                int(entry['headroom_bytes']) >= MIN_HEADROOM_BYTES and
                int(entry['total']) - int(entry['peak_allocated']) >= MIN_HEADROOM_BYTES
                for entry in eval_peaks),
                f'Formal rank {rank} evaluation headroom proof is incomplete')
        else:
            require(eval_peaks == [], f'Formal rank {rank} has unbound evaluation evidence')
        records.append({'rank': rank, 'pid': item['pid'],
                        'measurement_count': len(measurements),
                        'save_steps': [entry['step'] for entry in saves],
                        'evaluation_count': len(eval_peaks),
                        'sha256': digest(path)})
    require(len(pids) == 4, 'Formal runtime proof does not contain four distinct rank processes')
    gate_names = sorted(path.name for path in ROOT.glob('SAVE_GATE-formal-*.json'))
    expected_gates = [f'SAVE_GATE-formal-{step:06d}.json' for step in expected_saves]
    require(gate_names == expected_gates,
            'Formal save quota gate set is incomplete or contains an unexpected step')
    for name, step in zip(gate_names, expected_saves):
        gate = read(ROOT / name)
        require(gate['passed'] is True and gate['stage'] == 'formal' and
                gate['step'] == step and int(gate['minimum_free_bytes']) == MIN_SAVE_QUOTA_BYTES and
                gate['quota']['verified'] is True and
                int(gate['quota']['quota_free_bytes']) >= MIN_SAVE_QUOTA_BYTES,
                f'Formal save quota gate failed at step {step}')
    return {'passed': True, 'ranks': records, 'config_sha256': config_hash,
            'authorization_sha256': digest(AUTHORIZATION)}


def validate_formal():
    state = read(ROOT / 'STATUS-formal.json')
    require(state['status'] == 'exited' and state['exit_code'] == 0 and
            state['output_dir'] == str(FORMAL_OUTPUT) and
            state['automatic_restart'] is False and
            state.get('runtime_env') == {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
            'Formal Insert did not exit successfully')
    audit = checkpoint(FORMAL_OUTPUT, 20000)
    finite = finite_log(FORMAL_OUTPUT / 'train.log', 20000)
    state_dirs = sorted(path.name for path in (FORMAL_OUTPUT / 'checkpoints/state').glob('step_*'))
    weights = sorted(path.name for path in (FORMAL_OUTPUT / 'checkpoints/weights').glob('step_*.pt'))
    require(state_dirs == ['step_020000'], 'Formal state retention is not exactly one final state')
    require(weights == ['step_016000.pt', 'step_018000.pt', 'step_020000.pt'],
            'Formal weight retention is not exactly three final weights')
    runtime = validate_formal_runtime()
    return {'passed': True, 'stage': 'formal', 'exit_code': 0,
            'final_step': 20000, 'config_sha256': digest(CONFIG),
            'formal_output': str(FORMAL_OUTPUT), 'checkpoint': audit,
            'runtime': runtime,
            'finite_log': finite, 'early_stop_handoff_sha256': digest(HANDOFF),
            'old_insert_baseline_sha256': read(AUTHORIZATION)['old_insert_baseline_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only preparation/predecessor check')
    args = parser.parse_args(argv)
    if args.check:
        verify_preparation()
        evidence = predecessor_ready()
        print({'prelaunch_checks_passed': evidence is not None,
               'predecessor_ready': evidence is not None,
               'gpu_lock_acquired': False})
        return

    lock = None
    try:
        lock = (ROOT / 'queue.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        # Queue ownership/claim failures are not this attempt's failures.  In
        # particular, never overwrite a prior running/complete/blocked record.
        if lock is not None:
            lock.close()
        raise

    try:
        require(not (ROOT / 'QUEUE_STATUS.json').exists(),
                'Queue already attempted; manual investigation required')
        write_new(ROOT / 'QUEUE_STATUS.json', {'status': 'initializing', 'utc': now(),
                                               'queue_pid': os.getpid(),
                                               'automatic_restart': False})
    except BaseException:
        lock.close()
        raise

    def interrupted(signum, frame):
        raise RuntimeError(f'Successor controller interrupted by signal {signum}; no automatic retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    gpu_lock = None
    try:
        verify_preparation()
        predecessor = wait_for_predecessor()
        gpu_lock = wait_for_gpu_release_and_idle()
        verify_preparation()
        require(predecessor_ready() is not None, 'Authorized early-stop handoff changed')
        check_quota(live_quota(), MIN_PRELAUNCH_QUOTA_BYTES)
        write_new(ROOT / 'EARLY_STOP_HANDOFF_USED.json', {
            'verified': True, 'utc': now(), 'drawer_20000_complete': False,
            'handoff_sha256': digest(HANDOFF),
            'evidence': predecessor,
        })
        queue_status('running_preflight')
        launch('fresh', gpu_lock)
        fresh = validate_preflight_report(ROOT, 'fresh')
        queue_status('running_preflight_resume')
        launch('resume', gpu_lock)
        resumed = validate_preflight_report(ROOT, 'resume')
        write_new(ROOT / 'PREFLIGHT_ACCEPTANCE.json', {
            'passed': True, 'utc': now(), 'fresh': fresh, 'resume': resumed,
            'config_sha256': digest(CONFIG), 'formal_initialization': 'WanRobot fresh',
            'drawer_recovery_root': str(DRAWER_RECOVERY),
        })
        check_quota(live_quota(), MIN_FORMAL_QUOTA_BYTES)
        queue_status('running_formal', output_dir=str(FORMAL_OUTPUT))
        launch('formal', gpu_lock)
        completion = validate_formal()
        write_new(ROOT / 'COMPLETION_AUDIT.json', completion)
        queue_status('complete', final_step=20000, output_dir=str(FORMAL_OUTPUT), exit_code=0)
    except BaseException as exc:
        try:
            queue_status('blocked', error=repr(exc))
        finally:
            if gpu_lock is not None:
                gpu_lock.close()
            lock.close()
        raise
    else:
        if gpu_lock is not None:
            gpu_lock.close()
        lock.close()


if __name__ == '__main__':
    main()
