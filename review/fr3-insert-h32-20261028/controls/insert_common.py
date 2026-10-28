"""CPU-only contracts for Insert following an explicitly authorized early stop.

This attempt deliberately reuses the sealed Insert source/data/statistics from
the older preparation root.  It never copies those large inputs and never
accepts the old Drawer-comparison completion as a predecessor.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import zipfile

if sys.flags.optimize:
    raise RuntimeError('Insert safety checks require an unoptimized Python interpreter')

BASE = Path('/path/to/storage/SJ')
ROOT = BASE / 'training_launch/fr3-insert-tubes-after-drawer10000-20261028'
HERE = ROOT
OLD = BASE / 'training_launch/fr3-insert-tubes-fullfix-h32-20261027'
DRAWER_RECOVERY = BASE / 'training_launch/fr3-drawer-recovery-2000-ncclfix-20261027'
RUNS = BASE / 'Outer/EasyWAM/runs/real_world_fr3_fastwam_idm_wan22'
RUN_NAME = 'fr3_insert_tubes_after_drawer10000_h32_4gpu_b4a4_z1_20261028'
FORMAL_OUTPUT = RUNS / RUN_NAME
SOURCE = OLD / 'candidate'
PYTHON = BASE / 'Outer/EasyWAM/.venv/bin/python'
GPU_LOCK = BASE / 'training_launch/gpu-0-3-reservation.lock'
CONFIG = ROOT / 'B/train.json'
REFERENCE = ROOT / 'reference_baseline.json'
ARM_DIR = ROOT / 'B'
PREPARATION = ROOT / 'PREPARATION.json'
SEALED = ROOT / 'SEALED_INPUTS.json'
AUTHORIZATION = ROOT / 'AUTHORIZATION.json'
ARTIFACT_INTEGRITY = ROOT / 'ARTIFACT_INTEGRITY.json'
OLD_BASELINE_SHA256 = '042de3eca694042d45c2aa154f01cc2f230470cfcc7f7077749350930f6292bc'
DRAWER_BASELINE_SHA256 = '56430374b196aa0fd96a2505a6b902c8bb5a68606c9a5f0dd601a8393c5529b8'
NCCL_CUMEM_HOST_ENABLE = '0'
GIB = 1024**3
MIN_PRELAUNCH_QUOTA_BYTES = 405 * GIB
MIN_RESUME_QUOTA_BYTES = 314 * GIB
MIN_FORMAL_QUOTA_BYTES = 223 * GIB
MIN_SAVE_QUOTA_BYTES = 108 * GIB
MIN_HEADROOM_BYTES = 4 * GIB
STOP = 4
ARMS = {'B': {'batch': 4, 'gas': 4, 'zero': 1, 'ac': False, 'suffix': 'b4a4_z1'}}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_nccl_environment(environment=None):
    environment = os.environ if environment is None else environment
    require(environment.get('NCCL_CUMEM_HOST_ENABLE') == NCCL_CUMEM_HOST_ENABLE,
            'NCCL_CUMEM_HOST_ENABLE=0 is mandatory for this Insert attempt')
    return NCCL_CUMEM_HOST_ENABLE


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return [str(v) for v in (stat.st_dev, stat.st_ino, stat.st_size,
                             stat.st_mtime_ns, stat.st_ctime_ns)]


def checked_identity(value):
    require(isinstance(value, list) and len(value) == 5,
            'Identity requires five fields')
    require(all(isinstance(item, str) and item.isdecimal() for item in value),
            'Identity fields must be decimal strings')
    return value


def old_file(name):
    """Support the local evidence snapshot while preferring the remote root layout."""
    path = OLD / name
    if path.exists():
        return path
    evidence = OLD / 'evidence' / name
    return evidence


def old_baseline_path():
    path = OLD / 'B/train.json'
    return path if path.exists() else OLD / 'evidence/train.json'


def make_config(baseline):
    cfg = json.loads(json.dumps(baseline))
    cfg['output_dir'] = str(FORMAL_OUTPUT)
    cfg['wandb']['name'] = RUN_NAME
    cfg['resume'] = None
    return cfg


def validate_config(baseline, cfg):
    require(cfg == make_config(baseline),
            'Only the Insert formal output/name may differ from the sealed B recipe')
    require((cfg['batch_size'], cfg['gradient_accumulation_steps'], cfg['mixed_precision'])
            == (4, 4, 'bf16'), 'Insert requires B4/GAS4/BF16')
    require(cfg['max_steps'] == 20000 and cfg['warmup_ratio'] == 0.05,
            'Insert requires the absolute 20000-step horizon and 1000-step warmup')
    require(cfg['resume'] is None, 'Formal Insert must initialize WanRobot fresh')
    require((cfg['checkpoint_save_limit'], cfg['state_checkpoint_save_limit']) == (3, 1),
            'Insert requires three weights and one rolling state')
    data = cfg['data']['train']
    require((data['num_frames'], data['processor']['num_obs_steps'],
             data['action_video_freq_ratio'], data['global_sample_stride']) == (33, 33, 4, 1),
            'Insert H32 / nine-video-frame / stride-one contract changed')
    processor = data['processor']
    require(processor['action_clip_range'] is None and processor['state_clip_range'] is None,
            'Clipping must remain disabled for the fixed recipe')
    require(processor['require_raw_state_anchor'] is True,
            'Raw-state anchor contract changed')
    require(processor['binary_dims'] == {'action': {'default': [7]}, 'state': {'default': [7]}},
            'Binary gripper contract changed')
    for options in (cfg['model']['backbone'], cfg['model']['backbone']['dit_config'],
                    cfg['model']['action_dit_config']):
        require(options['use_gradient_checkpointing'] is False,
                'Activation checkpointing changed')


def global_slots(batch, gas, rank, micro):
    require(batch * gas * 4 == 64 and 0 <= rank < 4 and 0 <= micro < gas,
            'Invalid B4/GAS4 slot contract')
    start = (micro * 4 + rank) * batch
    return list(range(start, start + batch))


def stress_kind(slot, step=1):
    kind = (slot + (step - 1) * 3) % 8
    return kind if kind < 3 else None


def input_hashes():
    paths = [ROOT / name for name in (
        'insert_common.py', 'queue_successor.py', 'preflight_four_gpu.py',
        'formal_train.py', 'seal_preparation.py', 'prepare_successor.py',
        'monitor.py', 'test_successor.py', 'run_queue.sh', 'run_cpu_tests.sh',
        'run_cpu_tests.py', 'run_check.sh',
        'early_stop_handoff.py', 'test_early_stop.py', 'EARLY_STOP_HANDOFF.json',
        'README.md', 'B/train.json', 'AUTHORIZATION.json', 'SEALED_INPUTS.json',
        'reference_baseline.json')]
    return {str(path): digest(path) for path in sorted(paths) if path.is_file()}


def verify_sealed_inputs():
    """Recheck the old preparation by identity; do not recalculate large audits."""
    sealed = read(SEALED)
    require(sealed['old_baseline_config_sha256'] == OLD_BASELINE_SHA256,
            'Old Insert baseline hash is not the sealed B recipe')
    for key in ('old_preparation', 'old_inputs', 'old_cpu_audit', 'old_data_manifest',
                'stats', 'old_queue_status'):
        path = Path(sealed[f'{key}_path'])
        require(path.is_file() and digest(path) == sealed[f'{key}_sha256'],
                f'Sealed old input changed: {path}')
    baseline_path = Path(sealed['old_baseline_config_path'])
    require(digest(baseline_path) == OLD_BASELINE_SHA256,
            'Old Insert baseline config changed')
    baseline = read(baseline_path)
    require(baseline['resume'] is None and baseline['output_dir'] == sealed['old_formal_output'],
            'Old Insert baseline is not the original fresh recipe')
    preparation = read(sealed['old_preparation_path'])
    require(preparation['prepared'] is True, 'Old Insert preparation is not sealed')
    for path, expected in preparation['control_hashes'].items():
        require(digest(path) == expected, f'Old sealed control changed: {path}')
    old_queue = read(sealed['old_queue_status_path'])
    require(old_queue['status'] == 'blocked',
            'Old Insert queue is not preserved as blocked evidence')

    inputs = read(sealed['old_inputs_path'])
    source = Path(sealed['source_root'])
    require(source.is_dir() and not source.is_symlink(), 'Sealed source root missing/symlinked')
    actual_source = {str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()}
    require(actual_source == set(inputs['source']), 'Sealed source file set changed')
    for relative, record in inputs['source'].items():
        require(identity(source / relative) == [str(v) for v in record['identity']],
                f'Sealed source identity changed: {relative}')

    for section in ('assets', 'text_cache'):
        for path, record in inputs[section].items():
            require(identity(path) == [str(v) for v in record['identity']],
                    f'Sealed {section} identity changed: {path}')

    manifest = read(sealed['old_data_manifest_path'])
    data = Path(sealed['data_root'])
    require(data.is_dir() and not data.is_symlink(), 'Sealed data root missing/symlinked')
    require((manifest['episodes'], manifest['frames']) == (299, 132292) and
            manifest['excluded_original_merged_episodes'] == [250, 285],
            'Sealed Insert dataset counts/exclusions changed')
    actual_data = {str(path.relative_to(data)) for path in data.rglob('*') if path.is_file()}
    require(actual_data == set(manifest['files']), 'Sealed dataset file set changed')
    for relative, record in manifest['files'].items():
        require(identity(data / relative) == [str(v) for v in record['identity']],
                f'Sealed dataset identity changed: {relative}')

    audit = read(sealed['old_cpu_audit_path'])
    require(audit['passed'] is True and audit['stats_sha256'] == sealed['stats_sha256'] and
            audit['config_sha256'] == OLD_BASELINE_SHA256,
            'Old CPU audit is not bound to the sealed B config/stats')
    require(digest(sealed['stats_path']) == sealed['stats_sha256'],
            'Sealed H32 statistics changed')
    return sealed


def live_quota():
    """Read native Lustre user/group quota; project quota remains explicit unknown."""
    group_id = 13979
    require(group_id in {*os.getgroups(), os.getgid()},
            'Process is not in the enforced quota group')
    require(RUNS.stat().st_gid == group_id and ROOT.stat().st_gid == group_id,
            'Output/control group does not match the quota account')
    records = {}
    available = []
    descriptor = os.open(str(BASE.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name, quota_type, account in (('user', 0, os.getuid()), ('group', 1, group_id)):
            payload = bytearray(176)
            struct.pack_into('<6I', payload, 0, 0x800007, quota_type, account, 0, 0, 0)
            fcntl.ioctl(descriptor, 0xc0b066a2, payload, True)
            values = struct.unpack_from('<8Q2I', payload, 48)
            hard, soft, used, valid = values[0], values[1], values[2], values[8]
            require(valid & 3 == 3, f'Incomplete native {name} quota limits/space')
            limits = [value * 1024 for value in (hard, soft) if value > 0]
            if name == 'group':
                require(hard > 0 and soft > 0,
                        'Expected finite shared group quota')
            free = None if not limits else min(limits) - used
            if free is not None:
                available.append(free)
            records[name] = {'type': quota_type, 'id': account, 'valid': valid,
                             'soft_kib': str(soft), 'hard_kib': str(hard),
                             'used_bytes': str(used),
                             'quota_free_bytes': None if free is None else str(free)}
    finally:
        os.close(descriptor)
    require(available, 'No authoritative finite quota limit was obtained')
    return {'observed_at': now(), 'verified': True,
            'authority': 'Lustre Q_GETQUOTA LL_IOC_QUOTACTL',
            'quota_scope': 'group:13979 plus any stricter finite user limit',
            'project_quota': 'unknown_fsxattr_ENOTTY',
            'quota_free_bytes': str(min(available)), **records}


def check_quota(quota, minimum):
    require(quota['verified'] is True and int(quota['quota_free_bytes']) >= minimum,
            f'Quota budget below {minimum / GIB:.0f} GiB')


def check_save_quota(quota):
    check_quota(quota, MIN_SAVE_QUOTA_BYTES)


def check_headroom(total, peak, phase):
    total, peak = int(total), int(peak)
    remaining = total - peak
    require(remaining >= MIN_HEADROOM_BYTES,
            f'{phase} allocated-memory headroom below 4 GiB')
    return {'total': total, 'peak_allocated': peak,
            'headroom_bytes': remaining, 'phase': phase}


def distributed_save_gate(accelerator, stage, step):
    """Rank-zero native quota check, broadcast before any checkpoint write."""
    import torch

    payload = [None]
    if accelerator.is_main_process:
        result = {'passed': False, 'stage': stage, 'step': int(step), 'utc': now(),
                  'minimum_free_bytes': str(MIN_SAVE_QUOTA_BYTES)}
        try:
            result['quota'] = live_quota()
            check_save_quota(result['quota'])
            result['passed'] = True
            write_new(ROOT / f'SAVE_GATE-{stage}-{int(step):06d}.json', result)
        except BaseException as exc:
            result.update(passed=False, error=repr(exc))
        payload[0] = result
    require(torch.distributed.is_initialized(),
            'Distributed process group is required for the checkpoint save gate')
    torch.distributed.broadcast_object_list(payload, src=0, device=accelerator.device)
    require(payload[0] is not None and payload[0]['passed'] is True,
            f'All-rank checkpoint save gate failed: {payload[0]}')
    return payload[0]


def archive_identity(path, cache):
    path = Path(path)
    before = identity(path)
    previous = cache.get(str(path))
    if previous is not None:
        require(checked_identity(previous) == before,
                f'Artifact changed after CRC audit: {path}')
    else:
        with zipfile.ZipFile(path) as archive:
            require(any('/data/' in name for name in archive.namelist()),
                    f'Archive has no tensor storage: {path}')
            require(archive.testzip() is None, f'Archive CRC failure: {path}')
        require(identity(path) == before, f'Archive changed during CRC audit: {path}')
        cache[str(path)] = before
    return before


def checkpoint(run, step, *, require_comparison_sidecars=False):
    """CRC and bind one completed checkpoint without loading tensors."""
    run = Path(run)
    state = run / f'checkpoints/state/step_{step:06d}'
    latest_path = state / 'latest'
    marker_path = state / 'trainer_state.json'
    require(latest_path.is_file() and not latest_path.is_symlink(),
            f'Missing checkpoint tag: {latest_path}')
    require(latest_path.read_text().strip() == 'pytorch_model',
            f'Checkpoint tag is not pytorch_model: {latest_path}')
    require(marker_path.is_file() and not marker_path.is_symlink(),
            f'Missing checkpoint marker: {marker_path}')
    marker = read(marker_path)
    require(marker['global_step'] == marker['deepspeed_global_step'] == step and
            (marker['num_processes'], marker['batch_size'], marker['gradient_accumulation_steps'])
            == (4, 4, 4) and isinstance(marker['epoch'], int) and marker['epoch'] >= 0 and
            isinstance(marker['batch_in_epoch'], int) and marker['batch_in_epoch'] >= 0,
            'Checkpoint marker recipe/cursor mismatch')
    optimizers = sorted(state.rglob('*optim_states.pt'))
    models = list(state.rglob('*model_states.pt'))
    require(len(optimizers) == 4 and len(models) == 1 and
            models[0].name == 'mp_rank_00_model_states.pt' and
            {p.name for p in optimizers} == {
                f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt' for rank in range(4)},
            'Checkpoint lacks the four expected optimizer shards/model shard')
    model_dir = state / 'pytorch_model'
    require(model_dir.is_dir() and
            all(path.parent == model_dir for path in [*optimizers, *models]),
            'Checkpoint model/optimizer shards must be directly under pytorch_model')
    random_states = [state / f'random_states_{rank}.pkl' for rank in range(4)]
    sidecars = [state / f'comparison-rank-{rank}.json' for rank in range(4)]
    required_state = {latest_path, marker_path, *optimizers, *models, *random_states}
    optional_state = {state / 'zero_to_fp32.py'} if (state / 'zero_to_fp32.py').exists() else set()
    expected_state = required_state | optional_state
    if require_comparison_sidecars:
        expected_state |= set(sidecars)
    actual_state = {path for path in state.rglob('*') if path.is_file()}
    require(actual_state == expected_state,
            'Checkpoint state member set differs from the authorized DeepSpeed layout')
    if require_comparison_sidecars:
        require(all(path.is_file() and not path.is_symlink() for path in sidecars),
                'Preflight checkpoint lacks all four comparison-rank sidecars')
    weight = run / f'checkpoints/weights/step_{step:06d}.pt'
    artifacts = [weight, *optimizers, *models, *random_states]
    cache = read(ARTIFACT_INTEGRITY) if ARTIFACT_INTEGRITY.exists() else {}
    records = {}
    for path in artifacts:
        require(path.is_file() and not path.is_symlink(), f'Missing checkpoint artifact: {path}')
        records[str(path)] = {'identity': archive_identity(path, cache), 'crc_passed': True}
    metadata = [latest_path, marker_path]
    if (state / 'zero_to_fp32.py').exists():
        metadata.append(state / 'zero_to_fp32.py')
    if require_comparison_sidecars:
        metadata.extend(sidecars)
    for path in metadata:
        before = identity(path)
        sha256 = digest(path)
        require(identity(path) == before, f'Checkpoint metadata changed during audit: {path}')
        records[str(path)] = {'identity': before, 'sha256': sha256}
    atomic(ARTIFACT_INTEGRITY, cache)
    return {'step': step, 'state_dir': str(state), 'marker': marker,
            'files': records}


def finite_log(log, final_step):
    text = Path(log).read_text(errors='replace')
    updates = []
    for line in text.splitlines():
        if 'Training:' not in line:
            continue
        marker = line.split('/', 1)
        if len(marker) != 2:
            continue
        try:
            step = int(marker[0].rsplit(' ', 1)[-1])
            loss = float(line.split('loss=', 1)[1].split(',', 1)[0])
            grad = float(line.split('grad_norm=', 1)[1].split(',', 1)[0].split(']', 1)[0])
        except (ValueError, IndexError):
            continue
        require(math.isfinite(loss) and math.isfinite(grad),
                f'Nonfinite terminal log value: {line}')
        updates.append((step, loss, grad))
    require(updates and max(item[0] for item in updates) == final_step,
            'Missing terminal finite update evidence')
    return {'logged_updates': len(updates), 'last_step': updates[-1][0],
            'last_loss': updates[-1][1], 'last_grad_norm': updates[-1][2]}


def validate_preflight_report(root, mode):
    output_dir = ARM_DIR / mode
    report_dir = ARM_DIR
    status = read(ARM_DIR / f'STATUS-{mode}.json')
    require(status['status'] == 'exited' and status['exit_code'] == 0 and
            status['mode'] == mode and status['arm'] == 'B' and
            status['automatic_restart'] is False and
            status['output_dir'] == str(output_dir) and
            status['config_sha256'] == digest(CONFIG) and
            status.get('runtime_env') == {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
            f'{mode} preflight did not exit cleanly')
    expected = (0, 3) if mode == 'fresh' else (3, 4)
    reports = []
    pids = set()
    for rank in range(4):
        item = read(report_dir / f'acceptance-{mode}-rank-{rank}.json')
        require(item['mode'] == mode and item['arm'] == 'B' and
                item['rank'] == rank and isinstance(item['pid'], int) and item['pid'] > 0 and
                item['pid'] not in pids and
                item['output_dir'] == str(output_dir) and
                item['config_sha256'] == digest(CONFIG) and
                item.get('runtime_env') == {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE} and
                item['spec'] == ARMS['B'],
                f'{mode} rank {rank} identity/config contract mismatch')
        pids.add(item['pid'])
        require((item['start_step'], item['final_step'], item['global_batch'])
                == (*expected, 64), f'{mode} rank {rank} range mismatch')
        require(item['grad_accum_dtype'] == 'torch.bfloat16' and
                item['scheduler_total_steps'] == 20000 and
                item['gradients'] and
                all(v['finite'] and v['nonzero'] for v in item['gradients'].values()),
                f'{mode} rank {rank} runtime recipe/gradient proof missing')
        require([entry['step'] for entry in item['measurements']]
                == list(range(expected[0] + 1, expected[1] + 1)),
                f'{mode} rank {rank} update sequence mismatch')
        require(all(math.isfinite(entry['global_grad_norm']) for entry in item['measurements']),
                f'{mode} rank {rank} nonfinite update')
        require(all(int(entry['headroom_bytes']) >= MIN_HEADROOM_BYTES and
                    int(entry['total']) - int(entry['peak_allocated']) >= MIN_HEADROOM_BYTES
                    for entry in item['measurements']),
                f'{mode} rank {rank} measurement headroom proof missing')
        require(item['save_peaks'] and all(
                    int(entry['headroom_bytes']) >= MIN_HEADROOM_BYTES and
                    int(entry['total']) - int(entry['peak_allocated']) >= MIN_HEADROOM_BYTES
                    for entry in item['save_peaks']),
                f'{mode} rank {rank} save headroom proof missing')
        gate = read(ROOT / f'SAVE_GATE-{mode}-{expected[1]:06d}.json')
        require(gate['passed'] is True and gate['stage'] == mode and
                gate['step'] == expected[1] and
                int(gate['minimum_free_bytes']) == MIN_SAVE_QUOTA_BYTES and
                gate['quota']['verified'] is True and
                int(gate['quota']['quota_free_bytes']) >= MIN_SAVE_QUOTA_BYTES,
                f'{mode} rank {rank} save quota gate missing')
        if mode == 'resume':
            require(item['resume_verified'] is True and item['evaluation'],
                    f'{mode} rank {rank} missing restore/eval proof')
            require(item['evaluation_peak'] and
                    int(item['evaluation_peak']['headroom_bytes']) >= MIN_HEADROOM_BYTES and
                    int(item['evaluation_peak']['total']) -
                    int(item['evaluation_peak']['peak_allocated']) >= MIN_HEADROOM_BYTES,
                    f'{mode} rank {rank} evaluation headroom proof missing')
            restored = read(output_dir / f'restored-step4-rank-{rank}.json')
            require(restored['trainer_next_step'] == restored['engine_step'] == 4 and
                    restored['resume_verified'] is True and restored['rank'] == rank and
                    isinstance(restored['pid'], int) and restored['pid'] > 0 and
                    restored['pid'] == item['pid'] and
                    restored['output_dir'] == str(output_dir) and
                    restored['config_sha256'] == digest(CONFIG),
                    f'{mode} rank {rank} missing step-4 proof')
        reports.append(item)
    require(len(pids) == 4, f'{mode} preflight does not have four distinct rank processes')
    checkpoint_info = checkpoint(output_dir, expected[1], require_comparison_sidecars=True)
    return {'mode': mode, 'reports': reports, 'checkpoint': checkpoint_info}
