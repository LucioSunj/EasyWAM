"""Explicit early-stop authorization, distinct from successful Drawer completion."""
from pathlib import Path

from insert_common import (AUTHORIZATION, BASE, DRAWER_RECOVERY, ROOT, RUNS,
                           checked_identity, digest, identity, now, read, require, write_new)

AUDIT = BASE / 'storage_cleanup/drawer-stop10000-20261028T075100Z'
KEEP = RUNS / ('fr3_drawer_fastwam_idm_fullfix_h32_4gpu_b4a4_z1_20261027_recovery2000_ncclfix/'
               'checkpoints/state/step_010000')
HANDOFF = ROOT / 'EARLY_STOP_HANDOFF.json'
REASON = 'user_requested_early_stop'
BOOT_PATH = Path('/proc/sys/kernel/random/boot_id')
CONTINUATION_SHA256 = '2361864f42bc48659966cb65e28829102ca94492152b1e2e96b485a752a1f21d'
MEMBERS = {'latest', 'trainer_state.json', 'zero_to_fp32.py',
           'pytorch_model/mp_rank_00_model_states.pt',
           *(f'pytorch_model/bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(4)),
           *(f'random_states_{r}.pkl' for r in range(4))}
AUDIT_NAMES = ('AUTHORIZATION.json', 'STOPPED.json', 'KEEPER_INTEGRITY.json',
               'KEEPER_CRC_CACHE.json', 'CLEANUP_AUTHORIZATION.json',
               'CLEANUP_MANIFEST.json', 'CLEANUP_DELETED.jsonl',
               'CLEANUP_COMPLETION.json', 'CLEANUP_POSTCHECK.json')


def evidence_paths():
    return [*(AUDIT / name for name in AUDIT_NAMES),
            *(DRAWER_RECOVERY / name for name in
              ('PREPARATION.json', 'AUTHORIZATION.json', 'config-continuation.json')),
            *(DRAWER_RECOVERY / f'saved-continuation-010000-rank-{r}.json' for r in range(4))]


def process(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()
        return {'state': fields[0], 'start_ticks': fields[19]}
    except FileNotFoundError:
        return None


def safe_regular(path):
    require(path.is_file(), f'Missing handoff evidence: {path}')
    require(not any(parent.is_symlink() for parent in (path, *path.parents)),
            f'Symlink in handoff evidence: {path}')


def create_handoff():
    stopped = read(AUDIT / 'STOPPED.json')
    steps = [item['step'] for item in stopped['last_updates']]
    require(len(steps) == 4 and len(set(steps)) == 1 and 10000 <= steps[0] < 20000,
            'Invalid actual Drawer stopping step')
    paths = evidence_paths()
    for path in paths:
        safe_regular(path)
    write_new(HANDOFF, {'schema': 1, 'utc': now(), 'reason': REASON,
        'drawer_20000_complete': False, 'original_target_step': 20000,
        'actual_stop_step': steps[0], 'keep_step': 10000, 'keep_state': str(KEEP),
        'evidence_hashes': {str(path): digest(path) for path in paths}})


def verify_early_stop():
    safe_regular(HANDOFF)
    authorization = read(AUTHORIZATION)
    require(authorization.get('approved') is True and authorization.get('automatic_retry') is False
            and authorization.get('predecessor_reason') == REASON
            and authorization.get('early_stop_handoff_sha256') == digest(HANDOFF),
            'Missing explicit authorization bound to this early-stop handoff')
    handoff = read(HANDOFF)
    require(handoff.get('schema') == 1 and handoff.get('reason') == REASON
            and handoff.get('drawer_20000_complete') is False
            and handoff.get('original_target_step') == 20000
            and handoff.get('keep_step') == 10000 and handoff.get('keep_state') == str(KEEP),
            'Wrong early-stop reason, step, or keeper path')
    paths = evidence_paths()
    require(set(handoff['evidence_hashes']) == {str(path) for path in paths}, 'Handoff evidence set changed')
    for path in paths:
        safe_regular(path)
        require(digest(path) == handoff['evidence_hashes'][str(path)], f'Handoff evidence changed: {path}')
    request = read(AUDIT / 'AUTHORIZATION.json')
    require(request['approved'] is True and request['reason'] == REASON
            and request['keep_step'] == 10000 and request['keep_state'] == str(KEEP)
            and request['drawer_20000_complete'] is False and request['automatic_restart'] is False,
            'Drawer stop was not explicitly authorized')
    require(BOOT_PATH.read_text().strip() == request['boot_id'], 'Host boot changed after early stop')
    stopped = read(AUDIT / 'STOPPED.json')
    require(stopped['passed'] is True and stopped['status'] == REASON
            and stopped['drawer_20000_complete'] is False and stopped['keep_step'] == 10000
            and stopped['all_target_processes_inactive'] is True,
            'Missing verified Drawer early stop')
    steps = [item['step'] for item in stopped['last_updates']]
    require(steps == [handoff['actual_stop_step']] * 4 and 10000 <= steps[0] < 20000,
            'Stopping step evidence changed')
    require(set(stopped['original_processes']) == {str(pid) for pid in request['stop_pids']},
            'Stopped process identity set changed')
    for pid, expected in stopped['original_processes'].items():
        item = process(int(pid))
        require(item is None or (item['state'] == 'Z' and item['start_ticks'] == expected['start_ticks']),
                f'Old process is active or PID was reused: {pid}')
    report = read(AUDIT / 'KEEPER_INTEGRITY.json')
    require(report['passed'] is True and report['archives_checked'] == 9
            and report['cache_sha256'] == digest(AUDIT / 'KEEPER_CRC_CACHE.json'),
            'Keeper full CRC evidence is missing or changed')
    checkpoint = report['checkpoint']
    require(checkpoint['step'] == 10000 and checkpoint['state_dir'] == str(KEEP), 'Wrong keeper audit')
    require(KEEP.is_dir() and not KEEP.is_symlink(), 'Keeper directory missing or symlinked')
    children = list(KEEP.rglob('*'))
    require(not any(path.is_symlink() for path in children), 'Keeper contains symlink')
    actual = {str(path.relative_to(KEEP)) for path in children if path.is_file()}
    require(actual == MEMBERS and set(checkpoint['files']) == {str(KEEP / name) for name in MEMBERS},
            'Keeper member set incomplete or unexpected')
    marker = read(KEEP / 'trainer_state.json')
    require(marker == checkpoint['marker'] == stopped['keeper']['marker']
            and marker['global_step'] == marker['deepspeed_global_step'] == 10000
            and (marker['num_processes'], marker['batch_size'], marker['gradient_accumulation_steps']) == (4, 4, 4)
            and (KEEP / 'latest').read_text().strip() == 'pytorch_model', 'Keeper marker mismatch')
    cache = read(AUDIT / 'KEEPER_CRC_CACHE.json')
    for name, record in checkpoint['files'].items():
        path = Path(name)
        safe_regular(path)
        require(identity(path) == checked_identity(record['identity']), f'Keeper identity changed: {path}')
        require(record['identity'] == stopped['keeper']['files'][str(path.relative_to(KEEP))],
                'Keeper no longer matches stopped state')
        if path.suffix in ('.pt', '.pkl'):
            require(record.get('crc_passed') is True and name in cache
                    and checked_identity(cache[name]) == identity(path), 'Keeper CRC cache not bound to identity')
        else:
            require(record.get('sha256') == digest(path), 'Keeper metadata hash mismatch')
    require(digest(DRAWER_RECOVERY / 'config-continuation.json') == CONTINUATION_SHA256,
            'Drawer continuation configuration changed')
    for rank in range(4):
        receipt = read(DRAWER_RECOVERY / f'saved-continuation-010000-rank-{rank}.json')
        require(receipt == stopped['keeper']['save_receipts'][rank]
                and receipt['rank'] == rank and receipt['global_step'] == 10000
                and receipt['state_path'] == str(KEEP) and receipt['scheduler_total_steps'] == 20000
                and receipt['config_sha256'] == CONTINUATION_SHA256, 'Keeper save receipt changed')
    cleanup = read(AUDIT / 'CLEANUP_COMPLETION.json')
    post = read(AUDIT / 'CLEANUP_POSTCHECK.json')
    require(cleanup['passed'] is True and cleanup['deleted_files'] == 30
            and cleanup['keep_step'] == 10000 and cleanup['keep_state'] == str(KEEP)
            and cleanup['drawer_20000_complete'] is False
            and cleanup['manifest_sha256'] == digest(AUDIT / 'CLEANUP_MANIFEST.json')
            and cleanup['postcheck_sha256'] == digest(AUDIT / 'CLEANUP_POSTCHECK.json')
            and cleanup['keeper_integrity_sha256'] == digest(AUDIT / 'KEEPER_INTEGRITY.json')
            and post['passed'] is True and post['keeper_unchanged'] is True
            and post['deleted_files_absent'] is True and post['protected_files_unchanged'] is True,
            'Cleanup completion is not bound to the preserved keeper')
    require(all(not Path(row['path']).exists() for row in read(AUDIT / 'CLEANUP_MANIFEST.json')),
            'Retired Drawer payload reappeared')
    return {'verified': True, 'reason': REASON, 'drawer_20000_complete': False,
            'actual_stop_step': handoff['actual_stop_step'], 'keep_step': 10000,
            'keep_state': str(KEEP), 'handoff_sha256': digest(HANDOFF)}
