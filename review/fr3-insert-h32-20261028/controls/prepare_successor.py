"""Bind a new Insert attempt to the already sealed old preparation.

CPU-only.  This script writes only small control/configuration files; it never
copies source, data, statistics, caches, weights, or starts a GPU process.
"""
from pathlib import Path

from early_stop_handoff import HANDOFF, REASON, create_handoff, verify_early_stop

from insert_common import (AUTHORIZATION, CONFIG, DRAWER_RECOVERY, NCCL_CUMEM_HOST_ENABLE,
                           OLD_BASELINE_SHA256, OLD, REFERENCE, ROOT, SEALED, SOURCE, digest,
                           make_config, now, old_baseline_path, old_file, read, require,
                           validate_config, write_new)


def main():
    require(ROOT.is_dir(), 'New Insert control root must already exist')
    require(not SEALED.exists() and not CONFIG.exists() and not AUTHORIZATION.exists(),
            'New Insert preparation was already attempted; never overwrite it')
    create_handoff()
    baseline_path = old_baseline_path()
    require(baseline_path.is_file() and digest(baseline_path) == OLD_BASELINE_SHA256,
            'Old Insert B config hash changed or is unavailable')
    baseline = read(baseline_path)
    inputs_path = old_file('INPUTS.json')
    preparation_path = old_file('PREPARATION.json')
    audit_path = old_file('CPU_AUDIT.json')
    manifest_path = old_file('DATA_MANIFEST.json')
    stats_path = Path(baseline['data']['train']['pretrained_norm_stats'])
    cache_path = Path(baseline['data']['train']['text_embedding_cache_dir'])
    queue_path = old_file('QUEUE_STATUS.json')
    for path in (inputs_path, preparation_path, audit_path, manifest_path, stats_path,
                 queue_path, SOURCE):
        require(path.exists(), f'Missing sealed old Insert input: {path}')
    inputs = read(inputs_path)
    manifest = read(manifest_path)
    sealed = {
        'old_root': str(OLD),
        'old_baseline_config_path': str(baseline_path),
        'old_baseline_config_sha256': digest(baseline_path),
        'old_formal_output': str(baseline['output_dir']),
        'old_preparation_path': str(preparation_path),
        'old_preparation_sha256': digest(preparation_path),
        'old_inputs_path': str(inputs_path),
        'old_inputs_sha256': digest(inputs_path),
        'old_cpu_audit_path': str(audit_path),
        'old_cpu_audit_sha256': digest(audit_path),
        'old_data_manifest_path': str(manifest_path),
        'old_data_manifest_sha256': digest(manifest_path),
        'old_queue_status_path': str(queue_path),
        'old_queue_status_sha256': digest(queue_path),
        'source_root': str(SOURCE),
        'data_root': str(manifest['dataset']),
        'stats_path': str(stats_path),
        'stats_sha256': digest(stats_path),
        'text_cache_root': str(cache_path),
        'episodes': manifest['episodes'],
        'frames': manifest['frames'],
        'excluded_original_merged_episodes': manifest['excluded_original_merged_episodes'],
        'source_file_count': len(inputs['source']),
        'dataset_file_count': len(manifest['files']),
        'reuse_policy': 'identity-bound reuse of sealed old source/data/stats/cache; no re-audit',
    }
    write_new(SEALED, sealed)
    cfg = make_config(baseline)
    validate_config(baseline, cfg)
    ROOT.joinpath('B').mkdir(exist_ok=False)
    write_new(REFERENCE, baseline)
    write_new(CONFIG, cfg)
    write_new(AUTHORIZATION, {
        'approved': True,
        'automatic_retry': False,
        'utc': now(),
        'request': 'User explicitly stopped Drawer early, keeping only state10000; start fresh Insert with the same approved recipe.',
        'predecessor': str(DRAWER_RECOVERY),
        'predecessor_reason': REASON,
        'drawer_20000_complete': False,
        'early_stop_handoff_sha256': digest(HANDOFF),
        'initialization': 'WanRobot fresh',
        'resume': None,
        'formal_steps': 20000,
        'world_size': 4,
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'global_batch': 64,
        'horizon': [33, 9, 32],
        'zero_stage': 1,
        'activation_checkpointing': False,
        'preflight_fresh_steps': 3,
        'preflight_resume_from': 3,
        'preflight_final_step': 4,
        'dataset': sealed['data_root'],
        'episodes': sealed['episodes'],
        'frames': sealed['frames'],
        'excluded_original_merged_episodes': sealed['excluded_original_merged_episodes'],
        'old_insert_baseline_sha256': OLD_BASELINE_SHA256,
        'drawer_recovery_baseline_sha256': '56430374b196aa0fd96a2505a6b902c8bb5a68606c9a5f0dd601a8393c5529b8',
        'runtime_environment': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
        'config_sha256': digest(CONFIG),
        'sealed_inputs_sha256': digest(SEALED),
    })
    verify_early_stop()
    print(f'INSERT_SUCCESSOR_PREPARED config={digest(CONFIG)}', flush=True)


if __name__ == '__main__':
    main()
