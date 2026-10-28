"""Seal the new Insert controls after CPU-only tests; never launch training."""
from early_stop_handoff import verify_early_stop
from insert_common import (AUTHORIZATION, CONFIG, NCCL_CUMEM_HOST_ENABLE, ROOT, SEALED,
                           digest, input_hashes, now, read, require,
                           verify_sealed_inputs, write_new)


def main():
    require(read(AUTHORIZATION)['approved'] is True and
            read(AUTHORIZATION)['automatic_retry'] is False,
            'Missing explicit one-shot Insert authorization')
    tests = read(ROOT / 'CPU_TEST_RESULTS.json')
    require(tests['passed'] is True and tests['input_hashes'] == input_hashes(),
            'CPU evidence does not bind current successor controls')
    sealed = verify_sealed_inputs()
    handoff = verify_early_stop()
    config_hash = digest(CONFIG)
    authorization = read(AUTHORIZATION)
    require(authorization['config_sha256'] == config_hash and
            authorization['sealed_inputs_sha256'] == digest(SEALED),
            'Authorization is not bound to successor config/sealed inputs')
    controls = [ROOT / name for name in (
        'insert_common.py', 'queue_successor.py', 'preflight_four_gpu.py',
        'formal_train.py', 'seal_preparation.py', 'prepare_successor.py',
        'monitor.py', 'test_successor.py', 'run_queue.sh', 'run_cpu_tests.sh',
        'run_cpu_tests.py', 'run_check.sh',
        'early_stop_handoff.py', 'test_early_stop.py', 'EARLY_STOP_HANDOFF.json',
        'README.md', 'B/train.json', 'AUTHORIZATION.json', 'SEALED_INPUTS.json',
        'CPU_TEST_RESULTS.json', 'reference_baseline.json')]
    require(all(path.is_file() for path in controls), 'Preparation control set incomplete')
    write_new(ROOT / 'PREPARATION.json', {
        'prepared': True,
        'utc': now(),
        'runtime_environment': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
        'control_hashes': {str(path): digest(path) for path in controls},
        'predecessor': handoff,
        'formal_started': False,
        'gpu_recipe': {'world_size': 4, 'batch': 4, 'gas': 4, 'zero': 1,
                       'ac': False, 'global_batch': 64},
        'preflight': {'fresh_steps': 3, 'resume_start': 3, 'resume_stop': 4},
        'data': {'episodes': sealed['episodes'], 'frames': sealed['frames'],
                 'excluded_original_merged_episodes': sealed['excluded_original_merged_episodes']},
        'large_asset_policy': 'Reuse old identity-bound source/data/stats/cache; no re-audit or transfer',
        'quota_policy': {'fresh_min_free_gib': 405, 'resume_min_free_gib': 314,
                         'formal_min_free_gib': 223, 'save_min_free_gib': 108},
    })
    print('INSERT_SUCCESSOR_PREPARATION_SEALED', flush=True)


if __name__ == '__main__':
    main()
