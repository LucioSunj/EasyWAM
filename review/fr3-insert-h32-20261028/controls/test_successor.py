"""CPU-only, fail-closed successor contract tests; never SSH or use a GPU."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

import insert_common as common
import queue_successor as queue


class SuccessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_config_contract_and_fresh_initialization(self):
        baseline_path = common.old_baseline_path()
        if not baseline_path.is_file():
            self.skipTest('sealed local baseline snapshot is unavailable')
        baseline = common.read(baseline_path)
        cfg = common.make_config(baseline)
        common.validate_config(baseline, cfg)
        self.assertIsNone(cfg['resume'])
        for field, value in (('batch_size', 8), ('max_steps', 100),
                             ('resume', '/stale/checkpoint')):
            bad = json.loads(json.dumps(cfg))
            bad[field] = value
            with self.assertRaises(RuntimeError):
                common.validate_config(baseline, bad)

    def test_sealed_input_stats_schema_and_identity_roundtrip(self):
        source, data = self.base / 'source', self.base / 'data'
        source.mkdir()
        data.mkdir()
        (source / 'code.py').write_text('pass\n')
        (data / 'episode.json').write_text('{}\n')
        baseline = self.base / 'baseline.json'
        common.write_new(baseline, {'resume': None, 'output_dir': 'original-output'})
        baseline_hash = common.digest(baseline)
        stats = self.base / 'stats.json'
        common.write_new(stats, {'action': {}})
        stats_hash = common.digest(stats)
        inputs = {'source': {'code.py': {'identity': common.identity(source / 'code.py')}},
                  'assets': {}, 'text_cache': {}}
        manifest = {'episodes': 299, 'frames': 132292,
                    'excluded_original_merged_episodes': [250, 285],
                    'files': {'episode.json': {'identity': common.identity(data / 'episode.json')}}}
        documents = {
            'old_preparation': {'prepared': True, 'control_hashes': {}},
            'old_inputs': inputs,
            'old_cpu_audit': {'passed': True, 'stats_sha256': stats_hash,
                              'config_sha256': baseline_hash},
            'old_data_manifest': manifest,
            'old_queue_status': {'status': 'blocked'},
        }
        sealed = {'old_baseline_config_path': str(baseline),
                  'old_baseline_config_sha256': baseline_hash,
                  'old_formal_output': 'original-output', 'source_root': str(source),
                  'data_root': str(data), 'stats_path': str(stats), 'stats_sha256': stats_hash}
        for key, document in documents.items():
            path = self.base / (key + '.json')
            common.write_new(path, document)
            sealed[key + '_path'] = str(path)
            sealed[key + '_sha256'] = common.digest(path)
        seal = self.base / 'SEALED_INPUTS.json'
        common.write_new(seal, sealed)
        with mock.patch.object(common, 'SEALED', seal), \
                mock.patch.object(common, 'OLD_BASELINE_SHA256', baseline_hash):
            self.assertEqual(common.verify_sealed_inputs(), sealed)
            stats.write_text('{"changed": true}')
            with self.assertRaisesRegex(RuntimeError, 'Sealed old input changed'):
                common.verify_sealed_inputs()

    def test_nccl_environment_is_mandatory_from_a_clean_shell(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'NCCL_CUMEM_HOST_ENABLE=0'):
                common.require_nccl_environment()
        with mock.patch.dict(os.environ, {'NCCL_CUMEM_HOST_ENABLE': '1'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'NCCL_CUMEM_HOST_ENABLE=0'):
                common.require_nccl_environment()
        with mock.patch.dict(os.environ, {'NCCL_CUMEM_HOST_ENABLE': '0'}, clear=True):
            self.assertEqual(common.require_nccl_environment(), '0')

    def test_nccl_entry_guards_precede_gpu_imports_and_are_recorded(self):
        local_root = Path(__file__).resolve().parent
        preflight = (local_root / 'preflight_four_gpu.py').read_text()
        formal = (local_root / 'formal_train.py').read_text()
        for text in (preflight, formal):
            self.assertIn('require_nccl_environment()', text)
            self.assertLess(text.index('require_nccl_environment()'), text.index('import torch'))
        self.assertIn("'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE}", preflight)
        self.assertIn("'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE}", formal)
        queue_text = (local_root / 'run_queue.sh').read_text()
        self.assertIn('export NCCL_CUMEM_HOST_ENABLE=0', queue_text)
        self.assertNotIn('NCCL_P2P_DISABLE', queue_text)
        self.assertNotIn('NCCL_IB_DISABLE', queue_text)

    def test_predecessor_is_the_ncclfix_recovery_root(self):
        self.assertEqual(common.DRAWER_RECOVERY,
                         common.BASE / 'training_launch/fr3-drawer-recovery-2000-ncclfix-20261027')
        text = (Path(__file__).resolve().parent / 'prepare_successor.py').read_text()
        self.assertIn("'predecessor': str(DRAWER_RECOVERY)", text)



    def test_gpu_release_waits_for_lock_holder_and_idle_devices(self):
        root = self.base / 'queue'
        root.mkdir()
        lock_path = self.base / 'gpu.lock'
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(queue, 'ROOT', root))
            stack.enter_context(mock.patch.object(queue, 'GPU_LOCK', lock_path))
            stack.enter_context(mock.patch.object(queue, 'gpu_idle', side_effect=[False, True]))
            stack.enter_context(mock.patch.object(queue.time, 'sleep'))
            candidate = queue.wait_for_gpu_release_and_idle()
        self.assertFalse(candidate.closed)
        candidate.close()






    def make_checkpoint_fixture(self, sidecars=True):
        run = self.base / 'checkpoint'
        state = run / 'checkpoints/state/step_000003'
        weights = run / 'checkpoints/weights'
        state.mkdir(parents=True)
        weights.mkdir(parents=True)
        model_dir = state / 'pytorch_model'
        model_dir.mkdir()
        (state / 'latest').write_text('pytorch_model\n')
        common.write_new(state / 'trainer_state.json', {
            'global_step': 3, 'deepspeed_global_step': 3,
            'num_processes': 4, 'batch_size': 4,
            'gradient_accumulation_steps': 4, 'epoch': 0, 'batch_in_epoch': 0,
        })
        paths = [weights / 'step_000003.pt', model_dir / 'mp_rank_00_model_states.pt']
        paths.extend(model_dir / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
                     for rank in range(4))
        paths.extend(state / f'random_states_{rank}.pkl' for rank in range(4))
        for path in paths:
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('archive/data/0', b'fixture')
        if sidecars:
            for rank in range(4):
                (state / f'comparison-rank-{rank}.json').write_text(
                    json.dumps({'rank': rank}) + '\n')
        return run, state, paths

    def test_checkpoint_requires_all_rank_shards_and_crc(self):
        run, state, paths = self.make_checkpoint_fixture(sidecars=True)
        integrity = self.base / 'ARTIFACT_INTEGRITY.json'
        with mock.patch.object(common, 'ARTIFACT_INTEGRITY', integrity):
            evidence = common.checkpoint(run, 3, require_comparison_sidecars=True)
            self.assertEqual(len(evidence['files']), 16)
            cache_before = json.loads(integrity.read_text())
            repeated = common.checkpoint(run, 3, require_comparison_sidecars=True)
            self.assertEqual(repeated['files'], evidence['files'])
            self.assertEqual(json.loads(integrity.read_text()), cache_before)
            paths[-1].unlink()
            with self.assertRaises(RuntimeError):
                common.checkpoint(run, 3, require_comparison_sidecars=True)

    def test_checkpoint_rejects_shards_outside_pytorch_model(self):
        run, state, paths = self.make_checkpoint_fixture(sidecars=False)
        wrong_dir = state / 'unexpected'
        wrong_dir.mkdir()
        paths[1].rename(wrong_dir / paths[1].name)
        integrity = self.base / 'ARTIFACT_INTEGRITY-layout.json'
        with mock.patch.object(common, 'ARTIFACT_INTEGRITY', integrity):
            with self.assertRaisesRegex(RuntimeError, 'directly under pytorch_model'):
                common.checkpoint(run, 3)

    def test_formal_checkpoint_allows_no_comparison_sidecars(self):
        run, state, _ = self.make_checkpoint_fixture(sidecars=False)
        integrity = self.base / 'ARTIFACT_INTEGRITY-formal.json'
        with mock.patch.object(common, 'ARTIFACT_INTEGRITY', integrity):
            evidence = common.checkpoint(run, 3)
            self.assertEqual(len(evidence['files']), 12)
            with self.assertRaises(RuntimeError):
                common.checkpoint(run, 3, require_comparison_sidecars=True)

    def test_checkpoint_requires_correct_latest_tag_and_metadata(self):
        run, state, _ = self.make_checkpoint_fixture(sidecars=False)
        integrity = self.base / 'ARTIFACT_INTEGRITY-latest.json'
        with mock.patch.object(common, 'ARTIFACT_INTEGRITY', integrity):
            (state / 'latest').write_text('wrong_tag\n')
            with self.assertRaisesRegex(RuntimeError, 'pytorch_model'):
                common.checkpoint(run, 3)
            (state / 'latest').write_text('pytorch_model\n')
            (state / 'trainer_state.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing checkpoint marker'):
                common.checkpoint(run, 3)

    def test_quota_gate_is_fail_closed(self):
        with self.assertRaises(RuntimeError):
            common.check_quota({'verified': True, 'quota_free_bytes': str(1)}, 2)

    def test_distributed_save_gate_broadcasts_rank_zero_failure(self):
        fake_torch = types.SimpleNamespace(
            distributed=types.SimpleNamespace(
                is_initialized=lambda: True,
                broadcast_object_list=lambda payload, src, device: None))
        accelerator = types.SimpleNamespace(is_main_process=True, device='cpu')
        with mock.patch.object(common, 'live_quota', return_value={
                'verified': True, 'quota_free_bytes': '1'}), \
             mock.patch.dict(sys.modules, {'torch': fake_torch}):
            with self.assertRaisesRegex(RuntimeError, 'save gate failed'):
                common.distributed_save_gate(accelerator, 'formal', 2000)

    def test_stage_quota_budgets_are_distinct(self):
        self.assertEqual(queue.minimum_quota('fresh'), 405 * common.GIB)
        self.assertEqual(queue.minimum_quota('resume'), 314 * common.GIB)
        self.assertEqual(queue.minimum_quota('formal'), 223 * common.GIB)
        with self.assertRaises(RuntimeError):
            queue.minimum_quota('unknown')

    def test_queue_claim_failures_do_not_overwrite_status(self):
        root = self.base / 'queue'
        root.mkdir()
        status_path = root / 'QUEUE_STATUS.json'
        original = {'status': 'complete', 'final_step': 20000}
        status_path.write_text(json.dumps(original))
        with mock.patch.object(queue, 'ROOT', root):
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                queue.main([])
        self.assertEqual(json.loads(status_path.read_text()), original)

    def test_queue_lock_rejection_does_not_create_status(self):
        root = self.base / 'queue'
        root.mkdir()
        lock_path = root / 'queue.lock'
        with lock_path.open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(queue, 'ROOT', root):
                with self.assertRaises(BlockingIOError):
                    queue.main([])
        self.assertFalse((root / 'QUEUE_STATUS.json').exists())

    def test_preflight_reports_use_writer_paths_and_identity_contract(self):
        arm_dir = self.base / 'B'
        output = arm_dir / 'fresh'
        arm_dir.mkdir()
        output.mkdir()
        config = self.base / 'train.json'
        config.write_text('{"recipe":"fixed"}\n')
        status = {
            'status': 'exited', 'exit_code': 0, 'mode': 'fresh', 'arm': 'B',
            'automatic_restart': False, 'output_dir': str(output),
            'config_sha256': common.digest(config),
            'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': common.NCCL_CUMEM_HOST_ENABLE},
        }
        (arm_dir / 'STATUS-fresh.json').write_text(json.dumps(status))
        (self.base / 'SAVE_GATE-fresh-000003.json').write_text(json.dumps({
            'passed': True, 'stage': 'fresh', 'step': 3,
            'minimum_free_bytes': common.MIN_SAVE_QUOTA_BYTES,
            'quota': {'verified': True,
                      'quota_free_bytes': str(200 * common.GIB)},
        }))
        for rank in range(4):
            (arm_dir / f'acceptance-fresh-rank-{rank}.json').write_text(json.dumps({
                'mode': 'fresh', 'arm': 'B', 'rank': rank, 'pid': 100 + rank,
                'output_dir': str(output), 'config_sha256': common.digest(config),
                'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': common.NCCL_CUMEM_HOST_ENABLE},
                'spec': common.ARMS['B'], 'start_step': 0, 'final_step': 3,
                'global_batch': 64, 'grad_accum_dtype': 'torch.bfloat16',
                'scheduler_total_steps': 20000,
                'gradients': {'state_encoder.x': {'finite': True, 'nonzero': True}},
                'measurements': [{'step': step, 'global_grad_norm': 1.0,
                                  'total': 10 * common.GIB,
                                  'peak_allocated': 5 * common.GIB,
                                  'headroom_bytes': 5 * common.GIB}
                                 for step in (1, 2, 3)],
                'save_peaks': [{'step': 3, 'total': 10 * common.GIB,
                                'peak_allocated': 5 * common.GIB,
                                'headroom_bytes': 5 * common.GIB}],
                'evaluation_peak': None,
                'resume_verified': False, 'evaluation': None,
            }))
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(common, 'ARM_DIR', arm_dir))
            stack.enter_context(mock.patch.object(common, 'ROOT', self.base))
            stack.enter_context(mock.patch.object(common, 'CONFIG', config))
            checkpoint = stack.enter_context(mock.patch.object(
                common, 'checkpoint', return_value={'step': 3}))
            result = common.validate_preflight_report(self.base, 'fresh')
        self.assertEqual(result['checkpoint']['step'], 3)
        checkpoint.assert_called_once_with(output, 3, require_comparison_sidecars=True)
        bad = json.loads((arm_dir / 'acceptance-fresh-rank-2.json').read_text())
        bad['config_sha256'] = 'wrong'
        (arm_dir / 'acceptance-fresh-rank-2.json').write_text(json.dumps(bad))
        with mock.patch.object(common, 'ARM_DIR', arm_dir), \
             mock.patch.object(common, 'ROOT', self.base), \
             mock.patch.object(common, 'CONFIG', config), \
             mock.patch.object(common, 'checkpoint', return_value={'step': 3}):
            with self.assertRaises(RuntimeError):
                common.validate_preflight_report(self.base, 'fresh')

    def test_terminal_log_requires_finite_final_update(self):
        log = self.base / 'train.log'
        log.write_text('Training: 100%|bar| 3/3 [x, loss=0.12, grad_norm=1.2]\n')
        self.assertEqual(common.finite_log(log, 3)['last_step'], 3)
        log.write_text('Training: 100%|bar| 3/3 [x, loss=nan, grad_norm=1.2]\n')
        with self.assertRaises(RuntimeError):
            common.finite_log(log, 3)


if __name__ == '__main__':
    unittest.main()
