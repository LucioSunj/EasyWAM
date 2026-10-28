"""CPU-only early-stop handoff and unchanged one-shot sequence tests."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import early_stop_handoff as handoff
import insert_common as common
import queue_successor as queue


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.audit = self.base / 'audit'
        self.audit.mkdir()
        self.recovery = self.base / 'recovery'
        self.recovery.mkdir()
        self.keep = self.base / 'output/checkpoints/state/step_010000'
        self.keep.mkdir(parents=True)
        self.handoff_path = self.base / 'EARLY_STOP_HANDOFF.json'
        self.auth = self.base / 'AUTHORIZATION.json'
        self.boot = self.base / 'boot'
        self.boot.write_text('fixture-boot')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [('AUDIT', self.audit), ('DRAWER_RECOVERY', self.recovery),
                            ('KEEP', self.keep), ('HANDOFF', self.handoff_path),
                            ('AUTHORIZATION', self.auth), ('BOOT_PATH', self.boot)]:
            self.stack.enter_context(mock.patch.object(handoff, name, value))
        self.probe = self.stack.enter_context(mock.patch.object(handoff, 'process', return_value=None))
        config = self.recovery / 'config-continuation.json'
        config.write_text('{"max_steps":20000}')
        self.config_hash = common.digest(config)
        self.stack.enter_context(mock.patch.object(handoff, 'CONTINUATION_SHA256', self.config_hash))
        marker = {'global_step': 10000, 'deepspeed_global_step': 10000,
                  'batch_size': 4, 'num_processes': 4, 'gradient_accumulation_steps': 4}
        cache, files, relative = {}, {}, {}
        for name in handoff.MEMBERS:
            path = self.keep / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in ('.pt', '.pkl'):
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('tensor/data/0', b'fixture')
                with zipfile.ZipFile(path) as archive:
                    self.assertIsNone(archive.testzip())
                cache[str(path)] = common.identity(path)
                record = {'identity': common.identity(path), 'crc_passed': True}
            else:
                path.write_text(json.dumps(marker) if name == 'trainer_state.json'
                                else 'pytorch_model\n' if name == 'latest' else '# fixture\n')
                record = {'identity': common.identity(path), 'sha256': common.digest(path)}
            files[str(path)] = record
            relative[name] = record['identity']
        common.write_new(self.audit / 'KEEPER_CRC_CACHE.json', cache)
        common.write_new(self.audit / 'KEEPER_INTEGRITY.json', {
            'passed': True, 'archives_checked': 9,
            'cache_sha256': common.digest(self.audit / 'KEEPER_CRC_CACHE.json'),
            'checkpoint': {'step': 10000, 'state_dir': str(self.keep), 'marker': marker, 'files': files}})
        receipts = []
        for rank in range(4):
            value = {'rank': rank, 'global_step': 10000, 'state_path': str(self.keep),
                     'scheduler_total_steps': 20000, 'config_sha256': self.config_hash}
            receipts.append(value)
            common.write_new(self.recovery / f'saved-continuation-010000-rank-{rank}.json', value)
        common.write_new(self.recovery / 'PREPARATION.json', {'prepared': True})
        common.write_new(self.recovery / 'AUTHORIZATION.json', {'approved': True})
        common.write_new(self.audit / 'AUTHORIZATION.json', {
            'approved': True, 'reason': handoff.REASON, 'keep_step': 10000,
            'keep_state': str(self.keep), 'drawer_20000_complete': False,
            'automatic_restart': False, 'boot_id': 'fixture-boot', 'stop_pids': [123]})
        common.write_new(self.audit / 'STOPPED.json', {
            'passed': True, 'status': handoff.REASON, 'drawer_20000_complete': False,
            'keep_step': 10000, 'all_target_processes_inactive': True,
            'last_updates': [{'step': 11158}] * 4,
            'original_processes': {'123': {'start_ticks': '100'}},
            'keeper': {'marker': marker, 'files': relative, 'save_receipts': receipts}})
        common.write_new(self.audit / 'CLEANUP_AUTHORIZATION.json', {'approved': True})
        common.write_new(self.audit / 'CLEANUP_MANIFEST.json', [
            {'path': str(self.base / 'deleted' / str(n))} for n in range(30)])
        (self.audit / 'CLEANUP_DELETED.jsonl').write_text('{}\n')
        common.write_new(self.audit / 'CLEANUP_POSTCHECK.json', {
            'passed': True, 'keeper_unchanged': True, 'deleted_files_absent': True,
            'protected_files_unchanged': True})
        common.write_new(self.audit / 'CLEANUP_COMPLETION.json', {
            'passed': True, 'deleted_files': 30, 'keep_step': 10000, 'keep_state': str(self.keep),
            'drawer_20000_complete': False,
            'manifest_sha256': common.digest(self.audit / 'CLEANUP_MANIFEST.json'),
            'postcheck_sha256': common.digest(self.audit / 'CLEANUP_POSTCHECK.json'),
            'keeper_integrity_sha256': common.digest(self.audit / 'KEEPER_INTEGRITY.json')})
        handoff.create_handoff()
        common.write_new(self.auth, {'approved': True, 'automatic_retry': False,
            'predecessor_reason': handoff.REASON,
            'early_stop_handoff_sha256': common.digest(self.handoff_path)})

    def test_accepts_state_only_without_20k_or_weight(self):
        evidence = handoff.verify_early_stop()
        self.assertEqual(evidence['keep_step'], 10000)
        self.assertEqual(evidence['actual_stop_step'], 11158)
        self.assertFalse(evidence['drawer_20000_complete'])
        self.assertFalse((self.keep.parent.parent / 'weights').exists())

    def test_missing_handoff_not_replaced_by_generic_failure_or_success(self):
        self.handoff_path.unlink()
        for status in ('blocked', 'failed', 'complete'):
            (self.recovery / 'QUEUE_STATUS.json').write_text(json.dumps({'status': status}))
            with self.assertRaisesRegex(RuntimeError, 'Missing handoff'):
                handoff.verify_early_stop()

    def test_wrong_reason_step_or_path_is_rejected(self):
        original = common.read(self.handoff_path)
        auth = common.read(self.auth)
        for field, value in [('reason', 'training_failure'), ('keep_step', 2000),
                             ('keep_state', str(self.base / 'arbitrary')),
                             ('drawer_20000_complete', True)]:
            self.handoff_path.write_text(json.dumps({**original, field: value}))
            self.auth.write_text(json.dumps({**auth, 'early_stop_handoff_sha256': common.digest(self.handoff_path)}))
            with self.assertRaisesRegex(RuntimeError, 'Wrong early-stop'):
                handoff.verify_early_stop()

    def test_changed_authorization_or_bound_audit_is_rejected(self):
        auth = common.read(self.auth)
        self.auth.write_text(json.dumps({**auth, 'early_stop_handoff_sha256': 'wrong'}))
        with self.assertRaisesRegex(RuntimeError, 'authorization'):
            handoff.verify_early_stop()
        self.auth.write_text(json.dumps(auth))
        (self.audit / 'STOPPED.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'evidence changed'):
            handoff.verify_early_stop()

    def test_live_and_reused_pids_rejected_zombies_accepted(self):
        for item in ({'state': 'S', 'start_ticks': '100'}, {'state': 'Z', 'start_ticks': '101'}):
            self.probe.return_value = item
            with self.assertRaisesRegex(RuntimeError, 'active or PID was reused'):
                handoff.verify_early_stop()
        self.probe.return_value = {'state': 'Z', 'start_ticks': '100'}
        self.assertTrue(handoff.verify_early_stop()['verified'])

    def test_missing_extra_and_symlink_keeper_members_rejected(self):
        extra = self.keep / 'extra'
        extra.write_text('unexpected')
        with self.assertRaisesRegex(RuntimeError, 'member set'):
            handoff.verify_early_stop()
        extra.unlink()
        extra.symlink_to(self.keep / 'latest')
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            handoff.verify_early_stop()
        extra.unlink()
        (self.keep / 'random_states_3.pkl').unlink()
        with self.assertRaisesRegex(RuntimeError, 'member set'):
            handoff.verify_early_stop()

    def test_mutated_archive_identity_rejected(self):
        with (self.keep / 'random_states_0.pkl').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            handoff.verify_early_stop()

    def test_missing_crc_proof_rejected_even_after_rebinding_hash(self):
        path = self.audit / 'KEEPER_INTEGRITY.json'
        value = common.read(path)
        value['archives_checked'] = 8
        path.write_text(json.dumps(value))
        h = common.read(self.handoff_path)
        h['evidence_hashes'][str(path)] = common.digest(path)
        self.handoff_path.write_text(json.dumps(h))
        auth = common.read(self.auth)
        auth['early_stop_handoff_sha256'] = common.digest(self.handoff_path)
        self.auth.write_text(json.dumps(auth))
        with self.assertRaisesRegex(RuntimeError, 'CRC evidence'):
            handoff.verify_early_stop()

    def test_queue_runs_each_stage_once_in_order(self):
        root = self.base / 'queue'
        root.mkdir()
        lock = (self.base / 'gpu.lock').open('a')
        self.addCleanup(lock.close)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(queue, 'ROOT', root))
            stack.enter_context(mock.patch.object(queue, 'HANDOFF', self.handoff_path))
            stack.enter_context(mock.patch.object(queue, 'CONFIG', self.auth))
            stack.enter_context(mock.patch.object(queue, 'verify_preparation'))
            stack.enter_context(mock.patch.object(queue, 'predecessor_ready', return_value=handoff.verify_early_stop()))
            stack.enter_context(mock.patch.object(queue, 'wait_for_gpu_release_and_idle', return_value=lock))
            stack.enter_context(mock.patch.object(queue, 'live_quota', return_value={'verified': True, 'quota_free_bytes': str(700 * common.GIB)}))
            stack.enter_context(mock.patch.object(queue, 'validate_preflight_report', return_value={'passed': True}))
            stack.enter_context(mock.patch.object(queue, 'validate_formal', return_value={'passed': True}))
            launch = stack.enter_context(mock.patch.object(queue, 'launch'))
            queue.main([])
            self.assertEqual([call.args[0] for call in launch.call_args_list], ['fresh', 'resume', 'formal'])
            self.assertEqual(common.read(root / 'QUEUE_STATUS.json')['status'], 'complete')
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                queue.main([])
            self.assertEqual(launch.call_count, 3)

    def test_stage_refuses_existing_status_before_popen(self):
        root = self.base / 'stage'
        root.mkdir()
        status = root / 'STATUS-fresh.json'
        status.write_text('{"status":"exited"}')
        gpu_lock_path = self.base / 'gpu-stage.lock'
        with gpu_lock_path.open('a') as lock, ExitStack() as stack:
            stack.enter_context(mock.patch.object(queue, 'GPU_LOCK', gpu_lock_path))
            stack.enter_context(mock.patch.object(queue, 'verify_preparation'))
            stack.enter_context(mock.patch.object(queue, 'predecessor_ready', return_value={'verified': True}))
            stack.enter_context(mock.patch.object(queue, 'live_quota', return_value={'verified': True, 'quota_free_bytes': str(700 * common.GIB)}))
            stack.enter_context(mock.patch.object(queue, '_stage_paths', return_value=(root / 'output', status, root / 'train.py')))
            popen = stack.enter_context(mock.patch.object(queue.subprocess, 'Popen'))
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                queue.launch('fresh', lock)
            popen.assert_not_called()
            self.assertEqual(common.read(status), {'status': 'exited'})

    def test_formal_telemetry_does_not_change_recipe_or_initialization(self):
        source = (Path(__file__).parent / 'formal_train.py').read_text()
        self.assertIn('self.start_step == 0 and self.model.global_steps == 0', source)
        self.assertIn('cfg.resume is None and cfg.max_steps == 20000', source)
        self.assertIn('self.measurements.append(measurement)', source)
        self.assertIn("self.update_log.open('a')", source)
        self.assertIn('allow_nan=False', source)


if __name__ == '__main__':
    unittest.main()
