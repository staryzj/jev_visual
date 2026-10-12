"""CPU-only synthetic admission/scheduler checks, not benchmark evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.workflows.native_overlap_gate import reservations, eligible_gpus, parse_inventory, read_state
from scripts.workflows import run_native_jev_overlap as workflow


def state(completed=28, active=None):
    if active is None:
        active = {str(i): dict(pid=100+i, job=['b', f'd{i}']) for i in range(8)}
    return dict(status='running', phase='evaluate', total_jobs=36, completed_jobs=completed, active=active)


def devices():
    return {i: dict(uuid=f'GPU-{i}', memory_mib=0, bus=f'{i:02x}', pids=set()) for i in range(8)}


class GateTests(unittest.TestCase):
    def test_low_utilization_does_not_release_an_active_reservation(self):
        # No actual CUDA process yet: worker is loading images on CPU.
        free, _ = eligible_gpus(list(range(8)), state(), 36, 0, devices())
        self.assertEqual(free, [])

    def test_drained_final_queue_admits_only_permanently_released_gpus(self):
        s = state(34, {str(i): dict(pid=100+i, job=['b', f'd{i}']) for i in (0, 3)})
        free, _ = eligible_gpus(list(range(8)), s, 36, 0, devices())
        self.assertEqual(free, [1, 2, 4, 5, 6, 7])

    def test_an_idle_card_is_not_admitted_while_old_jobs_are_queued(self):
        free, reason = eligible_gpus(list(range(8)), state(2, {}), 36, 0, devices())
        self.assertEqual(free, [])
        self.assertIn('34 undispatched', reason)

    def test_cache_and_train_tail_cannot_admit_native_jobs(self):
        for phase in ('cache', 'train'):
            s = state(36, {})
            s['phase'] = phase
            self.assertFalse(reservations(s, 36, 0)[1])

    def test_stale_and_missing_snapshots_fail_closed(self):
        self.assertFalse(reservations(state(36, {}), 36, 46)[1])
        self.assertFalse(reservations({}, 36, 0)[1])

    def test_completed_final_matrix_admits_even_after_status_age_limit(self):
        s = dict(status='accessible_matrix_complete_access_gaps_remain',
                 full_accessible_matrix_complete=True, complete_cases=36)
        free, _ = eligible_gpus([0, 1], s, 36, 1000, devices())
        self.assertEqual(free, [0, 1])

    def test_actual_unrelated_gpu_process_and_memory_exclude_cards(self):
        d = devices()
        d[2]['pids'].add(900)
        d[3]['memory_mib'] = 2000
        free, _ = eligible_gpus([1, 2, 3, 4], state(36, {}), 36, 0, d)
        self.assertEqual(free, [1, 4])

    def test_original_worker_physical_gpu_mismatch_is_rejected(self):
        d = devices()
        d[1]['pids'].add(100)
        s = state(35, {'0': dict(pid=100, job=['b', 'd'])})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            eligible_gpus([2], s, 36, 0, d)

    def test_failure_count_mismatch_and_malformed_status_rejected(self):
        for s in (dict(status='failed'), dict(status='interrupted')):
            with self.assertRaises(RuntimeError):
                reservations(s, 36, 0)
        for s in (state(37, {}), state(-1, {}), state(30, {'0': dict(pid=0, job=['b', 'd'])}),
                  dict(full_accessible_matrix_complete=True, complete_cases=35)):
            with self.assertRaises(ValueError):
                reservations(s, 36, 0)

    def test_nvidia_physical_inventory_and_processes(self):
        parsed = parse_inventory('0, GPU-a, 10, 00000000:01:00.0\n1, GPU-b, 8000, 00000000:02:00.0\n',
                                 'GPU-b, 12345\n')
        self.assertEqual(parsed[0]['pids'], set())
        self.assertEqual(parsed[1]['pids'], {12345})
        self.assertEqual(parsed[1]['memory_mib'], 8000)

    def test_unknown_mig_unparseable_and_reordered_inventory_rejected(self):
        good = '0, GPU-a, 10, 00000000:01:00.0\n'
        for gpu, apps in ((good, 'MIG-abc, 100\n'), (good, 'GPU-a, N/A\n'), ('', ''),
                          ('0, GPU-a, 0, 00000000:02:00.0\n1, GPU-b, 0, 00000000:01:00.0', '')):
            with self.assertRaises(ValueError):
                parse_inventory(gpu, apps)

    def test_conflicting_restart_stops_native_not_original(self):
        for s in (dict(status='running', phase='train'), state(29, {'1': dict(pid=1, job=['b', 'd'])}),
                  state(20, {})):
            with self.assertRaises(RuntimeError):
                workflow.check_conflicts(s, 36, 0, {1: object()}, 28)

    def test_missing_status_returns_no_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            s, age = read_state(Path(temporary) / 'missing.json')
            self.assertEqual(s, {})
            self.assertEqual(age, float('inf'))

    def test_mocked_scheduler_uses_only_released_cards_preserves_full_N(self):
        class Child:
            def __init__(self):
                self.calls = 0
            def poll(self):
                self.calls += 1
                return None if self.calls == 1 else 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'native'
            output.mkdir()
            s = dict(status='running', phase='evaluate', completed_jobs=1, total_jobs=2,
                     active={'0': dict(pid=100, job=['b', 'd1'])})
            d = devices()
            d[0]['pids'].add(100)
            args = type('Args', (), dict(gpus=[0, 1, 2], native_batch=32, cpu_threads=2))()
            plan = dict(datasets={'d1': 12578, 'd2': 17901})
            with patch.object(workflow, 'ROOT', root), patch.object(workflow, 'read_state', return_value=(s, 0)), \
                    patch.object(workflow, 'gpu_inventory', return_value=d), \
                    patch.object(workflow.subprocess, 'Popen', side_effect=lambda *a, **k: Child()) as spawn, \
                    patch.object(workflow.time, 'sleep'), patch('builtins.print'):
                workflow.schedule(dict(backbones=['b']), copy.deepcopy(plan), args, output, root / 'jev')
            self.assertEqual(spawn.call_count, 2)
            commands = [call.args[0] for call in spawn.call_args_list]
            self.assertEqual([cmd[cmd.index('--device')+1] for cmd in commands], ['cuda:1', 'cuda:2'])
            self.assertEqual([cmd[cmd.index('--dataset')+1] for cmd in commands], ['d1', 'd2'])
            self.assertTrue(all('--worker' in cmd for cmd in commands))
            self.assertEqual(plan['datasets'], {'d1': 12578, 'd2': 17901})


if __name__ == '__main__':
    unittest.main()
