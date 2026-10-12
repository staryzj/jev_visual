"""CPU-only file replacement race tests. No GPU processes are started."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.workflows import native_overlap_gate as gate


class StatusSnapshotTests(unittest.TestCase):
    def test_atomic_replacement_during_read_retries_to_new_complete_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pipeline_status.json'
            replacement = path.with_suffix('.tmp')
            path.write_text(json.dumps(dict(status='running', phase='evaluate', completed_jobs=33)))
            final = dict(status='accessible_matrix_complete_access_gaps_remain',
                         full_accessible_matrix_complete=True, complete_cases=36)
            replacement.write_text(json.dumps(final))
            original_load = json.load
            calls = []

            def replace_after_load(handle):
                state = original_load(handle)
                if not calls:
                    os.replace(replacement, path)
                calls.append(state)
                return state

            with patch.object(gate.json, 'load', side_effect=replace_after_load), patch.object(gate.time, 'sleep') as sleep:
                state, age = gate.read_state(path)
            self.assertEqual(state, final)
            self.assertEqual(len(calls), 2)
            sleep.assert_called_once_with(.02)
            self.assertLess(age, 1)
            self.assertTrue(gate.reservations(state, 36, age)[1])

    def test_in_place_write_during_read_retries_instead_of_returning_old_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pipeline_status.json'
            path.write_text('{"status":"running"}')
            final = dict(full_accessible_matrix_complete=True, complete_cases=36)
            original_load = json.load
            calls = []

            def rewrite_after_load(handle):
                state = original_load(handle)
                if not calls:
                    path.write_text(json.dumps(final))
                calls.append(state)
                return state

            with patch.object(gate.json, 'load', side_effect=rewrite_after_load), patch.object(gate.time, 'sleep'):
                state, _ = gate.read_state(path)
            self.assertEqual(state, final)
            self.assertEqual(len(calls), 2)

    def test_transient_incomplete_json_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pipeline_status.json'
            final = dict(status='running', phase='evaluate')
            path.write_text(json.dumps(final))
            original_load = json.load
            calls = []

            def once_incomplete(handle):
                calls.append(1)
                if len(calls) == 1:
                    raise json.JSONDecodeError('SYNTHETIC transient', '', 0)
                return original_load(handle)

            with patch.object(gate.json, 'load', side_effect=once_incomplete), patch.object(gate.time, 'sleep'):
                state, _ = gate.read_state(path)
            self.assertEqual(state, final)
            self.assertEqual(len(calls), 2)

    def test_persistent_corruption_is_not_treated_as_free_gpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pipeline_status.json'
            path.write_text('{')
            with patch.object(gate.time, 'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'no GPU admission'):
                    gate.read_state(path, attempts=3)
            self.assertEqual(sleep.call_count, 2)

    def test_absent_status_is_unavailable_and_non_object_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pipeline_status.json'
            with patch.object(gate.time, 'sleep'):
                self.assertEqual(gate.read_state(path), ({}, float('inf')))
            path.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'JSON object'):
                gate.read_state(path)


if __name__ == '__main__':
    unittest.main()
