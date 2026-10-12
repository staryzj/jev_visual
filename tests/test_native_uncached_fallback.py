"""Synthetic CPU correctness/resume tests; not benchmark performance."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from scripts.evaluation.native_uncached_fallback import (
    ColdNativeBaseline, evaluate_with_fallback, selected_native_root,
)
from scripts.evaluation.native_jev_comparison import evaluate_native, compare_case, build_tables
from scripts.training.train_adapter_study_matrix import train
from scripts.evaluation.evaluate_adapter_study_matrix import evaluate
from scripts.evaluation.jev_pair_protocol import sha256
from tests.test_native_jev_comparison import FullComparisonTests, ToyNative, CausalToy, Tokenizer


class ColdScoreTests(unittest.TestCase):
    def test_every_candidate_uses_full_forward_never_prefix_cache(self):
        baseline = ColdNativeBaseline.__new__(ColdNativeBaseline)
        baseline.backbone, baseline.device, baseline.dtype = 'internvl', 'cpu', torch.float32
        model = CausalToy()
        model.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=128))
        baseline.model = Mock(side_effect=model)
        baseline.model.config = model.config
        baseline.processor = SimpleNamespace(tokenizer=Tokenizer(), apply_chat_template=Mock(return_value=dict(
            input_ids=torch.tensor([[3, 4, 5]]), attention_mask=torch.ones((1, 3), dtype=torch.long),
            pixel_values=torch.tensor([2.]))))
        result = baseline.scores(None, 'Full question', ['', 'red', 'many words'])
        self.assertEqual(len(result), 3)
        self.assertEqual(baseline.model.call_count, 3)
        for call in baseline.model.call_args_list:
            self.assertFalse(call.kwargs['use_cache'])
            self.assertNotIn('past_key_values', call.kwargs)
        self.assertIn('EVERY', baseline.protocol['inference_path'])

    def test_context_overflow_is_not_silently_truncated(self):
        baseline = ColdNativeBaseline.__new__(ColdNativeBaseline)
        baseline.backbone, baseline.device, baseline.dtype = 'internvl', 'cpu', torch.float32
        baseline.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4)))
        baseline.processor = SimpleNamespace(tokenizer=Tokenizer(), apply_chat_template=lambda *a, **k: dict(
            input_ids=torch.tensor([[3, 4, 5]]), attention_mask=torch.ones((1, 3), dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, 'no question/candidate truncation'):
            baseline.scores(None, 'question', ['a long answer'])


class FallbackTests(FullComparisonTests):
    def failing_baseline(self, succeed_first=0, message=None):
        class Failing(ToyNative):
            def __init__(self): self.calls = 0
            def scores(self, *args):
                self.calls += 1
                if self.calls <= succeed_first: return [1., 0.]
                raise ValueError(message or 'Native cached/uncached score difference 0.04438018798828125 exceeds tolerance')
        return Failing()

    def test_failed_guard_restarts_entire_case_cold_without_mixing_prefix(self):
        config, _ = self.native_config()
        output = self.root/'run/native'
        cold = Mock(spec=ToyNative, windowed_texts=0, last_check=None, protocol=ToyNative.protocol)
        cold.scores.return_value = [0., 1.]
        result = evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu',
                                       legacy_baseline=self.failing_baseline(2), cold_baseline=cold)
        self.assertEqual(result['N'], 4)
        self.assertEqual(cold.scores.call_count, 4)  # not just the remaining 2 rows
        self.assertFalse(result['identity']['use_cache'])
        self.assertEqual(result['numerical_fallback']['tolerance'], .02)
        rejected = output/'toy/toy/predictions.jsonl'
        self.assertEqual(len(rejected.read_text().splitlines()), 2)
        chosen = selected_native_root(output, 'toy', 'toy')
        self.assertEqual(chosen, output.parent/'native_uncached')
        records = [json.loads(line) for line in (chosen/'toy/toy/predictions.jsonl').read_text().splitlines()]
        self.assertEqual([record['uid'] for record in records], [row['uid'] for row in self.rows])
        self.assertTrue(all(record['scoring_path'] == 'full_uncached' for record in records))
        digest = sha256(rejected)
        again = evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu',
                                       legacy_baseline=self.failing_baseline(), cold_baseline=cold)
        self.assertEqual(result, again)
        self.assertEqual(cold.scores.call_count, 4)
        self.assertEqual(sha256(rejected), digest)

    def test_completed_legacy_result_reused_unchanged_without_cold_model(self):
        config, _ = self.native_config()
        output = self.root/'run/native'
        old = evaluate_native(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu', baseline=ToyNative())
        path = output/'toy/toy/result.json'
        digest = sha256(path)
        with patch('scripts.evaluation.native_uncached_fallback.ColdNativeBaseline') as factory:
            result = evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu')
        self.assertEqual(result, old)
        self.assertEqual(sha256(path), digest)
        factory.assert_not_called()
        self.assertEqual(selected_native_root(output, 'toy', 'toy'), output)

    def test_other_errors_are_not_suppressed_as_numerical_fallback(self):
        config, _ = self.native_config()
        for index, message in enumerate(['Missing image', 'Native resume protocol/model changed',
                                         'Native cached/uncached score difference 0.01 exceeds tolerance']):
            output = self.root/f'run{index}/native'
            with self.assertRaisesRegex(ValueError, message):
                evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu',
                                       legacy_baseline=self.failing_baseline(message=message), cold_baseline=ToyNative())
            self.assertFalse((output.parent/'native_uncached').exists())

    def test_cold_resume_retains_committed_full_forward_rows(self):
        config, _ = self.native_config()
        output = self.root/'run/native'
        evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu',
                               legacy_baseline=self.failing_baseline(), cold_baseline=ToyNative())
        directory = output.parent/'native_uncached/toy/toy'
        (directory/'result.json').unlink()
        pp = directory/'predictions.jsonl'
        lines = pp.read_bytes().splitlines(keepends=True)
        pp.write_bytes(b''.join(lines[:2])+b'{"uid":')
        cold = Mock(spec=ToyNative, windowed_texts=0, last_check=None, protocol=ToyNative.protocol)
        cold.scores.return_value = [1., 0.]
        result = evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu', cold_baseline=cold)
        self.assertEqual(result['N'], 4)
        self.assertEqual(cold.scores.call_count, 2)
        self.assertEqual(len(list(directory.glob('uncommitted-native-tail-*.bin'))), 1)
        # An incomplete cold case must never route to any old cached result.
        (directory/'result.json').unlink()
        with self.assertRaisesRegex(ValueError, 'remains incomplete'):
            selected_native_root(output, 'toy', 'toy')

    def test_cold_result_joins_with_unchanged_full_JEV_evidence_verifier(self):
        config, manifest = self.native_config()
        output = self.root/'run/native'
        evaluate_with_fallback(self.root, config, 'toy', 'toy', {'fixture': 'sha'}, output, 'cpu',
                               legacy_baseline=self.failing_baseline(), cold_baseline=ToyNative())
        training = train(self.root, config, manifest, self.rows, self.refs, self.pool, self.source,
                         'toy', self.root/'training', 'cpu')
        adapted = self.root/'adapted'
        evaluate(self.root, config, 'toy', manifest, self.rows, 'toy', training, self.pool, adapted, 'cpu')
        selected = selected_native_root(output, 'toy', 'toy')
        result = compare_case(self.root, config, 'toy', 'toy', selected, adapted, self.root/'comparisons')
        self.assertEqual(result['N'], 4)
        self.assertIn('native_uncached', result['source_results'][0]['path'])
        summary = build_tables(self.root/'comparisons', self.root/'reports')
        self.assertEqual(summary['complete_cases'], 1)
        self.assertTrue((self.root/'reports/native_vs_jev_main.tex').exists())


class VerifiedControllerTests(unittest.TestCase):
    def test_worker_redirect_is_scoped_and_comparison_selects_correct_root(self):
        import tempfile
        from pathlib import Path
        from scripts.workflows import run_native_jev_verified as verified
        import scripts.workflows.run_native_jev_overlap as scheduler
        import scripts.evaluation.native_jev_comparison as comparison
        original_worker = scheduler.ORIGINAL
        calls = []

        def old_schedule(*args):
            calls.append(scheduler.ORIGINAL)

        def old_compare(*args):
            calls.append(args[4])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'native'
            cold = output.parent/'native_uncached'

            def fake_main():
                scheduler.schedule({}, {}, SimpleNamespace(), output, Path(temporary)/'study')
                comparison.compare_case(Path(temporary), {}, 'b', 'd', output, Path(temporary)/'study', output/'comparisons')

            with patch.object(scheduler, 'schedule', old_schedule), patch.object(scheduler, 'main', fake_main), \
                    patch.object(comparison, 'compare_case', old_compare), patch.object(verified.sys, 'argv', ['verified']), \
                    patch('scripts.evaluation.native_uncached_fallback.selected_native_root', return_value=cold):
                verified.main()
                self.assertIs(scheduler.schedule, old_schedule)
                self.assertIs(comparison.compare_case, old_compare)
            self.assertEqual(calls, [Path(verified.__file__).resolve(), cold])
            self.assertEqual(scheduler.ORIGINAL, original_worker)


if __name__ == '__main__':
    unittest.main()
