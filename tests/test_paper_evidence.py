"""Read-only aggregate evidence tests; no model/GPU or benchmark inference."""
import hashlib
import json
import unittest
from pathlib import Path
from scripts.tables.build_paper_evidence import ROOT, BACKBONES, TASKS, verify_snapshot


class PublishedEvidenceTests(unittest.TestCase):
    def test_exact_server_snapshot_hashes(self):
        self.assertGreater(len(verify_snapshot()['files']), 68)

    def test_main_table_contains_only_complete_native_pairs(self):
        text=(ROOT/'reports/full_study_20261008/native_main.tex').read_text()
        self.assertNotIn('--',text)
        self.assertNotIn('GQA balanced',text)
        self.assertNotIn('IconQA fill',text)
        self.assertIn('TextVQA',text)
        self.assertIn(r'\textbf{69.42}',text)  # Retain a native advantage.

    def test_complete_and_adverse_evidence_remains_in_full_table(self):
        text=(ROOT/'reports/full_study_20261008/full_results.tex').read_text()
        self.assertEqual(text.count(' & -- & '),4)
        self.assertIn('38,589',text)
        self.assertIn(r'\textbf{35.40}',text)

    def test_timing_scope_and_counts(self):
        d=json.loads((ROOT/'reports/full_study_20261008/evidence_audit.json').read_text())
        self.assertEqual((d['jev_complete'],d['native_complete'],d['jev_improvements']),(36,32,21))
        self.assertIn('no native end-to-end speedup',d['latency_scope'])
        self.assertEqual(d['main_paired_tasks'],10)


if __name__=='__main__':unittest.main()
