import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.open_jev_27b.metrics import distribution
from scripts.evaluation.open_jev_27b.prompts import parse_answer, parse_rationale, rationale_question
from scripts.evaluation.open_jev_27b.summarize import read_rows, report


def record(uid, correct=True, supported=True, error=False, latency=2, generated=0):
    return dict(uid=uid, dataset="scienceqa", cluster=uid, label=0, candidate_N=2,
                has_image=uid == "image", correct=correct, supported=supported,
                execution_error=error, prediction=0 if correct else None, e2e_ms=latency,
                generated_tokens=generated)


class ProtocolTests(unittest.TestCase):
    def test_parsers_preserve_options_without_using_gold(self):
        row = dict(question="Which object?", candidates=["cat", "dog"], label=0)
        self.assertEqual(parse_answer("B. dog", 2, row["candidates"]), 1)
        self.assertIsNone(parse_answer("B. cat", 2, row["candidates"]))
        self.assertEqual(parse_rationale("I compare both.\nFinal answer: B.", row)[0], 1)
        self.assertIsNone(parse_rationale("Final answer: B and C", row)[0])
        self.assertIn("A. cat\nB. dog", rationale_question(row))

    def test_paired_report_excludes_unsupported_items(self):
        route = {uid: record(uid) for uid in ("image", "text")}
        reference = {
            "image": record("image", correct=False, latency=4, generated=6),
            "text": record("text", correct=False, supported=False),
        }
        result = report(route, reference, route, "efficiency")
        self.assertTrue(result["final"])
        self.assertEqual(result["paired_accuracy"]["gained"], 1)
        self.assertEqual(result["paired_efficiency"]["comparator_generated_tokens"]["total"], 6)
        self.assertEqual(result["paired_efficiency"]["latency_sum_ratio_comparator_over_route"], 2)
        self.assertEqual(result["route"]["tasks"]["scienceqa_image"]["required_N"], 1)
        self.assertAlmostEqual(distribution([0, 1, 2, 3])["p95"], 2.85)

    def test_partial_or_failed_route_cannot_be_final(self):
        expected = {"image": record("image")}
        for rows in ({}, {"image": record("image", error=True)}):
            result = report(rows, expected, expected, "accuracy")
            self.assertFalse(result["final"])
            self.assertNotIn("paired_accuracy", result)
        with self.assertRaisesRegex(ValueError, "mismatched label"):
            report({"image": dict(record("image"), label=1)}, expected, expected, "accuracy")

    def test_duplicate_uid_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"uid":"same"}\n{"uid":"same"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate UID"):
                read_rows([path])

    def test_geometry_identity_and_teacher_is_detached(self):
        import torch

        from scripts.evaluation.open_jev_27b.geometry import align
        from scripts.evaluation.open_jev_27b.training import trusted_decision_loss

        tokens = torch.randn(6, 4)
        layout = dict(encoder="qwen", shape=list(tokens.shape),
                      processor_geometry={"image_grid_thw": torch.tensor([[1, 4, 6]])})
        self.assertTrue(torch.equal(align({"tensor": tokens, "layout": layout}, [1, 4, 6]), tokens))
        student = torch.tensor([0.1, 0.2], requires_grad=True)
        teacher = torch.tensor([2.0, 0.0], requires_grad=True)
        loss, stats = trusted_decision_loss(student, teacher, 0)
        loss.backward()
        self.assertTrue(stats["teacher_trusted"])
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
