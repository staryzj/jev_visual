import tempfile
import unittest
from pathlib import Path
import zipfile

from visualjev.reproduce import score, safe_extract


class PaperReproductionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(uid="one", dataset="scienceqa", label=0, candidates=["a", "b"], image="one.png"),
                     dict(uid="two", dataset="scienceqa", label=1, candidates=["a", "b"], image=None)]

    def test_invalid_answers_count_as_errors(self):
        measured = score(self.rows, [dict(uid="one", prediction=0), dict(uid="two", prediction=None)])
        self.assertEqual(measured["overall"]["accuracy_pct"], 50)
        self.assertEqual(measured["scienceqa_image"]["accuracy_pct"], 100)

    def test_duplicate_or_missing_questions_rejected(self):
        with self.assertRaises(ValueError):
            score(self.rows, [dict(uid="one", prediction=0), dict(uid="one", prediction=0)])
        with self.assertRaises(ValueError):
            score(self.rows, [dict(uid="one", prediction=0)])

    def test_changed_candidate_order_rejected(self):
        with self.assertRaises(ValueError):
            score(self.rows, [dict(uid="one", prediction=0, candidates=["b", "a"])], allow_subset=True)

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.txt", "invalid")
            with self.assertRaises(ValueError):
                safe_extract(archive, Path(folder) / "out")


if __name__ == "__main__":
    unittest.main()
