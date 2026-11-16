"""CPU tests for the portable frozen-receiver transfer code."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from visualjev.assets import matches, content_hash


class AssetTests(unittest.TestCase):
    def test_hashes_detect_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "model.bin"
            file.write_bytes(b"released weights")
            record = {"size": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            self.assertTrue(matches(file, record))
            file.write_bytes(b"modified weights")
            self.assertFalse(matches(file, record))

    def test_git_blob_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "config.json"
            file.write_bytes(b"{}")
            self.assertEqual(content_hash(file, True), hashlib.sha1(b"blob 2\0{}").hexdigest())

    def test_namespaced_upstream_import(self):
        from visualjev.transfer.upstream import upstream_module
        prompts = upstream_module("api").candidate_prompts(
            dict(state="", question="Where?", kind="choice", options=["left", "right"]))
        self.assertEqual(len(prompts), 2)
        self.assertIn("Proposed answer: left", prompts[0])


class MappingTests(unittest.TestCase):
    def test_qwen_identity_and_layout(self):
        import torch
        from torch.nn import functional as F
        from visualjev.transfer.geometry import align
        value = torch.randn(12, 7)
        record = dict(tensor=value, layout=dict(encoder="qwen", shape=[12, 7],
                      processor_geometry={"image_grid_thw": torch.tensor([[1, 6, 8]])}))
        result = align(record, [1, 6, 8])
        expected = F.interpolate(value.T[None], size=12, mode="linear", align_corners=False)[0].T
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.stride(), expected.stride())

    def test_llava_padding_excluded(self):
        import torch
        from visualjev.transfer.geometry import resize_and_padding, source_spec
        from visualjev.transfer.mapping_controls import maps
        spec = source_spec("llava")
        geo = resize_and_padding((720, 480), spec)
        n = (1 + geo["target_h"] // 384 * (geo["target_w"] // 384)) * 27 * 27
        layout = dict(encoder="llava", image_size=[720, 480], shape=[n, 896])
        for name, table in maps(layout, [1, 26, 38]).items():
            self.assertTrue(torch.allclose(table["weights"].sum(-1), torch.ones(13 * 19), atol=2e-6))
            if name != "grid_2d":
                self.assertEqual(table["padding_mass"], 0.0)

    def test_spatial_constant_features(self):
        import torch
        from visualjev.transfer.geometry import align
        record = dict(tensor=torch.ones(196, 3), layout=dict(encoder="siglip2", shape=[196, 3],
                      canonical_grid=[14, 14]))
        self.assertTrue(torch.allclose(align(record, [1, 26, 38]), torch.ones(247, 3), atol=2e-6))

    @unittest.skipUnless(os.environ.get("VISUAL_JEV_ASSETS"), "Set VISUAL_JEV_ASSETS for actual checkpoint tests")
    def test_actual_selected_checkpoints(self):
        import torch
        from visualjev.transfer.connector import from_checkpoint
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "release_metadata/visual_jev_interfaces.json").read_text())
        for name, entry in manifest["interfaces"].items():
            path = Path(os.environ["VISUAL_JEV_ASSETS"]) / entry["path"]
            self.assertEqual(content_hash(path), entry["sha256"])
            bridge = from_checkpoint(path)
            self.assertEqual(bridge.spec["output_dim"], 5120)
            self.assertTrue(all(not p.requires_grad for p in bridge.parameters()))
            if name == "qwen":
                x = torch.randn(12, 2560)
                z = bridge([dict(tensor=x)], [[1, 6, 8]])[0]
                self.assertEqual(tuple(z.shape), (12, 5120))
                self.assertTrue(torch.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
