import hashlib

import pytest
from PIL import Image

from jev.vl_data import validate_vl_records


def _image(path, color):
    Image.new("RGB", (2, 2), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(split, image, digest, index):
    return {
        "id": f"vl:{index}",
        "group_id": f"group:{index}",
        "split": split,
        "source": "test",
        "state": {"sample": index},
        "images": [image.name],
        "question": "What is shown?",
        "kind": "choice",
        "options": ["red", "blue"],
        "target": [1.0, 0.0],
        "metadata": {
            "target_basis": "unit_test",
            "image_sha256": [digest],
            "provenance": {
                "type": "import",
                "input_sha256": digest,
                "source_url": "local unit-test fixture",
                "original_id": image.name,
                "license": "CC0",
                "split_policy": "one unique image per split",
            },
        },
    }


def test_validate_vl_records_accepts_unique_decodable_images(tmp_path):
    train = tmp_path / "train.png"
    test = tmp_path / "test.png"
    rows = [
        _row("train", train, _image(train, "red"), 1),
        _row("test", test, _image(test, "blue"), 2),
    ]

    summary = validate_vl_records(rows, tmp_path)

    assert summary["images"] == 2
    assert summary["unique_image_sha256"] == 2


def test_validate_vl_records_rejects_cross_split_image_leakage(tmp_path):
    image = tmp_path / "same.png"
    digest = _image(image, "red")
    rows = [
        _row("train", image, digest, 1),
        _row("test", image, digest, 2),
    ]

    with pytest.raises(ValueError, match="identical image bytes appear across splits"):
        validate_vl_records(rows, tmp_path)


def test_validate_vl_records_rejects_tampered_hash(tmp_path):
    image = tmp_path / "image.png"
    _image(image, "red")
    row = _row("train", image, "0" * 64, 1)

    with pytest.raises(ValueError, match="image SHA256 differs"):
        validate_vl_records([row], tmp_path)


def test_validate_vl_records_rejects_non_image_bytes(tmp_path):
    image = tmp_path / "fake.png"
    image.write_bytes(b"not an image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    row = _row("train", image, digest, 1)

    with pytest.raises(ValueError, match="image is not decodable"):
        validate_vl_records([row], tmp_path)
