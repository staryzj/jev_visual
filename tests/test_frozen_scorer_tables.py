"""Table-export regression checks with fixtures, not benchmark measurements."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest


@pytest.mark.parametrize('public_export', [False, True])
def test_core_table_preserves_source_identity_and_missing_controls(tmp_path, public_export):
    source_root = Path(__file__).resolve().parents[1]
    script = tmp_path/'scripts/build_frozen_scorer_paper_tables.py'
    script.parent.mkdir()
    shutil.copy2(source_root/'scripts/build_frozen_scorer_paper_tables.py', script)
    audit = tmp_path/'reports/backbone-adaptation-v6/audit.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'base_checkpoint_sha256': 'checkpoint-fixture'}))
    expected = {}
    for key in ('siglip2', 'internvl', 'llava'):
        path = tmp_path/'experiments/results/backbone_generality'/key/'results.json'
        path.parent.mkdir(parents=True)
        data = dict(status='success', fixed_decision_head=True,
                    base_checkpoint_sha256='checkpoint-fixture',
                    split_counts=dict(train=512,validation=80,calibration=80,test=80),
                    backbone_extraction=dict(native_shapes=[[196,768]],hidden_dim=768),
                    test_uncalibrated=dict(accuracy=.5), adapter_trainable_parameters=1000000,
                    visual_dependency={name:dict(accuracy=.5,flip_rate_from_original=0,js_from_original=0)
                                       for name in ('original','blank','noise','wrong_image')})
        if public_export:
            data['public_export_metadata'] = {'source_sha256': f'original-{key}'}
        path.write_text(json.dumps(data))
        expected[key] = f'original-{key}' if public_export else hashlib.sha256(path.read_bytes()).hexdigest()
    paper = tmp_path/'paper'
    (paper/'tables/submission').mkdir(parents=True)
    subprocess.run([sys.executable,str(script),'--paper',str(paper)],check=True,capture_output=True)
    sources = json.loads((tmp_path/'reports/frozen-scorer-v7/table_sources.json').read_text())
    for entry in sources:
        assert entry['sha256'] == expected[Path(entry['path']).parent.name]
    with (tmp_path/'reports/frozen-scorer-v7/core_backbone_table.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row['accuracy_pct'] == '50.0' for row in rows)
    assert all(row[key] == '' for row in rows for key in ('zero_visual','random_adapter','linear_adapter','semantic_pair'))
    text = (paper/'tables/submission/backbone_adaptation.tex').read_text()
    assert 'These are not official full test results.' in text
    assert text.count('& -- & -- & -- & -- '+r'\\') == 4
