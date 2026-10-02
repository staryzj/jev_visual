"""Read-only validation of a Visual-JEV public snapshot, without weights/data."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT / 'PUBLICATION_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert path.is_file(), entry['path']
        assert sha(path) == entry['exported_sha256'], entry['path']
        if entry['mode'] == 'byte-identical':
            assert entry['exported_sha256'] == entry['source_sha256'], entry['path']
        else:
            value = json.loads(path.read_text())
            assert value['public_export_metadata']['source_sha256'] == entry['source_sha256']
            assert value['public_export_metadata']['omitted_fields'] == entry['omitted_fields']
    total = 0
    for name in ('scienceqa', 'aokvqa', 'iconqa', 'ai2d', 'vsr', 'sugarcrepe_pp'):
        directory = ROOT / 'experiments/results/submission_full' / name
        result = json.loads((directory / 'result.json').read_text())
        assert result['protocol'] == 'full_benchmark' and result['status'] == 'complete'
        predictions = directory / 'predictions.jsonl'
        assert sha(predictions) == result['predictions_sha256']
        rows = [json.loads(line) for line in predictions.read_text().splitlines()]
        assert len({r['uid'] for r in rows}) == len(rows) == result['N'] == result['metrics']['count']
        total += len(rows)
    assert total == manifest['full_prediction_instances'] == 21742
    table = ROOT / 'experiments/tables/submission/main_full_benchmark.csv'
    with table.open(encoding='utf-8-sig', newline='') as handle:
        complete = [row for row in csv.DictReader(handle) if row['Status'] == 'complete']
    assert len(complete) == 6 and sum(int(row['N']) for row in complete) == total
    # V7 source hashes refer to original records; public aggregates have their
    # own file hashes, verified above, and declare original source identities.
    sources = json.loads((ROOT / 'reports/frozen-scorer-v7/table_sources.json').read_text())
    with (ROOT / 'reports/frozen-scorer-v7/core_backbone_table.csv').open(newline='') as handle:
        core = {r['source']: r for r in csv.DictReader(handle)}
    assert len(core) == len(sources) == 3
    expected_names = {'siglip2': 'SigLIP2', 'internvl': 'InternVL3.5', 'llava': 'LLaVA-OneVision'}
    for source in sources:
        record = json.loads((ROOT / source['path']).read_text())
        row = core[source['path']]
        assert record['public_export_metadata']['source_sha256'] == source['sha256'] == row['source_sha256']
        assert source['base_checkpoint_sha256'] == record['base_checkpoint_sha256']
        assert record['fixed_decision_head'] is True
        assert row['backbone'] == expected_names[Path(source['path']).parent.name]
        assert int(row['N']) == record['split_counts']['test'] == 80
        assert float(row['accuracy_pct']) == round(record['test_uncalibrated']['accuracy'] * 100, 2)
        assert int(row['adapter_parameters']) == record['adapter_trainable_parameters']
        assert all(row[key] == '' for key in ('zero_visual', 'random_adapter', 'linear_adapter', 'semantic_pair'))
    print(json.dumps({'files_hashed': len(manifest['files']), 'complete_splits': 6,
                      'unique_complete_predictions': total, 'v7_core_rows': len(core), 'status': 'passed'}))

if __name__ == '__main__':
    main()
