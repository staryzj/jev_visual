"""Audit the evidence-preserving V7 rewrite and render both PDFs for review."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import pymupdf as fitz

p = argparse.ArgumentParser()
p.add_argument('--before', type=Path, required=True)
p.add_argument('--paper', type=Path, required=True)
p.add_argument('--qa', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]

def numeric_rows(path):
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if '&' in line and re.search(r'\d', line) and line.rstrip().endswith(r'\\')]

checks = []
for old in (a.before/'tables/submission').glob('*.tex'):
    if old.name in {'backbone_adaptation.tex', 'backbone_controls.tex'}:
        continue
    assert numeric_rows(old) == numeric_rows(a.paper/'tables/submission'/old.name), old.name
    checks.append('Unchanged numerical rows: '+old.name)
for old in (a.before/'figures').rglob('*'):
    if old.is_file():
        new = a.paper/old.relative_to(a.before)
        assert old.read_bytes() == new.read_bytes(), old.name
assert (a.before/'main.bib').read_bytes() == (a.paper/'main.bib').read_bytes()
checks.append('All figure bytes and user-aligned bibliography unchanged')

main_sources = '\n'.join(f.read_text(encoding='utf-8') for f in (a.paper/'sec').glob('*.tex'))
assert r'\input{tables/submission/matched_visual_jev}' not in main_sources
assert r'\ref{tab:submission-matched}' not in main_sources
assert 'matched reference baseline' not in main_sources
assert main_sources.count('Official Visual Jev') == 1
assert 'Original-image accuracy did not exceed' in main_sources
checks.append('Official performance comparison is supplement-only; limiting target controls stay main')
table = (a.paper/'tables/submission/backbone_adaptation.tex').read_text()
sources = json.loads((root/'reports/frozen-scorer-v7/table_sources.json').read_text())
names = {'siglip2':'SigLIP2','internvl':'InternVL3.5','llava':'LLaVA-OneVision'}
for source in sources:
    file = root/source['path']
    assert hashlib.sha256(file.read_bytes()).hexdigest() == source['sha256']
    data = json.loads(file.read_text())
    name = names[file.parent.name]
    row = next(line for line in table.splitlines() if line.startswith(name+' &'))
    assert f"{100*data['test_uncalibrated']['accuracy']:.2f}" in row
    assert row.endswith('& -- & -- & -- & -- '+r'\\')
checks.append('Core rows match real result hashes; missing exact-protocol controls stay --')

pdftexts = {}
for stem in ('main', 'supplement'):
    doc = fitz.open(a.qa/(stem+'.pdf'))
    text = unicodedata.normalize('NFKC', '\n'.join(page.get_text() for page in doc))
    pdftexts[stem] = text
    assert '??' not in text
    (a.qa/(stem+'-text.txt')).write_text(text, encoding='utf-8')
    for start in range(0, len(doc), 6):
        sheet = fitz.open()
        page = sheet.new_page(width=900, height=1150)
        for slot, index in enumerate(range(start, min(start+6,len(doc)))):
            x = slot%2*450
            y = slot//2*383
            page.show_pdf_page(fitz.Rect(x+5,y+5,x+445,y+365),doc,index)
            page.insert_text((x+12,y+378),f'{stem}: {index+1}',fontsize=10)
        page.get_pixmap().save(a.qa/f'{stem}-contact-{start//6+1}.png')
    if stem == 'main':
        for index, page in enumerate(doc):
            if 'Core adapter-only cross-backbone study' in page.get_text():
                page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(a.qa/'core-table-page.png')
    else:
        for index, page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(a.qa/f'supplement-page-{index+1}.png')
    checks.append(f'{stem}.pdf: {len(doc)} pages, no unresolved-reference markers')
assert 'Official Visual Jev as a reference system' not in pdftexts['main']
assert 'Official Visual Jev as a reference system' in pdftexts['supplement']
assert 'Table S1.' in pdftexts['supplement']
for heading in ('4.1. Cross-backbone', '4.2. Adapter ablations', '4.3. Visual dependence', '4.4. Dataset generalization'):
    assert heading in pdftexts['main'], heading
for stem in ('main','supplement'):
    log = (a.qa/(stem+'.log')).read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'Undefined control sequence|undefined references|undefined citations|Overfull',log)
    blg = (a.qa/(stem+'.blg')).read_text()
    assert 'error' not in blg.lower()
checks.append('Both roots compile without undefined commands/citations/references, BibTeX errors or overfull boxes')
counts = {}
for name in ('0_abstract.tex','1_intro.tex','2_formatting.tex','3_finalcopy.tex','4_submission_experiments.tex','5_backbone_adaptation.tex','experimental_record.tex'):
    counts[name] = {label:len((folder/'sec'/name).read_text().split()) for label,folder in [('before',a.before),('after',a.paper)]}
(a.qa/'evidence-audit.json').write_text(json.dumps({'checks':checks,'source_whitespace_token_counts':counts},indent=2)+'\n')
print('\n'.join('PASS: '+s for s in checks))
print(json.dumps(counts,indent=2))
