"""Repository reorganization guards; these tests do not measure model quality."""
import ast
import math
from pathlib import Path

import pytest

from visualjev.metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


def test_no_root_python_and_required_packages_present():
    assert not list(ROOT.glob('*.py'))
    for folder in ('visualjev', 'jev', 'scripts'):
        assert (ROOT / folder / '__init__.py').is_file()
    for folder in ('training', 'evaluation', 'data', 'backbones', 'tables', 'demo', 'maintenance'):
        assert (ROOT / 'scripts' / folder / '__init__.py').is_file()


def test_local_import_targets_survived_cleanup():
    for folder in ('visualjev', 'jev', 'scripts', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            module_parts = list(path.relative_to(ROOT).with_suffix('').parts)
            package = module_parts[:-1]
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        prefix = package[:len(package) - node.level + 1]
                        names = ['.'.join(prefix + (node.module or '').split('.')).rstrip('.')]
                    elif node.module:
                        names = [node.module]
                for name in names:
                    if name.split('.')[0] not in ('visualjev', 'jev', 'scripts'):
                        continue
                    target = ROOT.joinpath(*name.split('.'))
                    assert target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), (path, name)


def test_shared_metrics_preserve_variable_candidate_count():
    value = metrics([[0., 0.], [0., 0., 0.]], [0, 1])
    assert value['count'] == 2
    assert value['accuracy'] == .5
    assert value['macro_f1'] == pytest.approx(1 / 3)
    assert value['nll'] == pytest.approx((math.log(2) + math.log(3)) / 2)
    assert value['brier'] == pytest.approx((.5 + 2 / 3) / 2)
    assert value['ece'] == pytest.approx((.5 + 1 / 3) / 2)
