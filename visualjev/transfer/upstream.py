"""Import the pinned upstream package without shadowing this repo's jev package."""
from importlib import import_module, util
from pathlib import Path
import sys


def upstream_module(name):
    alias = "visualjev._open_jev_upstream"
    if alias not in sys.modules:
        directory = Path(__file__).resolve().parents[2] / "third_party/open_jev/jev"
        if not (directory / "__init__.py").is_file():
            raise RuntimeError("Pinned Open-Jev source is missing; use an editable repository checkout")
        spec = util.spec_from_file_location(alias, directory / "__init__.py",
                                           submodule_search_locations=[str(directory)])
        module = util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return import_module(alias + "." + name)
