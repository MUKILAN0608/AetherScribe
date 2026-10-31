"""Load AetherScribe's story secret package."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def load_secret_package():
    name = "_aetherscribe_secrets"
    package = sys.modules.get(name)
    if package is None:
        root = Path(__file__).resolve().parent / "story_secrets"
        spec = importlib.util.spec_from_file_location(
            name,
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("AetherScribe secret package could not be loaded.")
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return package


def load_secret_module(module_name: str):
    package = load_secret_package()
    return importlib.import_module(f"{package.__name__}.{module_name}")
