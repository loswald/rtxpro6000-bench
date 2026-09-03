"""
families/__init__.py - the family registry.

A family is any module `families/<name>.py` whose name does not start with an underscore.
`load(name)` imports it and fills in the defaults from families/_base.py, so a family only has to
define NAME, prepare() and score() (see _base.py for the full interface).
"""
from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Iterable

from . import _base

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# design-spec dispatch order (slowest chains first); unknown families use their PRIORITY attribute (default 50)
KNOWN_PRIORITY = {"tools": 10, "code": 20, "math": 30, "longctx": 40, "knowledge": 50, "ifeval": 60}


def discover() -> list[str]:
    """Names of all family modules present on disk (sorted)."""
    return sorted(m.name for m in pkgutil.iter_modules([PKG_DIR]) if not m.name.startswith("_"))


def load(name: str, require: Iterable[str] = ()):
    """Import families.<name>, apply defaults, check the required hooks are real implementations."""
    if name.startswith("_") and name != "_base":
        raise ValueError(f"{name!r} is not a family module")
    mod = importlib.import_module(f"{__name__}.{name}")
    _base.apply_defaults(mod)
    if mod.NAME != name:
        raise ValueError(f"family module {name}.py declares NAME={mod.NAME!r}")
    for hook in require:
        fn = getattr(mod, hook, None)
        if fn is None or getattr(fn, "__module__", "") == _base.__name__:
            raise NotImplementedError(f"family {name!r} does not implement {hook}()")
    return mod


def priority(mod) -> int:
    return KNOWN_PRIORITY.get(mod.NAME, getattr(mod, "PRIORITY", 50))


def default_families() -> list[str]:
    """All discovered, non-hidden families in dispatch order."""
    mods = [load(n) for n in discover()]
    mods = [m for m in mods if not getattr(m, "HIDDEN", False)]
    return [m.NAME for m in sorted(mods, key=lambda m: (priority(m), m.NAME))]


def resolve(names: Iterable[str] | None, require: Iterable[str] = ()) -> list:
    """Modules for the requested names (None/'all' -> default set), sorted by dispatch priority."""
    if not names or list(names) == ["all"]:
        names = default_families()
    mods = [load(n.strip(), require) for n in names if n.strip()]
    return sorted(mods, key=lambda m: (priority(m), m.NAME))
