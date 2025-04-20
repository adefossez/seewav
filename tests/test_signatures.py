"""Meta‑tests ensuring the public *seewav* API is fully type‑annotated.

The goal is to guarantee that every public function is equipped with explicit
parameter and return annotations so that static type‑checkers (e.g. *mypy*)
can provide useful feedback to downstream users.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import seewav


_MODULE: ModuleType = seewav


def _is_public(name: str) -> bool:
    """Return *True* if *name* should be considered part of the public API."""

    return not name.startswith("_")  # exclude dunders / private helpers


def test_all_public_functions_are_typed():  # noqa: D401
    """Every public function must declare type annotations for all arguments and the return value."""

    failures: list[str] = []
    for name, obj in inspect.getmembers(_MODULE, inspect.isfunction):
        if not _is_public(name):
            continue

        sig = inspect.signature(obj)

        # Check parameter annotations ----------------------------------------------------------
        for param in sig.parameters.values():
            if param.annotation is inspect.Parameter.empty:
                failures.append(
                    f"Function '{name}': missing annotation for parameter '{param.name}'."
                )

        # Check return annotation --------------------------------------------------------------
        if sig.return_annotation is inspect.Signature.empty:
            failures.append(f"Function '{name}': missing return type annotation.")

    if failures:
        joined = "\n".join(failures)
        raise AssertionError(f"Missing type annotations found:\n{joined}")
