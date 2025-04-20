"""pytest configuration and fixtures for the *seewav* test‑suite.

Stub for the external *pycairo* dependency so that the library can be imported
in environments where building GTK/Cairo is not possible (e.g. CI runners).
Only the minimal surface required by ``seewav.draw_env`` is provided.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


# -----------------------------------------------------------------------------
# Install a dummy *cairo* module before *seewav* is imported.
# -----------------------------------------------------------------------------


def _install_cairo_stub() -> None:
    """Register a minimal fake implementation of the *cairo* API we need."""

    cairo_stub = types.ModuleType("cairo")

    # Constant used by *seewav.draw_env* when creating a surface.
    cairo_stub.FORMAT_ARGB32 = 0  # type: ignore[attr-defined]

    # Dummy surface ---------------------------------------------------------
    class _DummySurface:  # noqa: D401,D106
        def __init__(self, _fmt: int, _width: int, _height: int):
            pass

        def write_to_png(self, out: str | Path):  # noqa: D401
            # Touch an empty file so that callers can assert on its existence.
            Path(out).touch(exist_ok=True)

    cairo_stub.ImageSurface = _DummySurface  # type: ignore[attr-defined]

    # Dummy drawing context --------------------------------------------------
    class _DummyContext:  # noqa: D401,D106
        def __init__(self, _surface: _DummySurface):
            pass

        def scale(self, *_):
            pass

        def set_source_rgb(self, *_):
            pass

        def rectangle(self, *_):
            pass

        def fill(self):
            pass

        def set_line_width(self, *_):
            pass

        def move_to(self, *_):
            pass

        def line_to(self, *_):
            pass

        def stroke(self):
            pass

        def set_source_rgba(self, *_):
            pass

    cairo_stub.Context = _DummyContext  # type: ignore[attr-defined]

    sys.modules["cairo"] = cairo_stub


_install_cairo_stub()
