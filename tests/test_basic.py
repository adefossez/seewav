"""Core logic tests for *seewav*.

These tests validate the mathematical helpers that do not require the real
`pycairo` bindings.  A lightweight cairo stub is installed globally in
``tests/conftest.py``.
"""

from __future__ import annotations

import math
import subprocess as sp
from pathlib import Path

import numpy as np
import pytest


import seewav


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_sine_wav(path: Path, duration: float = 0.5, sr: int = 16000) -> None:
    """Generate a tiny mono sine wave with *ffmpeg* for read‑audio tests."""

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-loglevel",
        "panic",
        "-f",
        "lavfi",
        f"-i",
        f"sine=frequency=1000:duration={duration}",
        "-ac",
        "1",
        "-ar",
        str(sr),
        path.as_posix(),
    ]
    sp.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_sigmoid_monotonic():
    x = np.linspace(-4, 4, 100)
    y = seewav.sigmoid(x)
    # Sigmoid should be strictly increasing.
    assert np.all(np.diff(y) > 0)
    # Values are bounded between 0 and 1.
    assert 0 < y.min() < 1
    assert 0 < y.max() < 1


@pytest.mark.parametrize(
    "x1,y1,x2,y2,x,expected", [(0, 0, 10, 10, 5, 5), (3, 2, 7, 6, 5, 4)]
)
def test_interpole(x1, y1, x2, y2, x, expected):
    out = seewav.interpole(x1, y1, x2, y2, x)
    assert math.isclose(out, expected, rel_tol=1e-6)


def test_parse_color_valid():
    assert seewav.parse_color("0.1,0.2,0.3") == (0.1, 0.2, 0.3)


def test_parse_color_invalid():
    with pytest.raises(Exception):
        seewav.parse_color("bad,color")


def test_envelope_constant_signal():
    const = np.ones(100, dtype=np.float32)
    out = seewav.envelope(const, window=10, stride=5)
    # First frame is affected by zero‑padding, skip it.
    steady = out[1:]
    expected_value = 1.9 * (seewav.sigmoid(2.5 * 1) - 0.5)
    assert np.allclose(steady, expected_value, atol=1e-6)


def test_draw_env_runs(tmp_path):
    env = np.linspace(0, 1, 10, dtype=np.float32)
    out_img = tmp_path / "frame.png"
    # Function should execute without raising and create the PNG.
    seewav.draw_env([env], out_img, [(0.2, 0.2, 0.2)], (1, 1, 1), size=(100, 100))
    assert out_img.exists() and out_img.stat().st_size >= 0


@pytest.mark.skipif(
    sp.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg not available",
)
def test_read_audio_sine(tmp_path):
    audio_path = tmp_path / "tone.wav"
    _make_sine_wav(audio_path)
    wav, sr = seewav.read_audio(audio_path)
    # Expect mono (1 channel) and sample‑rate == 16000.
    assert wav.shape[0] == 1
    assert sr == 16000
