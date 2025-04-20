"""Argument‑validation tests for *seewav* helper utilities.

These checks make sure that common user errors are properly caught and that
the helpers behave in a predictable way.
"""

from __future__ import annotations
import pytest
import subprocess as _sp
import seewav


# ---------------------------------------------------------------------------
# _parse_size_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("640x360", (640, 360)),
        ("800X600", (800, 600)),  # upper‑case “X” allowed
        ("120x120", (120, 120)),
    ],
)
def test_parse_size_token_valid(token: str, expected: tuple[int, int]):  # noqa: D401
    """Valid *WxH* tokens are parsed into integer tuples."""

    assert seewav.parse_size_token(token) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "640",  # missing height
        "x360",  # missing width
        "-10x100",  # negative number
        "640x0",  # zero
        "abcx200",
        "640xdef",
        "640*480",
    ],
)
def test_parse_size_token_invalid(bad: str):  # noqa: D401
    """Malformed tokens should raise *ValueError*."""

    with pytest.raises(ValueError):
        seewav.parse_size_token(bad)


# ---------------------------------------------------------------------------
# parse_color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("0,0,0", (0.0, 0.0, 0.0)),
        ("0.1,0.2,0.3", (0.1, 0.2, 0.3)),
    ],
)
def test_parse_color_valid(token: str, expected: tuple[float, float, float]):
    assert seewav.parse_color(token) == expected


@pytest.mark.parametrize(
    "bad",
    ["", "bad", "0.1,0.2", "0.1,0.2,0.3,0.4"],
)
def test_parse_color_invalid(bad: str):
    with pytest.raises(Exception):
        seewav.parse_color(bad)


# ---------------------------------------------------------------------------
# read_audio error handling (ffprobe required)
# ---------------------------------------------------------------------------


_HAS_FFPROBE = _sp.run(["which", "ffprobe"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _HAS_FFPROBE, reason="ffprobe not available")
def test_read_audio_invalid(tmp_path):
    """Non‑audio files should raise *IOError*."""

    bogus = tmp_path / "dummy.txt"
    bogus.write_text("not audio")
    with pytest.raises(Exception):
        seewav.read_audio(bogus)


# ---------------------------------------------------------------------------
# visualize channel mismatch (ffmpeg required to generate a dummy wav)
# ---------------------------------------------------------------------------


_HAS_FFMPEG = _sp.run(["which", "ffmpeg"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not available")
def test_visualize_stereo_flag_mismatch(tmp_path):
    """Assert that *visualize* detects mismatch between --stereo flag and audio channels."""

    # Create a mono sine wave.
    audio_path = tmp_path / "mono.wav"
    seewav.sp.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "panic",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=0.1",
            "-ac",
            "1",
            audio_path.as_posix(),
        ],
        check=True,
    )

    with pytest.raises(AssertionError):
        seewav.visualize(
            audio_path,
            tmp_path,
            tmp_path / "out.mp4",
            size=(100, 100),
            stereo=True,  # requesting stereo on a mono file should fail
        )
