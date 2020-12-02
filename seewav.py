#!/usr/bin/env python3
# This is free and unencumbered software released into the public domain. For more detail,
# see the LICENCE file at https://github.com/adefossez/seewav
# Original author: adefossez
"""
Generates a nice waveform visualization from an audio file, save it as a mp4 file.
"""
import argparse
import json
import math
import subprocess as sp
import sys
import tempfile
from pathlib import Path

import cairo
import numpy as np
import tqdm

_is_main = False


def colorize(text, color):
    """
    Wrap `text` with ANSI `color` code. See
    https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def fatal(msg):
    """
    Something bad happened. Does nothing if this module is not __main__.
    Display an error message and abort.
    """
    if _is_main:
        head = "error: "
        if sys.stderr.isatty():
            head = colorize("error: ", 1)
        print(head + str(msg), file=sys.stderr)
        sys.exit(1)


def read_info(media):
    """
    Return some info on the media file.
    """
    proc = sp.run([
        'ffprobe', "-loglevel", "panic",
        str(media), '-print_format', 'json', '-show_format', '-show_streams'
    ],
                  capture_output=True)
    if proc.returncode:
        raise IOError(f"{media} does not exist or is of a wrong type.")
    return json.loads(proc.stdout.decode('utf-8'))


def read_audio(audio, seek=None, duration=None):
    """
    Read the `audio` file, starting at `seek` (or 0) seconds for `duration` (or all)  seconds.
    Returns `float[channels, samples]`.
    """

    info = read_info(audio)
    channels = None
    stream = info['streams'][0]
    if stream["codec_type"] != "audio":
        raise ValueError(f"{audio} should contain only audio.")
    channels = stream['channels']
    samplerate = float(stream['sample_rate'])

    # Good old ffmpeg
    command = ['ffmpeg', '-y']
    command += ['-loglevel', 'panic']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', audio]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le']
    command += ['-']

    proc = sp.run(command, check=True, capture_output=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav.reshape(-1, channels).T, samplerate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def envelope(wav, window, stride):
    """
    Extract the envelope of the waveform `wav` (float[samples]), using average pooling
    with `window` samples and the given `stride`.
    """
    # pos = np.pad(np.maximum(wav, 0), window // 2)
    wav = np.pad(wav, window // 2)
    out = []
    for off in range(0, len(wav) - window, stride):
        frame = wav[off:off + window]
        out.append(np.maximum(frame, 0).mean())
    out = np.array(out)
    # Some form of audio compressor based on the sigmoid.
    out = 1.9 * (sigmoid(2.5 * out) - 0.5)
    return out


def draw_env(env, out, fg_color, bg_color, size):
    """
    Internal function, draw a single frame using cairo and save it to the `out` file as png.
    env is float[bars], representing the height of the envelope to draw. Each entry will be
    represented by a bar.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)

    ctx.set_source_rgb(*bg_color)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    width = 1. / len(env)
    pad_ratio = 0.1
    width = 1. / (len(env) * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.set_line_width(width)
    for step in range(len(env)):
        half = 0.5 * env[step]
        ctx.set_source_rgb(*fg_color)
        ctx.move_to(pad + step * delta, 0.5 - half)
        ctx.line_to(pad + step * delta, 0.5)
        ctx.stroke()
        ctx.set_source_rgba(*fg_color, 0.8)
        ctx.move_to(pad + step * delta, 0.5)
        ctx.line_to(pad + step * delta, 0.5 + 0.9 * half)
        ctx.stroke()

    surface.write_to_png(out)


def interpole(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def visualize(audio,
              tmp,
              out,
              seek=None,
              duration=None,
              rate=60,
              bars=50,
              speed=4,
              time=0.4,
              oversample=3,
              fg_color=(.2, .2, .2),
              bg_color=(1, 1, 1),
              size=(400, 400)):
    """
    Generate the visualisation for the `audio` file, using a `tmp` folder and saving the final
    video in `out`.
    `seek` and `durations` gives the extract location if any.
    `rate` is the framerate of the output video.

    `bars` is the number of bars in the animation.
    `speed` is the base speed of transition. Depending on volume, actual speed will vary
        between 0.5 and 2 times it.
    `time` amount of audio shown at once on a frame.
    `oversample` higher values will lead to more frequent changes.
    `fg_color` is the rgb color to use for the foreground.
    `bg_color` is the rgb color to use for the background.
    `size` is the `(width, height)` in pixels to generate.
    """
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        fatal(err)
        raise
    wav = wav.mean(0)
    wav /= wav.std()

    window = int(sr * time / bars)
    stride = int(window / oversample)
    env = envelope(wav, window, stride)

    duration = len(wav) / sr
    frames = int(rate * duration)
    env = np.pad(env, (bars // 2, 2 * bars))
    smooth = np.hanning(bars)

    print("Generating the frames...")
    for idx in tqdm.tqdm(range(frames), unit=" frames", ncols=80):
        pos = (((idx / rate)) * sr) / stride / bars
        off = int(pos)
        loc = pos - off
        env1 = env[off * bars:(off + 1) * bars]
        env2 = env[(off + 1) * bars:(off + 2) * bars]

        # we want loud parts to be updated faster
        maxvol = math.log10(1e-4 + env2.max()) * 10
        speedup = np.clip(interpole(-6, 0.5, 0, 2, maxvol), 0.5, 2)
        w = sigmoid(speed * speedup * (loc - 0.5))
        denv = (1 - w) * env1 + w * env2
        denv *= smooth
        draw_env(denv, tmp / f"{idx:06d}.png", fg_color, bg_color, size)

    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", audio.resolve()]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]
    print("Encoding the animation video... ")
    # https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-shortest",
        out.resolve()
    ],
           check=True,
           cwd=tmp)


def parse_color(colorstr):
    """
    Given a comma separated rgb(a) colors, returns a 4-tuple of float.
    """
    try:
        r, g, b = [float(i) for i in colorstr.split(",")]
        return r, g, b
    except ValueError:
        fatal("Format for color is 3 floats separated by commas 0.xx,0.xx,0.xx, rgb order")
        raise


def main():
    parser = argparse.ArgumentParser(
        'seewav', description="Generate a nice mp4 animation from an audio file.")
    parser.add_argument("-r", "--rate", type=int, default=60, help="Video framerate.")
    parser.add_argument("-c",
                        "--color",
                        default=[0.03, 0.6, 0.3],
                        type=parse_color,
                        dest="color",
                        help="Color of the bars as `r,g,b` in [0, 1].")
    parser.add_argument("--white", action="store_true",
                        help="Use white background. Default is black.")
    parser.add_argument("-B",
                        "--bars",
                        type=int,
                        default=50,
                        help="Number of bars on the video at once")
    parser.add_argument("-O", "--oversample", type=float, default=4,
                        help="Lower values will feel less reactive.")
    parser.add_argument("-T", "--time", type=float, default=0.4,
                        help="Amount of audio shown at once on a frame.")
    parser.add_argument("-S", "--speed", type=float, default=4,
                        help="Higher values means faster transitions between frames.")
    parser.add_argument("-W",
                        "--width",
                        type=int,
                        default=480,
                        help="width in pixels of the animation")
    parser.add_argument("-H",
                        "--height",
                        type=int,
                        default=300,
                        help="height in pixels of the animation")
    parser.add_argument("-s", "--seek", type=float, help="Seek to time in seconds in video.")
    parser.add_argument("-d", "--duration", type=float, help="Duration in seconds from seek time.")
    parser.add_argument("audio", type=Path, help='Path to audio file')
    parser.add_argument("out",
                        type=Path,
                        nargs='?',
                        default=Path('out.mp4'),
                        help='Path to output file. Default is ./out.mp4')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        visualize(args.audio,
                  Path(tmp),
                  args.out,
                  seek=args.seek,
                  duration=args.duration,
                  rate=args.rate,
                  bars=args.bars,
                  speed=args.speed,
                  oversample=args.oversample,
                  time=args.time,
                  fg_color=args.color,
                  bg_color=[1. * bool(args.white)] * 3,
                  size=(args.width, args.height))


if __name__ == "__main__":
    _is_main = True
    main()
