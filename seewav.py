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
import PIL.Image as Image
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
    
def pil_to_surface(image):
    """
    Internal function, create cairo surface from Pillow image
    """
    if 'A' not in image.getbands():
        image.putalpha(int(256))
    return cairo.ImageSurface.create_for_data(bytearray(image.tobytes('raw', 'BGRa')), cairo.FORMAT_ARGB32, image.width, image.height)

def draw_env(envs, out, fg_colors, fg_opacity, bg_color, bg_image, center, size):
    """
    Internal function, draw a single frame (two frames for stereo) using cairo and save
    it to the `out` file as png. envs is a list of envelopes over channels, each env
    is a float[bars] representing the height of the envelope to draw. Each entry will
    be represented by a bar.
    """
    if bg_image is None:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
        offset = [0, 0]
    else:
        surface = pil_to_surface(bg_image)
        # offset needs to be relative to the size of the surface, not the size of the background image
        offset = [
            (bg_image.width * center[0] - size[0] / 2) / size[0],
            (bg_image.height * center[1] - size[1] / 2) / size[1]
        ]
    ctx = cairo.Context(surface)
    ctx.scale(*size)
    if bg_image is None:
        ctx.set_source_rgb(*bg_color)
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

    K = len(envs) # Number of waves to draw (waves are stacked vertically)
    T = len(envs[0]) # Numbert of time steps
    pad_ratio = 0.1 # spacing ratio between 2 bars
    width = 1. / (T * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.translate(*offset)
    ctx.set_line_width(width)
    for step in range(T):
        for i in range(K):
            half = 0.5 * envs[i][step] # (semi-)height of the bar
            half /= K # as we stack K waves vertically
            midrule = (1+2*i)/(2*K) # midrule of i-th wave
            ctx.set_source_rgba(*fg_colors[i], fg_opacity)
            ctx.move_to(pad + step * delta, midrule - half)
            ctx.line_to(pad + step * delta, midrule)
            ctx.stroke()
            ctx.set_source_rgba(*fg_colors[i], fg_opacity - fg_opacity / 5)
            ctx.move_to(pad + step * delta, midrule)
            ctx.line_to(pad + step * delta, midrule + 0.9 * half)
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
              fg_color2=(.5, .3, .6),
              fg_opacity=1,
              bg_color=(1, 1, 1),
              bg_image=None,
              center=(.5, .5),
              size=(400, 300),
              stereo=False,
              ):
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
    `fg_color2` is the rgb color to use for the second wav if stereo is set.
    `bg_color` is the rgb color to use for the background.
    `bg_image` is the path to the PNG image to use for the background.
    `size` is the `(width, height)` in pixels to generate.
    `stereo` is whether to create 2 waves.
    """
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        fatal(err)
        raise

    output_size = size
    image = None
    if bg_image is not None:
        try:
            image = Image.open(bg_image)
        except (IOError, ValueError) as err:
            fatal(err)
            raise
        # resize image to be compatible with ffmpeg
        if image.width % 2 == 1:
            image = image.resize((image.width + 1, image.height))
        if image.height % 2 == 1:
            image = image.resize((image.width, image.height + 1))
        output_size = image.width, image.height

    # wavs is a list of wav over channels
    wavs = []
    if stereo:
        assert wav.shape[0] == 2, 'stereo requires stereo audio file'
        wavs.append(wav[0])
        wavs.append(wav[1])
    else:
        wav = wav.mean(0)
        wavs.append(wav)

    for i, wav in enumerate(wavs):
        wavs[i] = wav/wav.std()

    window = int(sr * time / bars)
    stride = int(window / oversample)
    # envs is a list of env over channels
    envs = []
    for wav in wavs:
        env = envelope(wav, window, stride)
        env = np.pad(env, (bars // 2, 2 * bars))
        envs.append(env)

    duration = len(wavs[0]) / sr
    frames = int(rate * duration)
    smooth = np.hanning(bars)

    print("Generating the frames...")
    for idx in tqdm.tqdm(range(frames), unit=" frames", ncols=80):
        pos = (((idx / rate)) * sr) / stride / bars
        off = int(pos)
        loc = pos - off
        denvs = []
        for env in envs:
            env1 = env[off * bars:(off + 1) * bars]
            env2 = env[(off + 1) * bars:(off + 2) * bars]

            # we want loud parts to be updated faster
            maxvol = math.log10(1e-4 + env2.max()) * 10
            speedup = np.clip(interpole(-6, 0.5, 0, 2, maxvol), 0.5, 2)
            w = sigmoid(speed * speedup * (loc - 0.5))
            denv = (1 - w) * env1 + w * env2
            denv *= smooth
            denvs.append(denv)
        draw_env(denvs, tmp / f"{idx:06d}.png", (fg_color, fg_color2), fg_opacity, bg_color, image, center, size)

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
        str(rate), "-f", "image2", "-s", f"{output_size[0]}x{output_size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
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

def parse_coords(coordsstr):
    """
    Given a comma separated float x and y coords, returns a tuple of float.
    """
    try:
        x, y = [float(i) for i in coordsstr.split(",")]
        return x, y
    except ValueError:
        fatal("Format for coords is 2 floats separated by commas 0.x,0.y, xy order")
        raise


def main():
    parser = argparse.ArgumentParser(
        'seewav', description="Generate a nice mp4 animation from an audio file.")
    parser.add_argument("-r", "--rate", type=int, default=60, help="Video framerate.")
    parser.add_argument("--stereo", action='store_true',
                        help="Create 2 waveforms for stereo files.")
    parser.add_argument("-c",
                        "--color",
                        default=[0.03, 0.6, 0.3],
                        type=parse_color,
                        dest="color",
                        help="Color of the bars as `r,g,b` in [0, 1].")
    parser.add_argument("-c2",
                        "--color2",
                        default=[0.5, 0.3, 0.6],
                        type=parse_color,
                        dest="color2",
                        help="Color of the second waveform as `r,g,b` in [0, 1] (for stereo).")
    parser.add_argument("-o", "--opacity", type=float, default=1,
                        help="The opacity of the waveform on the background.")
    parser.add_argument("-b",
                        "--background",
                        default=[0, 0, 0],
                        type=parse_color,
                        dest="background",
                        help="Set the background. r,g,b` in [0, 1]. Default is black (0,0,0).")
    parser.add_argument("--white", action="store_true",
                        help="Use white background. Default is black.")
    parser.add_argument("-i",
                        "--image",
                        dest="image",
                        help="Set the background image.")
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
    parser.add_argument("-C",
                        "--center",
                        default=[0.5, 0.5],
                        type=parse_coords,
                        dest="center",
                        help="The center of the bars relative to the image.")
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
                  fg_color2=args.color2,
                  fg_opacity=args.opacity,
                  bg_color=[1.] * 3 if bool(args.white) else args.background,
                  bg_image=args.image,
                  center=args.center,
                  size=(args.width, args.height),
                  stereo=args.stereo)


if __name__ == "__main__":
    _is_main = True
    main()
