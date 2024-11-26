import argparse
import json
import math
import subprocess as sp
import sys
import tempfile
import threading  # Importar threading para manejar hilos
import time  # Importar time para medir tiempos
from pathlib import Path

import cairo
import cupy as cp  # CuPy para cálculos en GPU
import numpy as np  # Necesario para interoperabilidad y formato final
import tqdm

_is_main = False


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def fatal(msg):
    if _is_main:
        head = "error: "
        if sys.stderr.isatty():
            head = colorize("error: ", 1)
        print(head + str(msg), file=sys.stderr)
        sys.exit(1)


def read_info(media):
    proc = sp.run([
        'ffprobe', "-loglevel", "panic",
        str(media), '-print_format', 'json', '-show_format', '-show_streams'
    ], capture_output=True)
    if proc.returncode:
        raise IOError(f"{media} does not exist or is of a wrong type.")
    return json.loads(proc.stdout.decode('utf-8'))


def read_audio(audio, seek=None, duration=None):
    info = read_info(audio)
    stream = info['streams'][0]
    if stream["codec_type"] != "audio":
        raise ValueError(f"{audio} should contain only audio.")
    channels = stream['channels']
    samplerate = float(stream['sample_rate'])

    command = ['ffmpeg', '-y']
    command += ['-loglevel', 'panic']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', str(audio)]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le']
    command += ['-']

    proc = sp.run(command, check=True, capture_output=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav.reshape(-1, channels).T, samplerate


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def envelope_gpu(wav, window, stride):
    """
    Extrae la envolvente de la forma de onda `wav` utilizando la GPU.
    Preserva la calidad al no realizar aproximaciones que puedan degradarla.
    """
    wav = cp.pad(wav, window // 2)
    shape = ((len(wav) - window) // stride + 1, window)
    strides = (wav.strides[0] * stride, wav.strides[0])
    frames = cp.lib.stride_tricks.as_strided(wav, shape=shape, strides=strides)
    out = cp.maximum(frames, 0).mean(axis=1)
    out = 1.9 * (sigmoid(2.5 * out) - 0.5)
    return out


def draw_env(envs, out, fg_colors, bg_color, size):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)

    ctx.set_source_rgb(*bg_color)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    K = len(envs)
    T = len(envs[0])
    pad_ratio = 0.1
    width = 1. / (T * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.set_line_width(width)
    for step in range(T):
        for i in range(K):
            half = 0.5 * envs[i][step]
            half /= K
            midrule = (1 + 2 * i) / (2 * K)
            ctx.set_source_rgb(*fg_colors[i])
            ctx.move_to(pad + step * delta, midrule - half)
            ctx.line_to(pad + step * delta, midrule)
            ctx.stroke()
            ctx.set_source_rgba(*fg_colors[i], 0.8)
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
              time_param=0.4,
              oversample=3,
              fg_color=(.2, .2, .2),
              fg_color2=(.5, .3, .6),
              bg_color=(1, 1, 1),
              size=(400, 400),
              stereo=False,
              ):
    start_time = time.time()  # Tiempo de inicio total
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        fatal(err)
        raise

    wavs = []
    if stereo:
        assert wav.shape[0] == 2, 'stereo requires stereo audio file'
        wavs.append(wav[0])
        wavs.append(wav[1])
    else:
        wav = wav.mean(0)
        wavs.append(wav)

    for i, wav in enumerate(wavs):
        wavs[i] = wav / wav.std()

    window = int(sr * time_param / bars)
    stride = int(window / oversample)

    # Convertir wavs a CuPy arrays y procesar la envolvente en GPU
    envs = []
    for wav in wavs:
        wav_gpu = cp.array(wav)
        env = envelope_gpu(wav_gpu, window, stride)
        env = cp.pad(env, (bars // 2, 2 * bars))
        envs.append(env)  # Mantener en CuPy para procesamiento posterior

    duration = len(wavs[0]) / sr
    frames = int(rate * duration)
    smooth = cp.hanning(bars)

    print("Generating the frames...")

    # Dividir el proceso en 10 partes y utilizar CuPy Streams
    num_parts = 10
    frames_per_part = frames // num_parts
    remaining_frames = frames % num_parts

    streams = [cp.cuda.Stream() for _ in range(num_parts)]

    # Crear una barra de progreso compartida
    pbar = tqdm.tqdm(total=frames, unit=" frames", ncols=80)
    pbar_lock = threading.Lock()  # Lock para proteger actualizaciones concurrentes

    def process_part(part_idx):
        stream = streams[part_idx]
        start_frame = part_idx * frames_per_part
        end_frame = start_frame + frames_per_part
        if part_idx == num_parts - 1:
            end_frame += remaining_frames  # Añadir los frames restantes al último lote

        with stream:
            for idx in range(start_frame, end_frame):
                pos = (((idx / rate)) * sr) / stride / bars
                off = int(pos)
                loc = pos - off
                denvs = []
                for env in envs:
                    env1 = env[off * bars:(off + 1) * bars]
                    env2 = env[(off + 1) * bars:(off + 2) * bars]
                    maxvol = cp.log10(1e-4 + env2.max()) * 10
                    maxvol = maxvol.item()  # Convertir a escalar de Python
                    speedup = max(0.5, min(2, interpole(-6, 0.5, 0, 2, maxvol)))
                    w = sigmoid(speed * speedup * (loc - 0.5))
                    denv = (1 - w) * env1 + w * env2
                    denv *= smooth
                    denvs.append(cp.asnumpy(denv))
                draw_env(denvs, tmp / f"{idx:06d}.png", (fg_color, fg_color2), bg_color, size)
                with pbar_lock:
                    pbar.update(1)

    # Lanzar los procesos en paralelo
    threads = []
    for i in range(num_parts):
        t = threading.Thread(target=process_part, args=(i,))
        t.start()
        threads.append(t)

    # Esperar a que todos los threads terminen
    for t in threads:
        t.join()

    pbar.close()  # Cerrar la barra de progreso

    end_time = time.time()  # Tiempo de finalización total
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", str(audio.resolve())]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]
    print("Encoding the animation video... ")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "h264_nvenc", "-crf", "10", "-pix_fmt", "yuv420p",
        str(out.resolve())
    ],
           check=True,
           cwd=tmp)
    print(f"Video saved to {out.resolve()}")


def parse_color(colorstr):
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
                  time_param=args.time,
                  fg_color=args.color,
                  fg_color2=args.color2,
                  bg_color=[1. * bool(args.white)] * 3,
                  size=(args.width, args.height),
                  stereo=args.stereo)
        print(f"Video saved to {args.out.resolve()}")


if __name__ == "__main__":
    _is_main = True
    main()
