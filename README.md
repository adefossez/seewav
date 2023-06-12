# SeeWav: animation generator for audio waveforms

SeeWav can generate some nice animations for your waveform.
For a demo, click on the image:

<p align="center">
<a href="https://ai.honu.io/misc/seewav.mp4">
<img src="./seewav.png" alt="Demo of seewav"></a></p>

## Installation

You will need Python 3.7.
You will need `ffmpeg` installed with codec support for `libx264` and `aac`.
On Mac OS X with Homebrew, run `brew install ffmpeg`, on Ubuntu `sudo apt-get install ffmpeg`.
If you are using Anaconda, you can also do `conda install -c conda-forge ffmpeg`.


```bash
pip3 install seewav
```

## Usage


```bash
seewav AUDIO_FILE [OUTPUT_FILE]
```
By default, outputs to `out.mp4`. Available options:

```bash
usage: seewav [-h] [-r RATE] [--stereo] [-c COLOR] [-c2 COLOR2] [--white]
              [-B BARS] [-O OVERSAMPLE] [-T TIME] [-S SPEED] [-W WIDTH]
              [-H HEIGHT] [-s SEEK] [-d DURATION]
              audio [out]

Generate a nice mp4 animation from an audio file.

positional arguments:
  audio                 Path to audio file
  out                   Path to output file. Default is ./out.mp4

optional arguments:
  -h, --help            show this help message and exit
  -r RATE, --rate RATE  Video framerate.
  --stereo              Create 2 waveforms for stereo files.
  -c COLOR, --color COLOR
                        Color of the bars as `r,g,b` in [0, 1].
  -c2 COLOR2, --color2 COLOR2
                        Color of the second waveform as `r,g,b` in [0, 1] (for
                        stereo).
  --white               Use white background. Default is black.
  -B BARS, --bars BARS  Number of bars on the video at once
  -O OVERSAMPLE, --oversample OVERSAMPLE
                        Lower values will feel less reactive.
  -T TIME, --time TIME  Amount of audio shown at once on a frame.
  -S SPEED, --speed SPEED
                        Higher values means faster transitions between frames.
  -W WIDTH, --width WIDTH
                        width in pixels of the animation
  -H HEIGHT, --height HEIGHT
                        height in pixels of the animation
  -s SEEK, --seek SEEK  Seek to time in seconds in video.
  -d DURATION, --duration DURATION
                        Duration in seconds from seek time.
```
