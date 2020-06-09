# SeeWav: animation generator for audio waveforms

## Installation

You will need a decently recent version of python, most likely 3.6 or 3.7.
You will need `ffmpeg` installed somewhere with sufficient codec support. If you are using conda,
install it with `conda install -c conda-forge ffmpeg`.

```bash
# Optional, if you use conda, otherwise find a way to install it.
# conda install -c conda-forge ffmpeg
pip3 install seewave
```

## Usage


```bash
python3 -m seewave AUDIO_FILE [OUTPUT_FILE]
```

By default, outputs to `out.mp4`.
