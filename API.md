Module seewav
-------------
Generates a nice waveform visualization from an audio file, save it as a mp4 file.

Functions
---------
colorize(text, color)
    Wrap `text` with ANSI `color` code. See
    https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences

draw_env(env, out, fg_color, bg_color, size)
    Internal function, draw a single frame using cairo and save it to the `out` file as png.
    env is float[bars], representing the height of the envelope to draw. Each entry will be
    represented by a bar.

envelope(wav, window, stride)
    Extract the envelope of the waveform `wav` (float[samples]), using average pooling
    with `window` samples and the given `stride`.

fatal(msg)
    Something bad happened. Does nothing if this module is not __main__.
    Display an error message and abort.

main()

parse_color(colorstr)
    Given a comma separated rgb(a) colors, returns a 4-tuple of float.

read_audio(audio, seek=None, duration=None)
    Read the `audio` file, starting at `seek` (or 0) seconds for `duration` (or all)  seconds.
    Returns `float[channels, samples]`.

read_info(media)
    Return some info on the media file.

sigmoid(x)

visualize(audio, tmp, out, seek=None, duration=None, rate=60, bars=50, base_speed=4, oversample=3, fg_color=(0.2, 0.2, 0.2, 0.8), bg_color=[1, 1, 1, 1], size=(400, 400))
    Generate the visualisation for the `audio` file, using a `tmp` folder and saving the final
    video in `out`.
    `seek` and `durations` gives the extract location if any.
    `rate` is the framerate of the output video.

    `bars` is the number of bars in the animation.
    `base_speed` is the base speed of transition. Depending on volume, actual speed will vary
        between 0.5 and 2 times it.
    `oversample` higher values will lead to more frequent changes.
    `fg_color` is the rgba color to use for the foreground.
    `bg_color` is the rgba color to use for the background.
    `size` is the `(width, height)` in pixels to generate.
