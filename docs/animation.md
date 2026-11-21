# Sensor side-channel animation

This 30-second conceptual animation illustrates the privacy risk studied by
NNEdge-Privacy. All figures, motion, signals, and activity estimates are synthetic.
It contains no participant recordings, model predictions, or benchmark results.

- [Looping GIF](assets/sensor-side-channel.gif): 960 x 540, 10 frames/second.
- [MP4 video](assets/sensor-side-channel.mp4): 1280 x 720, 12 frames/second,
  H.264 with captions drawn into the video, no audio.

## Text transcript

1. **0-6 seconds: The exported interface.** An anonymous person wears a motion
   sensor. The device keeps raw accelerometer and gyroscope signals local while
   releasing binary motion events. The released sequence remains observable.
2. **6-12 seconds: The inference attack.** An observer uses patterns over time,
   such as rhythm, duration, and transitions, to estimate activities. Walking,
   jogging, and sitting are shown as illustrative, uncertain estimates. The
   animation does not assert a deterministic mapping from bits to activities.
3. **12-18 seconds: A possible real-world impact.** Illustrative estimates collect
   into a morning, midday, and evening timeline. This shows how repeated
   observations could expose everyday behavioral patterns; it is not a recorded
   person's day or a demonstrated deployment.
4. **18-24 seconds: G4 / MMBIND.** Auxiliary IMU and skeleton examples are aligned
   during training. At attack time, the adversary receives only IMU input and
   estimates private activity semantics. Private video stays hidden; the scene
   does not depict reconstruction of video or skeleton streams.
5. **24-30 seconds: What the project measures.** The four privacy games vary
   attacker access: binary events, one raw modality, multiple raw modalities, and
   aligned representations. Output detail, temporal continuity, sensor access,
   and auxiliary data affect what an attacker can infer.

The final scene blends into the first for the looping README preview. The MP4
version supports playback controls in a compatible video player.

## Recreate the assets

Use a separate environment for rendering; these packages are not dependencies
of the research experiments. Install DejaVu Sans and DejaVu Sans Mono fonts
(for example, the `fonts-dejavu-core` package on Debian/Ubuntu).

```bash
python3 -m venv .venv-animation
.venv-animation/bin/python -m pip install Pillow imageio-ffmpeg
.venv-animation/bin/python tools/render_illustration.py
```

An existing `ffmpeg` installation is used when available. Otherwise, the script
uses the binary bundled with `imageio-ffmpeg`. Rendering is local and does not
load research datasets or invoke a trained model.

For a storyboard preview without video encoding:

```bash
.venv-animation/bin/python tools/render_illustration.py \
  --stills-only --preview-dir /tmp/nneedge-animation-preview
```

The renderer source is [`tools/render_illustration.py`](../tools/render_illustration.py).
