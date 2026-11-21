#!/usr/bin/env python3
"""Render the synthetic README explainer as an MP4 and a looping GIF.

Install rendering dependencies in a separate environment:
    python -m pip install Pillow imageio-ffmpeg
Run from the repository root:
    python tools/render_illustration.py

An installed ffmpeg is also supported. No datasets, models, network calls, or
participant recordings are used. All motion, signals, and estimates are drawn
procedurally. The animation explains threat models rather than reporting results.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
SCENE_SECONDS = 6
DURATION = 5 * SCENE_SECONDS
SCALE = 2
BG = "#0b1324"
PANEL = "#132239"
EDGE = "#293c54"
INK = "#edf5ff"
MUTED = "#a9bbce"
TEAL = "#63e1d2"
CORAL = "#ff9a87"
GOLD = "#f7ce76"
BLUE = "#97b8ff"
COLORS = [TEAL, CORAL, GOLD, BLUE, TEAL]
FONT_CACHE: dict[tuple[int, bool, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False, mono: bool = False):
    key = (size, bold, mono)
    if key not in FONT_CACHE:
        family = "DejaVuSansMono" if mono else "DejaVuSans"
        name = family + ("-Bold" if bold else "") + ".ttf"
        try:
            FONT_CACHE[key] = ImageFont.truetype(name, size * SCALE)
        except OSError as error:
            raise SystemExit(
                "Install DejaVu Sans and DejaVu Sans Mono fonts to render this animation."
            ) from error
    return FONT_CACHE[key]


class Canvas:
    def __init__(self):
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
        self.draw = ImageDraw.Draw(self.image)

    def box(self, bounds, fill=PANEL, outline=EDGE, radius=20, width=1):
        self.draw.rounded_rectangle(
            [int(v * SCALE) for v in bounds], radius=radius * SCALE,
            fill=fill, outline=outline, width=width * SCALE,
        )

    def line(self, points, fill=MUTED, width=2):
        self.draw.line(
            [(int(x * SCALE), int(y * SCALE)) for x, y in points],
            fill=fill, width=width * SCALE, joint="curve",
        )

    def circle(self, x, y, radius, fill, outline=None, width=1):
        self.draw.ellipse(
            [(x - radius) * SCALE, (y - radius) * SCALE,
             (x + radius) * SCALE, (y + radius) * SCALE],
            fill=fill, outline=outline, width=width * SCALE,
        )

    def text(self, xy, value, size=22, fill=INK, bold=False, mono=False, anchor=None):
        self.draw.text(
            (int(xy[0] * SCALE), int(xy[1] * SCALE)), value,
            font=font(size, bold, mono), fill=fill, anchor=anchor,
        )

    def centered(self, x, y, value, size=22, fill=INK, bold=False):
        self.text((x, y), value, size, fill, bold, anchor="mt")

    def arrow(self, x1, x2, y, color=TEAL, time=0):
        self.line([(x1, y), (x2, y)], EDGE, 3)
        self.line([(x2 - 9, y - 7), (x2, y), (x2 - 9, y + 7)], color, 3)
        for i in range(2):
            x = x1 + ((time * 0.6 + i * 0.5) % 1) * (x2 - x1 - 12)
            self.circle(x, y, 4, color)

    def finish(self):
        return self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def person(c, x, y, time, activity="walking", size=1.0, color=TEAL):
    """Draw an anonymous, animated figure; (x, y) is its hip position."""
    phase = math.sin(time * (9 if activity == "jogging" else 5))
    bob = abs(phase) * (6 if activity == "jogging" else 2)
    if activity == "sitting":
        bob = 0
        joints = [(0, 0), (32, 6), (36, 52), (-8, 0), (18, 15), (20, 54)]
        c.box((x - 29 * size, y + 3 * size, x + 27 * size, y + 14 * size),
              EDGE, EDGE, 3)
        c.line([(x - 23 * size, y + 8 * size), (x - 23 * size, y + 63 * size)], EDGE, 5)
        arms = [(-4, -52), (23, -25), (44, -28), (4, -52), (30, -21), (48, -25)]
    else:
        stride = phase * (36 if activity == "jogging" else 26)
        joints = [(0, 0), (stride, 35), (stride * 1.4, 73 - max(phase, 0) * 14),
                  (0, 0), (-stride, 35), (-stride * 1.4, 73 + min(phase, 0) * 14)]
        arms = [(0, -55), (-stride, -29), (-stride * 1.35, -4),
                (0, -55), (stride, -29), (stride * 1.35, -4)]

    def points(coords):
        return [(x + px * size, y + (py - bob) * size) for px, py in coords]

    for part in [joints[:3], joints[3:], arms[:3], arms[3:]]:
        c.line(points(part), color, max(3, round(8 * size)))
        for px, py in points(part):
            c.circle(px, py, 4 * size, color)
    c.line(points([(0, -65), (0, 0)]), color, max(3, round(11 * size)))
    c.circle(x, y - (91 + bob) * size, 18 * size, color)
    wrist = points([arms[2]])[0]
    c.box((wrist[0] - 9, wrist[1] - 8, wrist[0] + 9, wrist[1] + 8), BG, INK, 4, 2)
    c.circle(wrist[0], wrist[1], 3, INK)


def pulse(c, x, y, width, height, time, color=TEAL):
    """A synthetic event sequence, not a classifier input or experimental trace."""
    bits = [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    points = []
    steps = 42
    for i in range(steps):
        bit = bits[(i + int(time * 2)) % len(bits)]
        px = x + width * i / steps
        level = y + (height * 0.18 if bit else height * 0.82)
        if points:
            points.append((px, points[-1][1]))
        points.extend([(px, level), (x + width * (i + 1) / steps, level)])
    for ratio in [0.18, 0.5, 0.82]:
        c.line([(x, y + height * ratio), (x + width, y + height * ratio)], EDGE, 1)
    c.line(points, color, 3)
    scan = x + (time * 0.2 % 1) * width
    c.line([(scan, y - 5), (scan, y + height + 5)], INK, 1)


def waveform(c, x, y, width, height, time, color=TEAL):
    points = []
    for i in range(101):
        px = x + width * i / 100
        value = math.sin(i * 0.2 + time * 4) * math.sin(i * 0.09) * height * 0.38
        points.append((px, y + height / 2 + value))
    c.line([(x, y + height / 2), (x + width, y + height / 2)], EDGE)
    c.line(points, color, 3)


def network(c, x, y, time, color=CORAL):
    columns = [[-35, 0, 35], [-52, -18, 18, 52], [-30, 30]]
    for column in range(2):
        for ay in columns[column]:
            for by in columns[column + 1]:
                c.line([(x + column * 55, y + ay),
                        (x + (column + 1) * 55, y + by)], EDGE, 2)
    for column, values in enumerate(columns):
        for row, offset in enumerate(values):
            active = int(time * 4 + row) % 3 == column
            c.circle(x + column * 55, y + offset, 8, color if active else MUTED)


HEADINGS = [
    ("01 / THE EXPORTED INTERFACE", "A tiny output can reveal a bigger story.",
     "A wearable can release just one motion event at a time."),
    ("02 / THE INFERENCE ATTACK", "Patterns over time become an attack surface.",
     "An observer uses the released sequence to estimate private activities."),
    ("03 / A POSSIBLE REAL-WORLD IMPACT", "Small inferences can add up to a routine.",
     "Repeated estimates could expose when someone is active or resting."),
    ("04 / G4: MULTIMODAL BINDING", "Aligned features can reveal private semantics.",
     "Auxiliary multimodal data can strengthen activity inference through a shared representation."),
    ("05 / WHAT THE PROJECT MEASURES", "Privacy depends on what a device exposes.",
     "NNEdge-Privacy evaluates sensor interfaces, attacker access, and context."),
]
TAKEAWAYS = [
    "Potential leakage starts at the exported interface.",
    "A coarse event stream can preserve behavioral structure.",
    "Illustrative estimates show a possible privacy impact, not a recorded person's day.",
    "G4 / MMBIND: activity semantics inferred from IMU-only inputs at attack time.",
    "Evaluate what an attacker can infer, even when processing stays local.",
]


def frame_base(scene, local):
    c = Canvas()
    accent = COLORS[scene]
    c.box((44, 29, 77, 62), BG, TEAL, 8, 2)
    c.line([(50, 47), (55, 47), (59, 38), (65, 54), (71, 45)], TEAL, 2)
    c.text((91, 29), "NNEdge-Privacy", 24, bold=True)
    c.box((947, 29, 1232, 63), BG, EDGE, 17)
    c.centered(1089, 36, "RESEARCH / ILLUSTRATION", 15, MUTED, True)
    c.line([(48, 84), (1232, 84)], EDGE, 1)
    eyebrow, title, subtitle = HEADINGS[scene]
    c.text((48, 108), eyebrow, 17, accent, True)
    c.text((45, 145), title, 42, INK, True)
    c.text((48, 210), subtitle, 22, MUTED)
    c.box((48, 612, 1232, 657), PANEL, PANEL, 12)
    c.circle(70, 634, 4, accent)
    c.text((88, 619), TAKEAWAYS[scene], 22)
    c.text((48, 676), "Conceptual animation  /  Synthetic signals  /  No participant data", 16, MUTED)
    c.text((1232, 676), "PoPETs 2026  /  Research artifact", 16, MUTED, anchor="rt")
    for i in range(5):
        start = 48 + i * 240
        c.line([(start, 709), (start + 222, 709)], EDGE, 3)
        if i < scene:
            c.line([(start, 709), (start + 222, 709)], accent, 3)
        elif i == scene:
            c.line([(start, 709), (start + 222 * local / SCENE_SECONDS, 709)], accent, 3)
    return c


def scene_output(c, t):
    c.box((48, 270, 409, 586))
    c.box((453, 270, 1232, 586))
    c.text((70, 290), "WEARABLE / ON-DEVICE ML", 17, MUTED, True)
    c.line([(115, 509), (345, 509)], EDGE, 3)
    person(c, 231, 431, t, size=0.88)
    c.text((71, 528), "Raw accelerometer + gyroscope", 18)
    c.text((71, 553), "remain on the device", 18, TEAL)
    c.arrow(413, 448, 409, TEAL, t)
    c.text((478, 290), "RELEASED MOTION OUTPUT", 17, TEAL, True)
    c.text((478, 330), "0 = no event     1 = motion event", 20, MUTED)
    pulse(c, 478, 373, 723, 93, t)
    bits = "00110100011100101000"
    offset = int(t * 2) % len(bits)
    c.text((478, 492), (bits + bits)[offset:offset + 20], 32, TEAL, mono=True)
    c.text((478, 547), "The output sequence remains observable.", 21)


def scene_attack(c, t):
    for left, right in [(48, 365), (449, 785), (869, 1232)]:
        c.box((left, 270, right, 586))
    c.text((71, 290), "OBSERVED EVENTS", 18, MUTED, True)
    pulse(c, 75, 363, 264, 93, t, CORAL)
    c.text((74, 500), "01001 11000 10101", 22, CORAL, mono=True)
    c.text((74, 545), "A sequence, over time", 19, MUTED)
    c.arrow(373, 439, 422, CORAL, t)
    c.text((475, 290), "ATTACKER'S MODEL", 18, CORAL, True)
    network(c, 559, 410, t)
    c.centered(617, 499, "Rhythm / duration", 21)
    c.centered(617, 532, "Transitions / context", 20, MUTED)
    c.arrow(794, 859, 422, CORAL, t)
    c.text((892, 290), "ILLUSTRATIVE ESTIMATES", 17, MUTED, True)
    current = min(int(t / 2), 2)
    for index, label in enumerate(["Walking?", "Jogging?", "Sitting?"]):
        y = 345 + index * 70
        active = index == current
        c.box((895, y, 1206, y + 54), BG, CORAL if active else EDGE, 12, 2)
        c.circle(918, y + 27, 5, CORAL if active else EDGE)
        c.text((938, y + 10), label, 25, INK if active else MUTED, active)


def scene_routine(c, t):
    current = min(int(t / 2), 2)
    labels = ["Walking", "Jogging", "Sitting"]
    c.box((48, 270, 362, 586))
    c.text((72, 290), "ILLUSTRATIVE DAY", 18, MUTED, True)
    c.line([(106, 510), (312, 510)], EDGE, 3)
    person(c, 192, 435, t, labels[current].lower(), 0.85, GOLD)
    c.centered(205, 535, labels[current] + " estimate", 22, GOLD, True)
    c.box((405, 270, 1232, 586))
    c.text((432, 290), "WHAT AN OBSERVER COULD LEARN", 18, GOLD, True)
    c.text((432, 332), "A pattern of activity across the day", 25, bold=True)
    c.line([(462, 420), (1176, 420)], EDGE, 4)
    for index, (period, label) in enumerate(zip(["Morning", "Midday", "Evening"], labels)):
        x = 433 + index * 257
        active = index <= current
        c.centered(x + 111, 381, period, 19, MUTED)
        c.box((x, 428, x + 227, 495), BG, GOLD if active else EDGE, 12, 2)
        c.centered(x + 113, 446, label if active else "...", 25,
                   GOLD if active else MUTED, active)
    c.text((433, 537), "Potential exposure: everyday behavioral patterns", 22)


def scene_binding(c, t):
    c.box((48, 270, 584, 586))
    c.box((628, 270, 1232, 586))
    c.text((72, 290), "AUXILIARY TRAINING", 18, BLUE, True)
    c.text((72, 328), "Cross-modal pairs connect motion and labels.", 20)
    c.box((73, 374, 256, 525), BG, EDGE, 12)
    waveform(c, 92, 400, 143, 58, t, BLUE)
    c.centered(164, 482, "IMU", 21, BLUE, True)
    c.arrow(270, 347, 439, BLUE, t)
    c.line([(279, 432), (270, 439), (279, 446)], BLUE, 3)
    c.centered(308, 466, "ALIGN", 12, MUTED, True)
    c.box((361, 374, 559, 525), BG, EDGE, 12)
    person(c, 463, 449, t * 0.5, size=0.4, color=BLUE)
    c.centered(460, 494, "Skeleton", 18, BLUE, True)
    c.text((73, 549), "Learn a shared representation", 21, MUTED)
    c.text((653, 290), "ATTACK TIME / IMU ONLY", 18, BLUE, True)
    waveform(c, 654, 399, 122, 59, t, BLUE)
    c.centered(716, 480, "IMU", 20, BLUE, True)
    c.arrow(788, 842, 428, BLUE, t)
    network(c, 867, 428, t, BLUE)
    c.arrow(992, 1037, 428, BLUE, t)
    c.box((1049, 383, 1206, 486), BG, BLUE, 12, 2)
    c.centered(1127, 399, "Activity", 23, INK, True)
    c.centered(1127, 439, "estimate", 21, BLUE)
    # A locked, closed camera, not a reconstruction of a hidden video.
    c.box((653, 537, 688, 560), BG, MUTED, 5, 2)
    c.line([(688, 542), (699, 537), (699, 559), (688, 554)], MUTED, 2)
    c.line([(648, 564), (703, 532)], CORAL, 3)
    c.text((717, 539), "Private video stays hidden", 21, MUTED)


def scene_conclusion(c, t):
    cards = [
        ("G1", "Binary", "events", TEAL),
        ("G2", "Accel OR", "gyro", CORAL),
        ("G3", "Accel +", "gyro", GOLD),
        ("G4", "Aligned", "embeddings", BLUE),
    ]
    for index, (game, first, second, color) in enumerate(cards):
        x = 48 + index * 303
        c.box((x, 275, x + 275, 509))
        c.text((x + 25, 293), game, 34, color, True)
        if index == 0:
            pulse(c, x + 26, 353, 220, 41, t, color)
        elif index in (1, 2):
            waveform(c, x + 26, 348, 220, 35, t, color)
            if index == 2:
                waveform(c, x + 26, 372, 220, 30, t + 2, color)
        else:
            for i in range(7):
                px = x + 40 + i * 31
                c.circle(px, 372 + math.sin(i * 1.9 + t) * 12, 6, color)
        c.text((x + 25, 421), first, 27, bold=True)
        c.text((x + 25, 460), second, 27, bold=True)
    c.box((48, 535, 1232, 586))
    c.text((72, 547), "Design choices matter:", 22, TEAL, True)
    c.text((377, 547), "output detail / time continuity / sensor access / auxiliary data", 21)


SCENES = [scene_output, scene_attack, scene_routine, scene_binding, scene_conclusion]


def scene_frame(scene, local):
    c = frame_base(scene, local)
    SCENES[scene](c, local)
    return c.finish()


def render_frame(time):
    scene = min(int(time / SCENE_SECONDS), 4)
    local = time - scene * SCENE_SECONDS
    image = scene_frame(scene, local)
    # Blend into the next scene, including the final scene back into the first.
    if local > SCENE_SECONDS - 0.3:
        alpha = (local - SCENE_SECONDS + 0.3) / 0.3
        image = Image.blend(image, scene_frame((scene + 1) % 5, 0), alpha)
    return image


def ffmpeg_path(override):
    if override:
        return override
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise SystemExit("Install ffmpeg or: python -m pip install imageio-ffmpeg") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs/assets")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--stills-only", action="store_true")
    args = parser.parse_args()
    if args.fps < 1 or args.gif_width < 2:
        parser.error("fps must be positive and gif-width must be at least 2")
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (960, 270 * 3), BG)
        for scene in range(5):
            frame = render_frame(scene * SCENE_SECONDS + 3)
            frame.save(args.preview_dir / f"scene-{scene + 1}.png")
            sheet.paste(frame.resize((480, 270)), ((scene % 2) * 480, (scene // 2) * 270))
        sheet.save(args.preview_dir / "contact-sheet.png")
    if args.stills_only:
        if not args.preview_dir:
            parser.error("--stills-only requires --preview-dir")
        return

    executable = ffmpeg_path(args.ffmpeg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mp4 = args.output_dir / "sensor-side-channel.mp4"
    gif = args.output_dir / "sensor-side-channel.gif"
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", "title=NNEdge-Privacy: sensor side-channel illustration",
        "-metadata", "comment=Synthetic conceptual animation; no participant data or experimental results.",
        str(mp4),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(DURATION * args.fps):
            process.stdin.write(render_frame(index / args.fps).tobytes())
            if index % (SCENE_SECONDS * args.fps) == 0:
                print(f"Rendering scene {index // (SCENE_SECONDS * args.fps) + 1}/5", flush=True)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdin.close()
    if process.wait():
        raise SystemExit("MP4 encoding failed")

    with tempfile.TemporaryDirectory(prefix="nneedge-palette-") as directory:
        palette = str(Path(directory) / "palette.png")
        scale = f"fps=10,scale={args.gif_width}:-2:flags=lanczos"
        subprocess.run([
            executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
            "-vf", scale + ",palettegen=max_colors=128:stats_mode=diff", palette,
        ], check=True)
        subprocess.run([
            executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4), "-i", palette,
            "-lavfi", f"{scale}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop", "0", str(gif),
        ], check=True)
    for asset in [mp4, gif]:
        print(f"Created {asset} ({asset.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
