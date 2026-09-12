"""Synthetic floor plan generator.

The proposal has the client supplying 2-3 sample floor plans at the start of
Phase 1. Until those land, this draws plans with a KNOWN ground truth, which is
worth more than a handful of real images anyway: it means the recognizer can be
scored -- detected wall length against actual, room count against actual --
instead of eyeballed.

Deliberately drawn the way a real estate export looks: solid black poche walls,
thin furniture and fixture lines, dimension text. The thin clutter is the point.
Anything that survives it by accident was never relying on the thickness signal.

    python -m app.synth --out ../../samples
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BLACK = 0
GRAY = 110
WHITE = 255


@dataclass
class GroundTruth:
    """What the drawing actually contains, for scoring a recognizer against."""

    pixels_per_meter: float
    wall_thickness_m: float
    room_count: int
    total_wall_length_m: float
    rooms: list[dict] = field(default_factory=list)


def _draw_wall(
    canvas: np.ndarray, a: tuple[int, int], b: tuple[int, int], thickness: int
) -> float:
    """Draw one wall as a filled rectangle and return its length in pixels."""
    cv2.line(canvas, a, b, BLACK, thickness=thickness, lineType=cv2.LINE_8)
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def _draw_clutter(canvas: np.ndarray, rng: np.random.Generator) -> None:
    """Thin lines and text: furniture, fixtures, dimensions, a title block.

    None of this is structural. A recognizer that reports it as walls is
    over-detecting, and that is exactly the failure this clutter is here to
    expose.
    """
    height, width = canvas.shape

    # Dimension lines with ticks, just inside the page margin.
    for y in (40, height - 40):
        cv2.line(canvas, (120, y), (width - 120, y), GRAY, 1)
        for x in (120, width - 120):
            cv2.line(canvas, (x, y - 6), (x, y + 6), GRAY, 1)
    cv2.putText(canvas, "12.00 m", (width // 2 - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

    # Furniture: rectangles and circles in thin stroke.
    for _ in range(14):
        x = int(rng.integers(180, width - 260))
        y = int(rng.integers(180, height - 260))
        w = int(rng.integers(40, 110))
        h = int(rng.integers(30, 90))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), GRAY, 1)
    for _ in range(5):
        center = (int(rng.integers(200, width - 200)), int(rng.integers(200, height - 200)))
        cv2.circle(canvas, center, int(rng.integers(14, 30)), GRAY, 1)

    # Room labels.
    for label, position in [
        ("LIVING", (230, 330)),
        ("KITCHEN", (700, 330)),
        ("BED 1", (230, 720)),
        ("BED 2", (700, 720)),
    ]:
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)

    # Title block, bottom right.
    cv2.rectangle(canvas, (width - 300, height - 120), (width - 40, height - 40), GRAY, 1)
    cv2.putText(
        canvas, "GROUND FLOOR", (width - 288, height - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1
    )
    cv2.putText(
        canvas, "SCALE 1:100", (width - 288, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1
    )


def generate_plan(
    *,
    width: int = 1200,
    height: int = 900,
    pixels_per_meter: float = 80.0,
    wall_thickness_m: float = 0.20,
    with_clutter: bool = True,
    seed: int = 7,
) -> tuple[np.ndarray, GroundTruth]:
    """Draw a four-room apartment with a corridor. Returns (image, ground truth).

    Layout, in metres from the top-left of the building:

        +---------------+---------------+
        |    LIVING     |    KITCHEN    |
        +------+ +------+------+ +------+   <- corridor gaps are doorways
        |           CORRIDOR            |
        +------+ +------+------+ +------+
        |    BED 1      |     BED 2     |
        +---------------+---------------+
    """
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), WHITE, dtype=np.uint8)

    thickness_px = max(3, int(round(wall_thickness_m * pixels_per_meter)))

    # Building envelope, inset from the page.
    left, top = 150, 120
    right, bottom = width - 200, height - 160

    mid_x = (left + right) // 2
    corridor_top = top + int((bottom - top) * 0.42)
    corridor_bottom = top + int((bottom - top) * 0.58)

    total_px = 0.0

    # Envelope, drawn thicker as a real exterior wall would be.
    exterior = int(thickness_px * 1.5)
    total_px += _draw_wall(canvas, (left, top), (right, top), exterior)
    total_px += _draw_wall(canvas, (right, top), (right, bottom), exterior)
    total_px += _draw_wall(canvas, (right, bottom), (left, bottom), exterior)
    total_px += _draw_wall(canvas, (left, bottom), (left, top), exterior)

    # Corridor walls, broken by door gaps. The gaps are the hard part: they are
    # why the wall mask has to be morphologically closed before room detection,
    # or the corridor leaks into every room and nothing comes out enclosed.
    door_gap = int(0.9 * pixels_per_meter)
    for y in (corridor_top, corridor_bottom):
        spans = [
            (left, mid_x - door_gap // 2 - 120),
            (mid_x - door_gap // 2 + door_gap - 120, mid_x),
            (mid_x, mid_x + 120 - door_gap // 2),
            (mid_x + 120 + door_gap // 2, right),
        ]
        for x0, x1 in spans:
            if x1 > x0:
                total_px += _draw_wall(canvas, (x0, y), (x1, y), thickness_px)

    # Vertical dividers above and below the corridor.
    total_px += _draw_wall(canvas, (mid_x, top), (mid_x, corridor_top), thickness_px)
    total_px += _draw_wall(canvas, (mid_x, corridor_bottom), (mid_x, bottom), thickness_px)

    if with_clutter:
        _draw_clutter(canvas, rng)

    def to_m(px: float) -> float:
        return round(px / pixels_per_meter, 3)

    living_w = to_m(mid_x - left)
    kitchen_w = to_m(right - mid_x)
    upper_h = to_m(corridor_top - top)
    corridor_h = to_m(corridor_bottom - corridor_top)
    lower_h = to_m(bottom - corridor_bottom)

    truth = GroundTruth(
        pixels_per_meter=pixels_per_meter,
        wall_thickness_m=wall_thickness_m,
        room_count=5,  # four rooms plus the corridor
        total_wall_length_m=to_m(total_px),
        rooms=[
            {"name": "LIVING", "approx_area_m2": living_w * upper_h},
            {"name": "KITCHEN", "approx_area_m2": kitchen_w * upper_h},
            {"name": "CORRIDOR", "approx_area_m2": (living_w + kitchen_w) * corridor_h},
            {"name": "BED 1", "approx_area_m2": living_w * lower_h},
            {"name": "BED 2", "approx_area_m2": kitchen_w * lower_h},
        ],
    )
    return canvas, truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic floor plans.")
    parser.add_argument("--out", type=Path, default=Path("samples"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    variants = {
        "clean": dict(with_clutter=False),
        "cluttered": dict(with_clutter=True),
        "thin_walls": dict(with_clutter=True, wall_thickness_m=0.12),
        "low_res": dict(with_clutter=True, pixels_per_meter=40.0, width=700, height=520),
    }

    import json

    manifest = {}
    for name, options in variants.items():
        image, truth = generate_plan(seed=args.seed, **options)
        path = args.out / f"synthetic_{name}.png"
        cv2.imwrite(str(path), image)
        manifest[path.name] = truth.__dict__
        print(f"wrote {path}  ({image.shape[1]}x{image.shape[0]})")

    (args.out / "ground_truth.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
