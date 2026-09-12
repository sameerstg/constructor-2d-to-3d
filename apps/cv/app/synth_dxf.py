"""Synthetic DXF floor plans, drawn the way real CAD drawings are.

Written to the same conventions an AutoCAD or Revit export follows, because
those conventions are exactly what the DXF recognizer keys off and testing
against a tidier file than reality would prove nothing:

- **Walls as double lines.** Two parallel faces a thickness apart, never a
  centreline. Recovering the centreline is the recognizer's main geometric job.
- **AIA/NCS layer names.** A-WALL, A-DOOR, A-GLAZ, A-AREA-IDEN, plus furniture
  and dimensions on their own layers, so layer filtering has something to do.
- **Real units.** $INSUNITS set to millimetres, the architectural default, and
  coordinates on a site grid far from the origin.
- **Doors and windows as block INSERTs**, which is how CAD represents them and
  why they are detectable here but not in a raster.
- **Clutter.** Furniture, dimension lines and a title block, none of which is
  structure.

    python -m app.synth_dxf --out ../../samples
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
from ezdxf import units as ezdxf_units

# Millimetres, like a real architectural drawing. Also a good test that unit
# conversion works: get it wrong and the building is 1000x too big.
MM = 1.0

# Placed well away from the origin, like a real site plan. Exercises the
# origin-normalisation that stops float precision collapsing in the mesh.
SITE_ORIGIN = (482_350.0, 5_741_200.0)


@dataclass
class DxfGroundTruth:
    units: str
    wall_thickness_m: dict[str, float]
    room_count: int
    door_count: int
    window_count: int
    total_wall_length_m: float
    building_size_m: tuple[float, float]
    rooms: list[str] = field(default_factory=list)


def _add_layers(doc) -> None:
    """AIA/NCS layer names, with the colours a real drawing would use."""
    for name, color in [
        ("A-WALL", 7),
        ("A-DOOR", 3),
        ("A-GLAZ", 4),
        ("A-AREA-IDEN", 2),
        ("A-FURN", 8),
        ("A-ANNO-DIMS", 1),
        ("A-ANNO-TTLB", 8),
    ]:
        doc.layers.add(name, color=color)


def _door_block(doc) -> None:
    """A 900mm door: leaf plus swing arc, drawn about its hinge."""
    block = doc.blocks.new(name="DOOR-900")
    block.add_line((0, 0), (900, 0))
    block.add_arc(center=(0, 0), radius=900, start_angle=0, end_angle=90)
    block.add_line((0, 0), (0, 900))


def _window_block(doc) -> None:
    """A 1200mm window: frame lines across the reveal."""
    block = doc.blocks.new(name="WINDOW-1200")
    for offset in (0, 50, 100):
        block.add_line((0, offset), (1200, offset))


def _wall(msp, a: tuple[float, float], b: tuple[float, float], thickness: float) -> float:
    """Draw a wall as its two faces and return its centreline length.

    This is the crux of the test: nothing in the file records the centreline.
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0

    # Unit normal, offset each way by half the thickness.
    nx, ny = -dy / length, dx / length
    half = thickness / 2

    for sign in (1, -1):
        ox, oy = nx * half * sign, ny * half * sign
        msp.add_line((ax + ox, ay + oy), (bx + ox, by + oy), dxfattribs={"layer": "A-WALL"})

    return length


def generate_dxf(
    *,
    exterior_thickness: float = 300 * MM,
    interior_thickness: float = 100 * MM,
    with_clutter: bool = True,
    unitless: bool = False,
):
    """A four-room apartment with a central corridor, drawn as real CAD.

        +------------------+------------------+
        |      LIVING      |     KITCHEN      |
        +----[door]--------+--------[door]----+
        |             CORRIDOR                |
        +----[door]--------+--------[door]----+
        |      BED 1       |      BED 2       |
        +------------------+------------------+
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf_units.MM
    if unitless:
        # Some exports genuinely omit units. The recognizer must warn, not guess
        # silently, so this variant exists to prove it does.
        doc.header["$INSUNITS"] = 0

    _add_layers(doc)
    _door_block(doc)
    _window_block(doc)
    msp = doc.modelspace()

    ox, oy = SITE_ORIGIN
    width, height = 12_000 * MM, 9_000 * MM
    mid_x = ox + width / 2
    corridor_bottom = oy + height * 0.40
    corridor_top = oy + height * 0.58

    total = 0.0

    # Exterior envelope.
    total += _wall(msp, (ox, oy), (ox + width, oy), exterior_thickness)
    total += _wall(msp, (ox + width, oy), (ox + width, oy + height), exterior_thickness)
    total += _wall(msp, (ox + width, oy + height), (ox, oy + height), exterior_thickness)
    total += _wall(msp, (ox, oy + height), (ox, oy), exterior_thickness)

    # Corridor walls, each broken by a doorway gap.
    door_width = 900 * MM
    for y in (corridor_bottom, corridor_top):
        for x_start, x_end in [
            (ox, ox + 3_500 * MM),
            (ox + 3_500 * MM + door_width, mid_x),
            (mid_x, mid_x + 2_600 * MM),
            (mid_x + 2_600 * MM + door_width, ox + width),
        ]:
            if x_end > x_start:
                total += _wall(msp, (x_start, y), (x_end, y), interior_thickness)

    # Vertical dividers above and below the corridor.
    total += _wall(msp, (mid_x, oy), (mid_x, corridor_bottom), interior_thickness)
    total += _wall(msp, (mid_x, corridor_top), (mid_x, oy + height), interior_thickness)

    # Doors, in the gaps left above.
    door_positions = [
        (ox + 3_500 * MM, corridor_bottom, 0),
        (mid_x + 2_600 * MM, corridor_bottom, 0),
        (ox + 3_500 * MM, corridor_top, 0),
        (mid_x + 2_600 * MM, corridor_top, 0),
    ]
    for x, y, rotation in door_positions:
        msp.add_blockref(
            "DOOR-900", (x, y), dxfattribs={"layer": "A-DOOR", "rotation": rotation}
        )

    # Windows in the exterior walls.
    window_positions = [
        (ox + 2_000 * MM, oy, 0),
        (ox + 8_000 * MM, oy, 0),
        (ox + 2_000 * MM, oy + height, 0),
        (ox + 8_000 * MM, oy + height, 0),
        (ox, oy + 3_000 * MM, 90),
        (ox + width, oy + 5_000 * MM, 90),
    ]
    for x, y, rotation in window_positions:
        msp.add_blockref(
            "WINDOW-1200", (x, y), dxfattribs={"layer": "A-GLAZ", "rotation": rotation}
        )

    # Room name text, placed inside each room so the recognizer can attribute it.
    room_labels = [
        ("LIVING", ox + 2_800 * MM, oy + 7_000 * MM),
        ("KITCHEN", mid_x + 2_400 * MM, oy + 7_000 * MM),
        ("CORRIDOR", ox + 5_200 * MM, oy + 4_400 * MM),
        ("BED 1", ox + 2_800 * MM, oy + 1_800 * MM),
        ("BED 2", mid_x + 2_400 * MM, oy + 1_800 * MM),
    ]
    for text, x, y in room_labels:
        msp.add_text(
            text,
            height=300 * MM,
            dxfattribs={"layer": "A-AREA-IDEN"},
        ).set_placement((x, y))

    if with_clutter:
        _add_clutter(msp, ox, oy, width, height)

    truth = DxfGroundTruth(
        units="unitless" if unitless else "mm",
        wall_thickness_m={
            "exterior": exterior_thickness / 1000,
            "interior": interior_thickness / 1000,
        },
        room_count=5,
        door_count=len(door_positions),
        window_count=len(window_positions),
        total_wall_length_m=round(total / 1000, 3),
        building_size_m=(width / 1000, height / 1000),
        rooms=[label for label, _, _ in room_labels],
    )
    return doc, truth


def _add_clutter(msp, ox: float, oy: float, width: float, height: float) -> None:
    """Furniture, dimensions and a title block: everything that is not a wall."""
    # Furniture, as closed polylines on the furniture layer.
    furniture = [
        (ox + 1_000, oy + 6_000, 2_000, 900),
        (ox + 7_500, oy + 6_500, 1_800, 700),
        (ox + 1_200, oy + 1_000, 1_400, 2_000),
        (ox + 7_000, oy + 1_000, 1_400, 2_000),
    ]
    for x, y, w, h in furniture:
        msp.add_lwpolyline(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            close=True,
            dxfattribs={"layer": "A-FURN"},
        )

    # Dimension lines, drawn as plain geometry the way many exports do.
    msp.add_line(
        (ox, oy - 1_200), (ox + width, oy - 1_200), dxfattribs={"layer": "A-ANNO-DIMS"}
    )
    msp.add_text(
        "12000", height=250, dxfattribs={"layer": "A-ANNO-DIMS"}
    ).set_placement((ox + width / 2, oy - 1_600))

    # Title block.
    msp.add_lwpolyline(
        [
            (ox + width + 1_000, oy),
            (ox + width + 5_000, oy),
            (ox + width + 5_000, oy + 2_000),
            (ox + width + 1_000, oy + 2_000),
        ],
        close=True,
        dxfattribs={"layer": "A-ANNO-TTLB"},
    )
    msp.add_text(
        "GROUND FLOOR PLAN", height=300, dxfattribs={"layer": "A-ANNO-TTLB"}
    ).set_placement((ox + width + 1_200, oy + 1_200))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DXF floor plans.")
    parser.add_argument("--out", type=Path, default=Path("samples"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    variants = {
        "cad": {},
        "cad_clean": {"with_clutter": False},
        "cad_unitless": {"unitless": True},
    }

    manifest = {}
    for name, options in variants.items():
        doc, truth = generate_dxf(**options)
        path = args.out / f"synthetic_{name}.dxf"
        doc.saveas(path)
        manifest[path.name] = {
            **truth.__dict__,
            "building_size_m": list(truth.building_size_m),
        }
        print(f"wrote {path}")

    truth_path = args.out / "ground_truth_dxf.json"
    truth_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
