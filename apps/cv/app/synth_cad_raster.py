"""CAD-style raster floor plans, rendered from real DXF geometry.

These exist to test the RASTER recognizer against what a CAD drawing actually
looks like when it arrives as a picture -- a PDF screenshot, a scanned print, an
image pasted into an email. That is a very common way plans turn up, and it is
much harder than the solid-wall plans in `synth.py`.

The images are rendered by ezdxf's own drawing addon from the same DXF the
vector recognizer reads, so the linework is genuine CAD output rather than a
hand-drawn imitation of it, and the ground truth is shared with `synth_dxf`.

Two styles, because they stress the recognizer very differently:

**AutoCAD** draws walls as two thin parallel lines with nothing between them.
The wall is a *void* bounded by hairlines. This is the hard case: the raster
pipeline finds walls by looking for THICK dark structures, and here there is
nothing thick anywhere - the wall lines are the same weight as the furniture and
the dimension strings.

**Revit** renders cut elements with solid poché - walls filled black at the cut
plane. This is much closer to the solid-wall assumption, so it should behave
more like the existing synthetic plans.

Expect the AutoCAD variant to be genuinely difficult. That is the point: it is
the honest test, and a recognizer that only works on filled walls should be seen
to only work on filled walls.

    python -m app.synth_cad_raster --out ../../samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ezdxf import units as ezdxf_units

from app.synth_dxf import MM, SITE_ORIGIN, generate_dxf

# Rendered at a resolution typical of a plan exported or scanned at ~150 DPI on
# A3. Big enough that a 100mm partition is a few pixels, which is the realistic
# hard case rather than an artificially generous one.
RENDER_WIDTH_PX = 2200
RENDER_DPI = 150


def _add_wall_poche(doc) -> None:
    """Fill the wall cavities with solid hatch, the way Revit renders a cut.

    Revit shades anything the view's cut plane passes through. For a floor plan
    that is the walls, so they read as solid black masses rather than pairs of
    lines - which is exactly why a Revit export is easier to recognize from a
    raster than an AutoCAD one.

    The hatch is built from the wall faces already in the drawing: every pair of
    parallel lines on A-WALL bounds one rectangular cavity.
    """
    msp = doc.modelspace()
    doc.layers.add("A-WALL-PATT", color=7)

    faces = [e for e in msp.query("LINE") if e.dxf.layer == "A-WALL"]

    # Faces were drawn in pairs, in order, by synth_dxf._wall.
    for i in range(0, len(faces) - 1, 2):
        a, b = faces[i], faces[i + 1]
        hatch = msp.add_hatch(color=7, dxfattribs={"layer": "A-WALL-PATT"})
        hatch.paths.add_polyline_path(
            [
                (a.dxf.start.x, a.dxf.start.y),
                (a.dxf.end.x, a.dxf.end.y),
                (b.dxf.end.x, b.dxf.end.y),
                (b.dxf.start.x, b.dxf.start.y),
            ],
            is_closed=True,
        )


def _add_room_tags(doc, truth) -> None:
    """Revit room tags: name over area, the way a room schedule prints them.

    Areas are COMPUTED from the same geometry constants `synth_dxf` draws with,
    not written in by hand. A fixture that states a wrong number is worse than
    one that states none, because someone will eventually trust it.
    """
    msp = doc.modelspace()
    ox, oy = SITE_ORIGIN

    width, height = 12_000 * MM, 9_000 * MM
    exterior, interior = 300 * MM, 100 * MM
    mid_x = ox + width / 2
    corridor_bottom = oy + height * 0.40
    corridor_top = oy + height * 0.58

    # Clear internal faces: half a wall thickness in from each centreline.
    left, right = ox + exterior / 2, ox + width - exterior / 2
    bottom, top = oy + exterior / 2, oy + height - exterior / 2
    mid_l, mid_r = mid_x - interior / 2, mid_x + interior / 2
    corr_lo, corr_hi = corridor_bottom + interior / 2, corridor_top - interior / 2

    # The rooms above and below start on the FAR side of the corridor walls,
    # not at the corridor's own inner faces.
    upper_h = top - (corridor_top + interior / 2)
    lower_h = (corridor_bottom - interior / 2) - bottom
    corridor_h = corr_hi - corr_lo

    def m2(w: float, h: float) -> float:
        return (w / 1000) * (h / 1000)

    tags = [
        ("LIVING", ox + 2_800 * MM, oy + 6_600 * MM, m2(mid_l - left, upper_h)),
        ("KITCHEN", ox + 8_400 * MM, oy + 6_600 * MM, m2(right - mid_r, upper_h)),
        ("CORRIDOR", ox + 5_200 * MM, oy + 3_900 * MM, m2(right - left, corridor_h)),
        ("BED 1", ox + 2_800 * MM, oy + 1_400 * MM, m2(mid_l - left, lower_h)),
        ("BED 2", ox + 8_400 * MM, oy + 1_400 * MM, m2(right - mid_r, lower_h)),
    ]
    for _name, x, y, area in tags:
        msp.add_text(
            f"{area:.1f} m2", height=220 * MM, dxfattribs={"layer": "A-AREA-IDEN"}
        ).set_placement((x, y - 420 * MM))


#: Plotted lineweights, in 1/100 mm - the values a real title block would use.
#: Walls plot heavy, annotation plots fine. This is not decoration: line weight
#: is the ONLY signal separating structure from furniture in a monochrome plot,
#: and it is what the raster recognizer's thickness test keys off.
LINEWEIGHTS = {
    "A-WALL": 50,       # 0.50mm
    "A-WALL-PATT": 50,
    "A-DOOR": 25,
    "A-GLAZ": 25,
    "A-FURN": 13,       # 0.13mm - the finest pen
    "A-ANNO-DIMS": 13,
    "A-ANNO-TTLB": 18,
    "A-AREA-IDEN": 18,
}


def _plot_ready(doc) -> None:
    """Make the drawing look like a PLOT, not a screenshot of model space.

    Two changes, both of which make the test harder and more honest:

    1. **Monochrome.** A plotted or scanned plan is black on white. Leaving
       AutoCAD's on-screen layer colours in would let a recognizer cheat by
       filtering on hue, which no real scan would ever permit.
    2. **Real lineweights.** Walls at 0.50mm, furniture at 0.13mm. In a
       monochrome drawing this is the only thing that distinguishes structure
       from clutter - which is exactly the signal the raster pipeline claims to
       use, so it should be made to earn it.
    """
    for layer in doc.layers:
        layer.color = 7  # black on a light background
        weight = LINEWEIGHTS.get(layer.dxf.name)
        if weight is not None:
            layer.dxf.lineweight = weight

    # Without this, lineweights are stored but not honoured when rendering.
    doc.header["$LWDISPLAY"] = 1


def render(doc, path: Path, *, background: str = "#FFFFFF") -> None:
    """Render a DXF modelspace to PNG through ezdxf's matplotlib backend.

    `qsave` is the supported path and handles the colour subtlety that matters
    here: DXF has no explicit foreground colour, and ACI colour 7 means
    "whatever contrasts with the paper". Passing a white background is what
    makes the linework resolve to black instead of white-on-white.
    """
    # Imported lazily: matplotlib is a dev-only dependency and the service must
    # never pull it in.
    from ezdxf.addons.drawing import matplotlib as mpl_backend
    from ezdxf.addons.drawing.config import Configuration, LineweightPolicy

    _plot_ready(doc)

    inches = RENDER_WIDTH_PX / RENDER_DPI
    mpl_backend.qsave(
        doc.modelspace(),
        path,
        bg=background,
        dpi=RENDER_DPI,
        size_inches=(inches, inches),
        # Honour the plotted lineweights set above rather than drawing every
        # entity at a uniform hairline, which would erase the only cue
        # separating walls from furniture in a monochrome plot.
        config=Configuration(lineweight_policy=LineweightPolicy.ABSOLUTE),
    )


def build_autocad_style():
    """Hollow double-line walls: the authentic, and difficult, AutoCAD look."""
    return generate_dxf()


def build_revit_style():
    """Solid poché walls plus room area tags: the Revit plan-view look."""
    doc, truth = generate_dxf()
    doc.units = ezdxf_units.MM
    _add_wall_poche(doc)
    _add_room_tags(doc, truth)
    return doc, truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render CAD-style raster floor plans from real DXF geometry."
    )
    parser.add_argument("--out", type=Path, default=Path("samples"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}

    for name, build in [
        ("autocad_style", build_autocad_style),
        ("revit_style", build_revit_style),
    ]:
        doc, truth = build()
        path = args.out / f"{name}.png"
        render(doc, path)
        manifest[path.name] = {
            "style": name,
            "rendered_from": "dxf",
            "pixels_per_meter": None,  # a raster has no stated scale; that is the point
            "wall_thickness_m": truth.wall_thickness_m,
            "room_count": truth.room_count,
            "door_count": truth.door_count,
            "window_count": truth.window_count,
            "total_wall_length_m": truth.total_wall_length_m,
            "building_size_m": list(truth.building_size_m),
            "rooms": truth.rooms,
        }
        print(f"wrote {path}")

    truth_path = args.out / "ground_truth_cad_raster.json"
    truth_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
