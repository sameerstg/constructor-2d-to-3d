"""IR -> 3D geometry -> GLB.

This is the easy half of "2D floor plan to 3D model", and saying so is the
point: once the plan is a clean vector IR, building the mesh is deterministic
parametric geometry, not machine learning. All the difficulty lives upstream in
recognition. Generative image-to-mesh models are the wrong tool here -- they
produce unstructured blobs with no metric scale and no addressable parts, so
you could never swap the material on one named wall, which is exactly what
Phase 2 of the proposal requires.

Coordinate frames
-----------------
IR plan space is metres, X right, Y DOWN. glTF is Y-up, right-handed. The
mapping is applied in exactly one place, `_plan_to_world`:

    plan (x, y)  ->  world (x, elevation, y)

Naming
------
Every mesh is named for the IR entity it came from (`wall_w_1a2b3c`,
`floor_r_4d5e6f`). Those names are the addressing scheme the web configurator
and the VR client both use to retarget materials at runtime, so a material
change is a lookup rather than a re-export.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import orient

from app.ir import FloorPlanIR, Level, Opening, Room, Wall, WallType

# Slab thickness for room floors. Thin enough to be invisible underfoot, thick
# enough to avoid z-fighting with the wall bottoms in VR.
FLOOR_SLAB_THICKNESS = 0.05

# Boolean cutters are oversized along the wall normal so the subtraction passes
# cleanly through both faces. Coplanar faces are the classic way to make a mesh
# boolean produce degenerate output.
CUTTER_OVERSHOOT = 1.5

# Muted neutral defaults. Real materials arrive in Phase 2; these exist so the
# walkthrough is readable rather than a uniform grey mass.
_COLORS: dict[str, list[int]] = {
    "wall_exterior": [196, 190, 180, 255],
    "wall_interior": [224, 221, 214, 255],
    "floor": [158, 141, 122, 255],
}


class ExtrusionError(RuntimeError):
    """Raised when an IR cannot be turned into geometry."""


# --------------------------------------------------------------------------
# Frame conversion
# --------------------------------------------------------------------------


def _plan_to_world(x: float, y: float, elevation: float) -> tuple[float, float, float]:
    """The single plan-space to world-space conversion in the codebase."""
    return (x, elevation, y)


def _wall_frame(wall: Wall) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """Return (length, yaw, unit direction, midpoint) for a wall in plan space.

    `yaw` is the rotation about world +Y that takes the box's local +X onto the
    wall direction. Rotating about +Y by phi sends X to (cos phi, 0, -sin phi),
    and we need it to land on (dx, 0, dy), so sin phi = -dy/L and
    phi = atan2(-dy, dx).
    """
    (x0, y0), (x1, y1) = wall.start, wall.end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        raise ExtrusionError(f"wall {wall.id} has zero length")

    yaw = math.atan2(-dy, dx)
    unit = (dx / length, dy / length)
    midpoint = ((x0 + x1) / 2, (y0 + y1) / 2)
    return length, yaw, unit, midpoint


# --------------------------------------------------------------------------
# Walls
# --------------------------------------------------------------------------


def _wall_mesh(wall: Wall, level: Level) -> trimesh.Trimesh:
    height = wall.height or level.wall_height
    length, yaw, _, (mid_x, mid_y) = _wall_frame(wall)

    # Local axes: X along the wall, Y up, Z across its thickness.
    box = trimesh.creation.box(extents=[length, height, wall.thickness])

    transform = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
    transform[:3, 3] = _plan_to_world(mid_x, mid_y, level.elevation + height / 2)
    box.apply_transform(transform)
    return box


def _opening_cutter(opening: Opening, wall: Wall, level: Level) -> trimesh.Trimesh:
    """A box to subtract from a wall, positioned along that wall."""
    length, yaw, (ux, uy), (mid_x, mid_y) = _wall_frame(wall)

    # `offset` is measured from the wall's start; the box is centred, so shift
    # from the wall midpoint by the difference.
    along = opening.offset - length / 2
    center_x = mid_x + ux * along
    center_y = mid_y + uy * along
    center_height = level.elevation + opening.sill + opening.height / 2

    cutter = trimesh.creation.box(
        extents=[
            opening.width,
            opening.height,
            wall.thickness * CUTTER_OVERSHOOT,
        ]
    )
    transform = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
    transform[:3, 3] = _plan_to_world(center_x, center_y, center_height)
    cutter.apply_transform(transform)
    return cutter


def _wall_with_openings(
    wall: Wall, openings: list[Opening], level: Level
) -> trimesh.Trimesh:
    mesh = _wall_mesh(wall, level)
    if not openings:
        return mesh

    cutters = [_opening_cutter(o, wall, level) for o in openings]
    try:
        result = trimesh.boolean.difference([mesh, *cutters])
    except Exception as exc:  # noqa: BLE001 - any engine failure is recoverable
        # A failed boolean must not lose the wall. A solid wall where a doorway
        # should be is an obvious, reportable defect; a silently missing wall is
        # a hole in the building that nobody notices until VR review.
        raise ExtrusionError(
            f"could not cut {len(cutters)} opening(s) into wall {wall.id}: {exc}"
        ) from exc

    if result is None or result.is_empty:
        raise ExtrusionError(f"cutting openings from wall {wall.id} produced empty geometry")
    return result


# --------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------


def _floor_mesh(room: Room) -> trimesh.Trimesh | None:
    """Extrude a room polygon into a floor slab.

    The polygon is mirrored in Y before extrusion so that the subsequent -90
    degree rotation about X lands it the right way round in world space, rather
    than mirrored. Doing it as a genuine rotation keeps the face winding -- and
    therefore the normals -- correct, which a coordinate swap would not.
    """
    mirrored = [(x, -y) for x, y in room.polygon]

    polygon = Polygon(mirrored)
    if not polygon.is_valid:
        # Self-intersecting contours are common from approxPolyDP on noisy
        # masks. buffer(0) resolves most of them.
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda p: p.area)

    if polygon.area < 1e-6:
        return None

    polygon = orient(polygon, sign=1.0)  # CCW, so normals point up after rotation

    slab = trimesh.creation.extrude_polygon(polygon, height=FLOOR_SLAB_THICKNESS)
    slab.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    return slab


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _tint(mesh: trimesh.Trimesh, key: str) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=np.tile(_COLORS[key], (len(mesh.faces), 1))
    )
    return mesh


def build_scene(ir: FloorPlanIR, *, level_index: int = 0) -> trimesh.Scene:
    """Turn one level of an IR into a named, addressable trimesh Scene."""
    try:
        level = ir.levels[level_index]
    except IndexError:
        raise ExtrusionError(
            f"level {level_index} does not exist; plan has {len(ir.levels)}"
        ) from None

    scene = trimesh.Scene()

    openings_by_wall: dict[str, list[Opening]] = {}
    for opening in level.openings:
        openings_by_wall.setdefault(opening.wall_id, []).append(opening)

    for wall in level.walls:
        mesh = _wall_with_openings(wall, openings_by_wall.get(wall.id, []), level)
        key = "wall_exterior" if wall.type is WallType.EXTERIOR else "wall_interior"
        name = f"wall_{wall.id}"
        scene.add_geometry(_tint(mesh, key), node_name=name, geom_name=name)

    for room in level.rooms:
        mesh = _floor_mesh(room)
        if mesh is None:
            continue
        mesh.apply_translation([0, level.elevation - FLOOR_SLAB_THICKNESS, 0])
        scene.add_geometry(
            _tint(mesh, "floor"), node_name=f"floor_{room.id}", geom_name=f"floor_{room.id}"
        )

    if not scene.geometry:
        raise ExtrusionError("nothing to extrude: the plan has no walls or rooms")

    return scene


def export_glb(ir: FloorPlanIR, *, level_index: int = 0) -> bytes:
    """IR in, GLB bytes out. This is the artefact the VR client consumes."""
    scene = build_scene(ir, level_index=level_index)
    return trimesh.exchange.gltf.export_glb(scene)
