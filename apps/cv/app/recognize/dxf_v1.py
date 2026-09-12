"""DXF floor plan recognizer.

A vector CAD drawing is a fundamentally better input than a picture of one, and
this recognizer exists to stop throwing that advantage away. Where the raster
pipeline has to infer everything from pixel thickness, a DXF states it:

| |raster| DXF |
| --- | --- | --- |
| scale | guessed, 15-40% error | **exact**, from the file's own units |
| wall thickness | one global estimate | **measured per wall**, face to face |
| which lines are walls | inferred from stroke weight | **layer name** |
| doors and windows | not detected | **block references**, with position |
| room names | not detected | **text**, placed inside the room |

Both AutoCAD and Revit export DXF, so this covers the formats a builder actually
works in. Native DWG and RVT are proprietary and have no permissively licensed
reader; see docs/CAD-INPUT.md for the one-step export.

What is still inferred, and why
-------------------------------
Rooms. DXF has no concept of an enclosed room unless the drawing happens to
carry one (AREA or SPACE polylines, which many do not). Rather than write a
second room finder, the recovered wall centrelines are rendered into a raster
mask and handed to the existing raster room detection. Reusing it means door-gap
bridging and the touches-the-border test only exist once, and a fix to either
improves both recognizers.
"""

from __future__ import annotations

import io
import logging
import math
import re
import time

import cv2
import ezdxf
import numpy as np
from ezdxf import recover
from ezdxf import units as ezdxf_units
from ezdxf.document import Drawing

from app.ir import (
    FloorPlanIR,
    Level,
    Opening,
    OpeningType,
    Provenance,
    Room,
    RoomType,
    Scale,
    ScaleSource,
    Severity,
    SourceImage,
    Wall,
    WallType,
    Warning,
)
from app.recognize.base import RecognizerError
from app.recognize.opencv_v1 import detect_rooms
from app.recognize.wallpairs import Line, pair_walls
from app.sources import SourceDocument, SourceKind

logger = logging.getLogger(__name__)

NAME = "dxf"
VERSION = "1.0.0"

# Layer naming is conventional, not enforced. These cover the AIA/NCS layer
# standard (A-WALL), ISO 13567, Revit's DXF export, and the informal names that
# turn up in practice, in several languages.
WALL_LAYER_PATTERN = re.compile(
    r"(^|[-_ ])(a[-_]?wall|wall|walls|mur|murs|wand|waende|wände|parede|muro|"
    r"pared|vagg|vägg|stena|duvar|стена)([-_ ]|$)",
    re.IGNORECASE,
)
DOOR_PATTERN = re.compile(r"door|porte|tuer|tür|puerta|porta|dorr|dörr|deur", re.IGNORECASE)
WINDOW_PATTERN = re.compile(
    r"window|fenetre|fenêtre|fenster|ventana|janela|fonster|fönster|raam", re.IGNORECASE
)
ROOM_TEXT_PATTERN = re.compile(
    r"(^|[-_ ])(a[-_]?area|room|rooms|space|spaces|anno|text|ident|piece|raum)([-_ ]|$)",
    re.IGNORECASE,
)

# Room-type keywords, matched against text found inside a room polygon.
ROOM_KEYWORDS: list[tuple[re.Pattern[str], RoomType]] = [
    (re.compile(r"living|lounge|salon|wohn|sala", re.I), RoomType.LIVING),
    (re.compile(r"kitchen|cuisine|küche|kuche|cocina|cozinha", re.I), RoomType.KITCHEN),
    (re.compile(r"dining|salle a manger|esszimmer|comedor", re.I), RoomType.DINING),
    (re.compile(r"bed\s?\d*|bedroom|chambre|schlaf|dormitorio|quarto", re.I), RoomType.BEDROOM),
    (
        re.compile(r"bath|wc|toilet|salle de bain|bad|baño|banho|shower|ensuite", re.I),
        RoomType.BATHROOM,
    ),
    (re.compile(r"hall|entry|entrance|foyer|vestibule|diele", re.I), RoomType.HALL),
    (re.compile(r"corridor|hallway|passage|couloir|flur", re.I), RoomType.CORRIDOR),
    (re.compile(r"closet|storage|store|utility|pantry|cupboard|abstell", re.I), RoomType.STORAGE),
    (re.compile(r"garage|carport", re.I), RoomType.GARAGE),
    (re.compile(r"balcon|terrace|terras|patio|loggia", re.I), RoomType.BALCONY),
    (re.compile(r"stair|escalier|treppe|escalera", re.I), RoomType.STAIRS),
]

# Resolution used only for the room-detection raster. High enough that a 0.1m
# partition is several pixels wide; low enough that a large site plan does not
# allocate a gigabyte.
ROOM_RASTER_PX_PER_M = 40.0
MAX_ROOM_RASTER_EDGE = 3000

DEFAULT_DOOR_WIDTH = 0.9
DEFAULT_WINDOW_WIDTH = 1.2
DEFAULT_WINDOW_HEIGHT = 1.2
DEFAULT_WINDOW_SILL = 0.9

# How far an opening block may sit from a wall centreline and still be attached
# to it. Generous, because a door block's insertion point is often at the jamb
# rather than the centre of the wall.
MAX_OPENING_ATTACH_DISTANCE = 0.75


class DxfRecognizer:
    """Implements the `Recognizer` protocol for DXF documents."""

    name = NAME
    version = VERSION
    accepts = frozenset({SourceKind.DXF})

    def recognize(
        self,
        document: SourceDocument,
        *,
        pixels_per_meter: float | None = None,
    ) -> FloorPlanIR:
        started = time.perf_counter()
        warnings: list[Warning] = []

        doc = _read_dxf(document.data)
        to_meters = _unit_scale(doc, warnings)

        modelspace = doc.modelspace()
        layers = {layer.dxf.name for layer in doc.layers}

        # --- walls ---------------------------------------------------------
        wall_layers = {name for name in layers if WALL_LAYER_PATTERN.search(name)}
        lines, used_layer_filter = _collect_wall_lines(modelspace, wall_layers, to_meters)

        if not lines:
            raise RecognizerError(
                "No line geometry found in the DXF modelspace. If the drawing lives "
                "in a paper space layout, export the model view instead."
            )

        if not used_layer_filter:
            warnings.append(
                Warning(
                    code="no_wall_layer",
                    severity=Severity.WARN,
                    message=(
                        "No layer looked like a wall layer, so every line in the "
                        "drawing was considered. Furniture and dimensions may appear "
                        "as walls. Naming the wall layer 'A-WALL' fixes this."
                    ),
                )
            )

        paired, unpaired = pair_walls(lines)
        if not paired:
            raise RecognizerError(
                "Found line geometry but could not resolve any walls from it."
            )

        walls = [
            Wall(
                start=[round(w.start[0], 4), round(w.start[1], 4)],
                end=[round(w.end[0], 4), round(w.end[1], 4)],
                thickness=round(w.thickness, 4),
                type=WallType.INTERIOR,
                # A measured pair is trustworthy geometry; a lone line is a
                # guess about both which side is which and how thick it is.
                confidence=0.95 if w.measured else 0.5,
            )
            for w in paired
        ]
        transform = _normalise_origin(walls)

        if unpaired:
            warnings.append(
                Warning(
                    code="unpaired_wall_lines",
                    severity=Severity.INFO,
                    message=(
                        f"{unpaired} line(s) had no parallel partner and were treated "
                        f"as single-thickness walls. Check these in review."
                    ),
                )
            )

        # --- scale ---------------------------------------------------------
        # DXF states its units, so the scale is exact and the whole
        # guess-and-calibrate problem simply does not arise. The IR still needs
        # a pixels-per-metre for the 2D overlay, so we publish the raster
        # resolution used below and mark it as authoritative.
        scale = Scale(
            pixels_per_meter=ROOM_RASTER_PX_PER_M,
            source=ScaleSource.MANUAL,
            confidence=1.0,
            note="Exact: taken from the DXF's own drawing units.",
        )

        # --- rooms ---------------------------------------------------------
        rooms = _detect_rooms_from_walls(walls)
        _name_rooms(modelspace, rooms, to_meters, transform)

        if not rooms:
            warnings.append(
                Warning(
                    code="no_rooms_detected",
                    severity=Severity.WARN,
                    message=(
                        "No enclosed rooms were found. The model will have walls but "
                        "no floors. This usually means wall lines do not quite meet "
                        "at corners in the source drawing."
                    ),
                )
            )

        _promote_exterior_walls(walls, rooms)

        # --- openings ------------------------------------------------------
        openings, unattached = _collect_openings(modelspace, walls, to_meters, transform)
        if unattached:
            warnings.append(
                Warning(
                    code="unattached_openings",
                    severity=Severity.WARN,
                    message=(
                        f"{unattached} door/window block(s) were too far from any wall "
                        f"to attach and have been dropped. Add them in review."
                    ),
                )
            )
        if not openings:
            warnings.append(
                Warning(
                    code="no_openings_detected",
                    severity=Severity.INFO,
                    message=(
                        "No door or window blocks were found. If the drawing draws "
                        "them as plain geometry rather than named blocks, they must "
                        "be added in the review editor."
                    ),
                )
            )

        level = Level(name="Ground Floor", walls=walls, openings=openings, rooms=rooms)

        return FloorPlanIR(
            name=document.stem,
            source=SourceImage(
                filename=document.filename,
                # A vector drawing has no pixel dimensions. Publish the extent of
                # the room raster so the 2D overlay has a coordinate space that
                # matches its pixels_per_meter.
                width=max(1, int(_extent(walls)[0] * ROOM_RASTER_PX_PER_M)),
                height=max(1, int(_extent(walls)[1] * ROOM_RASTER_PX_PER_M)),
            ),
            scale=scale,
            levels=[level],
            provenance=Provenance(
                recognizer=self.name,
                recognizer_version=self.version,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
            warnings=warnings,
        )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _read_dxf(data: bytes) -> Drawing:
    """Parse DXF bytes, tolerating the malformed files CAD exporters produce."""
    if data.startswith(b"AutoCAD Binary DXF"):
        raise RecognizerError(
            "This is a binary DXF, which is not supported. Re-export as ASCII "
            "DXF (in AutoCAD's SAVEAS dialog, choose 'AutoCAD DXF' rather than "
            "'Binary DXF')."
        )

    try:
        # `recover.read` repairs the structural damage real exports routinely
        # contain; plain `ezdxf.read` rejects files every CAD program opens
        # fine. It takes a BINARY stream and detects the encoding itself, which
        # matters because DXF predates Unicode and older files are cp1252.
        doc, auditor = recover.read(io.BytesIO(data))
    except ezdxf.DXFStructureError as exc:
        raise RecognizerError(f"This file is not valid DXF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - ezdxf raises a wide range here
        raise RecognizerError(f"Could not read this DXF: {exc}") from exc

    if auditor.has_errors:
        logger.info(
            "recovered DXF with %d structural error(s); proceeding",
            len(auditor.errors),
        )

    return doc


def _unit_scale(doc: Drawing, warnings: list[Warning]) -> float:
    """Metres per drawing unit, from the DXF header.

    This single value replaces the entire scale-guessing problem that the raster
    pipeline cannot solve. $INSUNITS is set by every modern CAD exporter.
    """
    raw = doc.header.get("$INSUNITS", 0)

    if raw == 0:
        # Unitless. Almost always means millimetres in architectural practice,
        # which is what AutoCAD templates default to, but it is a guess and must
        # be flagged as one.
        warnings.append(
            Warning(
                code="dxf_units_unspecified",
                severity=Severity.WARN,
                message=(
                    "The DXF does not state its units ($INSUNITS is 0). Millimetres "
                    "have been assumed. If the model comes out 1000x too large or "
                    "small, this is why - set the units in the source drawing."
                ),
            )
        )
        return 0.001

    try:
        factor = ezdxf_units.conversion_factor(raw, ezdxf_units.M)
    except Exception:  # noqa: BLE001 - unknown or exotic unit code
        warnings.append(
            Warning(
                code="dxf_units_unknown",
                severity=Severity.WARN,
                message=f"Unrecognised DXF unit code {raw}; assumed millimetres.",
            )
        )
        return 0.001

    return float(factor)


def _collect_wall_lines(
    modelspace, wall_layers: set[str], to_meters: float
) -> tuple[list[Line], bool]:
    """Gather candidate wall lines, in metres.

    Returns the lines and whether a wall-layer filter was actually applied, so
    the caller can warn when it was not.
    """
    use_filter = bool(wall_layers)

    def wanted(entity) -> bool:
        if not use_filter:
            return True
        return entity.dxf.layer in wall_layers

    lines: list[Line] = []

    for entity in modelspace.query("LINE"):
        if not wanted(entity):
            continue
        start, end = entity.dxf.start, entity.dxf.end
        lines.append(
            (
                (start.x * to_meters, start.y * to_meters),
                (end.x * to_meters, end.y * to_meters),
            )
        )

    # Polylines carry most wall geometry in practice. Explode each into its
    # straight segments; bulges (arcs) are skipped rather than approximated,
    # since a curved wall is rare and a bad approximation is worse than an
    # honest omission the reviewer can see.
    for entity in modelspace.query("LWPOLYLINE POLYLINE"):
        if not wanted(entity):
            continue
        try:
            points = [(p[0] * to_meters, p[1] * to_meters) for p in entity.get_points("xy")]
        except (AttributeError, TypeError):
            try:
                points = [
                    (v.dxf.location.x * to_meters, v.dxf.location.y * to_meters)
                    for v in entity.vertices
                ]
            except Exception:  # noqa: BLE001
                continue

        if getattr(entity, "closed", False) and len(points) > 2:
            points.append(points[0])

        lines.extend(
            (points[i], points[i + 1])
            for i in range(len(points) - 1)
            if math.dist(points[i], points[i + 1]) > 1e-9
        )

    return lines, use_filter


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------


#: The transform from DXF metres to normalised plan metres: (min_x, max_y).
PlanTransform = tuple[float, float]


def _normalise_origin(walls: list[Wall]) -> PlanTransform:
    """Move the plan so its bounding box starts at the origin, in place.

    CAD drawings are routinely placed on a site grid tens of thousands of metres
    from zero. Left alone that wrecks float precision in the mesh and puts the
    model off-camera in every viewer.

    DXF is Y-UP; the IR is Y-DOWN to match image space. This is the one place
    that flip happens.

    Returns the transform it applied, which MUST be threaded through to anything
    else that converts raw DXF coordinates -- door blocks, room labels. It
    cannot be recovered from the walls afterwards, because by then their
    minimum is zero by construction.
    """
    if not walls:
        return (0.0, 0.0)

    xs = [p[0] for w in walls for p in (w.start, w.end)]
    ys = [p[1] for w in walls for p in (w.start, w.end)]
    min_x, max_y = min(xs), max(ys)

    for wall in walls:
        for point in (wall.start, wall.end):
            point[0] = round(point[0] - min_x, 4)
            point[1] = round(max_y - point[1], 4)

    return (min_x, max_y)


def _to_plan(
    x: float, y: float, to_meters: float, transform: PlanTransform
) -> tuple[float, float]:
    """Map a raw DXF point into normalised, Y-down plan metres."""
    min_x, max_y = transform
    return (x * to_meters - min_x, max_y - y * to_meters)


def _extent(walls: list[Wall]) -> tuple[float, float]:
    if not walls:
        return (1.0, 1.0)
    xs = [p[0] for w in walls for p in (w.start, w.end)]
    ys = [p[1] for w in walls for p in (w.start, w.end)]
    return (max(max(xs), 1.0), max(max(ys), 1.0))


# --------------------------------------------------------------------------
# Rooms
# --------------------------------------------------------------------------


def _detect_rooms_from_walls(walls: list[Wall]) -> list[Room]:
    """Find rooms by rasterizing the walls and reusing the raster detector.

    Deliberately not a second room-finding algorithm. Door-gap bridging and the
    touches-the-border test are subtle and already solved once; rendering to a
    mask means both recognizers share that work and any improvement lands in
    both at once.
    """
    if not walls:
        return []

    width_m, height_m = _extent(walls)
    scale = ROOM_RASTER_PX_PER_M
    # Keep the raster bounded on very large site plans.
    longest = max(width_m, height_m) * scale
    if longest > MAX_ROOM_RASTER_EDGE:
        scale *= MAX_ROOM_RASTER_EDGE / longest

    margin = 4
    width = int(width_m * scale) + margin * 2
    height = int(height_m * scale) + margin * 2
    mask = np.zeros((height, width), dtype=np.uint8)

    for wall in walls:
        thickness = max(1, int(round(wall.thickness * scale)))
        cv2.line(
            mask,
            (int(wall.start[0] * scale) + margin, int(wall.start[1] * scale) + margin),
            (int(wall.end[0] * scale) + margin, int(wall.end[1] * scale) + margin),
            255,
            thickness=thickness,
        )

    rooms = detect_rooms(mask, scale)
    # Undo the margin so room polygons sit in the same frame as the walls.
    offset = margin / scale
    for room in rooms:
        for point in room.polygon:
            point[0] = round(point[0] - offset, 4)
            point[1] = round(point[1] - offset, 4)
    return rooms


def _name_rooms(
    modelspace, rooms: list[Room], to_meters: float, transform: PlanTransform
) -> None:
    """Label rooms from TEXT/MTEXT entities that fall inside them, in place.

    Room names are the thing raster recognition most obviously lacks and DXF
    hands over for free. A room called "KITCHEN" can be given a kitchen's
    materials and fixtures automatically in Phase 2.
    """
    if not rooms:
        return

    labels: list[tuple[tuple[float, float], str]] = []
    for entity in modelspace.query("TEXT MTEXT"):
        text = (
            entity.plain_text() if hasattr(entity, "plain_text") else str(entity.dxf.text)
        ).strip()
        if not text:
            continue
        try:
            location = (
                entity.dxf.insert if entity.dxftype() == "MTEXT" else entity.dxf.insert
            )
        except AttributeError:
            continue
        labels.append((_to_plan(location.x, location.y, to_meters, transform), text))

    if not labels:
        return

    for room in rooms:
        polygon = np.array(room.polygon, dtype=np.float32)
        for (px, py), text in labels:
            if cv2.pointPolygonTest(polygon, (float(px), float(py)), False) < 0:
                continue
            room.name = text
            for pattern, room_type in ROOM_KEYWORDS:
                if pattern.search(text):
                    room.type = room_type
                    break
            room.confidence = 0.9
            break


def _promote_exterior_walls(walls: list[Wall], rooms: list[Room]) -> None:
    """Mark perimeter walls exterior. Thickness is measured, so unlike the
    raster path this only sets the type and never overrides the dimension."""
    if not rooms:
        return

    points = np.array([p for room in rooms for p in room.polygon], dtype=np.float32)
    if len(points) < 3:
        return

    hull = cv2.convexHull(points)
    for wall in walls:
        midpoint = (
            (wall.start[0] + wall.end[0]) / 2,
            (wall.start[1] + wall.end[1]) / 2,
        )
        if cv2.pointPolygonTest(hull, midpoint, measureDist=True) < 0.5:
            wall.type = WallType.EXTERIOR


# --------------------------------------------------------------------------
# Openings
# --------------------------------------------------------------------------


def _collect_openings(
    modelspace, walls: list[Wall], to_meters: float, transform: PlanTransform
) -> tuple[list[Opening], int]:
    """Turn door and window block references into wall-anchored openings.

    This is the capability raster recognition simply does not have. CAD draws
    doors and windows as named block INSERTs, so they can be found by name,
    sized from the block's own extents, and attached to the nearest wall.
    """
    openings: list[Opening] = []
    unattached = 0

    for entity in modelspace.query("INSERT"):
        name = entity.dxf.name or ""
        layer = entity.dxf.layer or ""
        haystack = f"{name} {layer}"

        if DOOR_PATTERN.search(haystack):
            opening_type = OpeningType.DOOR
        elif WINDOW_PATTERN.search(haystack):
            opening_type = OpeningType.WINDOW
        else:
            continue

        insert = entity.dxf.insert
        position = _to_plan(insert.x, insert.y, to_meters, transform)

        wall, offset, distance = _nearest_wall(walls, position)
        if wall is None or distance > MAX_OPENING_ATTACH_DISTANCE:
            unattached += 1
            continue

        width = _block_width(entity, to_meters) or (
            DEFAULT_DOOR_WIDTH if opening_type is OpeningType.DOOR else DEFAULT_WINDOW_WIDTH
        )
        # Keep the opening inside its wall: the validator rejects overruns, and
        # a block measured across its swing arc is often wider than the doorway.
        width = min(width, wall.length * 0.95)
        offset = min(max(offset, width / 2), wall.length - width / 2)

        openings.append(
            Opening(
                wall_id=wall.id,
                type=opening_type,
                offset=round(offset, 4),
                width=round(width, 4),
                height=2.1 if opening_type is OpeningType.DOOR else DEFAULT_WINDOW_HEIGHT,
                sill=0.0 if opening_type is OpeningType.DOOR else DEFAULT_WINDOW_SILL,
                confidence=0.8,
            )
        )

    return openings, unattached


def _nearest_wall(
    walls: list[Wall], point: tuple[float, float]
) -> tuple[Wall | None, float, float]:
    """Closest wall to a point, with the distance along it and away from it."""
    best: tuple[Wall | None, float, float] = (None, 0.0, math.inf)

    for wall in walls:
        (x0, y0), (x1, y1) = wall.start, wall.end
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            continue

        # Project onto the segment, clamped to its ends.
        t = max(0.0, min(1.0, ((point[0] - x0) * dx + (point[1] - y0) * dy) / length_sq))
        closest = (x0 + t * dx, y0 + t * dy)
        distance = math.dist(point, closest)

        if distance < best[2]:
            best = (wall, t * math.sqrt(length_sq), distance)

    return best


def _block_width(entity, to_meters: float) -> float | None:
    """Width of a block reference, from its own geometry.

    Doors vary from 0.6m cupboards to 1.8m double doors, and the block knows
    which it is. Falls back to None when the extents cannot be determined.
    """
    try:
        block = entity.block()
        if block is None:
            return None

        xs: list[float] = []
        ys: list[float] = []
        for sub in block:
            dxftype = sub.dxftype()
            if dxftype == "LINE":
                xs.extend((sub.dxf.start.x, sub.dxf.end.x))
                ys.extend((sub.dxf.start.y, sub.dxf.end.y))
            elif dxftype == "LWPOLYLINE":
                for px, py in sub.get_points("xy"):
                    xs.append(px)
                    ys.append(py)
            elif dxftype in {"ARC", "CIRCLE"}:
                center, radius = sub.dxf.center, sub.dxf.radius
                xs.extend((center.x - radius, center.x + radius))
                ys.extend((center.y - radius, center.y + radius))

        if not xs:
            return None

        scale_x = abs(getattr(entity.dxf, "xscale", 1.0)) or 1.0
        scale_y = abs(getattr(entity.dxf, "yscale", 1.0)) or 1.0
        width = (max(xs) - min(xs)) * scale_x * to_meters
        height = (max(ys) - min(ys)) * scale_y * to_meters

        # The block is drawn in its own frame and then rotated into place, so
        # the doorway is whichever extent is the smaller: a door block's larger
        # extent is usually its swing arc.
        candidate = min(width, height) if min(width, height) > 0.1 else max(width, height)
        return candidate if 0.3 <= candidate <= 5.0 else None
    except Exception:  # noqa: BLE001 - block resolution is best-effort
        return None
