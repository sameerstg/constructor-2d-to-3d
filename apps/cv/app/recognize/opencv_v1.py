"""Classical OpenCV floor plan recognizer.

No model, no weights, no GPU, no licence encumbrance. On clean vector-exported
floor plans -- which is most of what estate agents and builders actually hand
over -- this gets remarkably close, and when it fails it fails legibly instead
of hallucinating a plausible wrong answer.

It exists for three reasons beyond its own output quality:

1. It is the baseline. "Is the ML model worth its cost and licence terms?" is
   unanswerable without a number to beat.
2. It is the fallback when a learned model meets a drawing style outside its
   training distribution.
3. It lets the entire product -- upload, review editor, extruder, viewer, VR
   export -- be built and tested before anyone commits to a recognition vendor.

The core idea is that walls are the THICK dark structures. Text, dimension
lines, hatching and furniture symbols are thin. A distance transform separates
those two populations more reliably than any threshold on colour or length.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from skimage.morphology import skeletonize

from app.ir import (
    FloorPlanIR,
    Level,
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
from app.recognize.skeleton import skeleton_to_segments
from app.sources import SourceDocument, SourceKind

NAME = "opencv"
VERSION = "1.0.0"

# Longest edge the pipeline works at. Bigger inputs are downscaled first: beyond
# this, runtime grows quadratically while the skeleton gets no more informative.
MAX_WORKING_EDGE = 2000

# Scale of last resort: assume the typical interior wall is this thick.
#
# Be clear about how weak this is. Measured against synthetic plans with known
# ground truth, wall-thickness inference lands within ~15% when the drawing
# happens to use a conventional 0.20m partition and is off by 25-40% when it
# does not -- and nothing in the image says which case you are in. The error is
# in the ASSUMPTION, not the measurement, so no amount of better image
# processing recovers it.
#
# Metric scale has to come from outside the wall geometry: dimension text read
# by OCR, a stated drawing scale, or a human calibrating against a known
# distance. Until one of those exists this value keeps the pipeline running with
# plausible proportions, flagged as untrustworthy. Correct proportions with
# wrong absolute size is a usable intermediate state; refusing to produce
# anything is not.
ASSUMED_WALL_THICKNESS_M = 0.20

# Percentile used for the FIRST, rough thickness measurement, over raw ink.
#
# 60, not 80. Wall half-widths form two populations - thin interior partitions
# and a thick exterior perimeter - and on a drawing with solid-filled walls and
# generous margins the 80th percentile lands inside the EXTERIOR population.
# The wall mask is then cut at 0.6x that figure, which deletes every interior
# partition and leaves only the envelope: a Revit-style plot returned 4 walls
# instead of 14 for exactly this reason. Measured across every sample, 60
# recovers those partitions (4 -> 14 walls, matching ground truth) and changes
# nothing on the synthetic plans.
ROUGH_THICKNESS_PERCENTILE = 60

# Below this, the "walls" are hairlines rather than filled strokes, so thickness
# has hit the resolution floor and everything derived from it is unreliable.
HAIRLINE_HALF_WIDTH_PX = 2.5

# How much darker than the paper a pixel must be to count as ink.
#
# 45 was chosen by sweeping 25-55 across every sample. Too small and paper
# texture and JPEG ringing become "walls"; too large and the grey linework this
# exists to recover is dropped again. At 45 a Revit plot and a low-resolution
# scan both return exactly their true room count.
PAPER_INK_MARGIN = 45

# A room smaller than this is a cupboard, a hatch pattern, or noise.
MIN_ROOM_AREA_M2 = 1.2

# Walls shorter than this are almost always fragments of a door symbol.
MIN_WALL_LENGTH_M = 0.35

# Widest gap in a wall still treated as a doorway when sealing rooms shut.
# Above roughly this, a gap is an intentional open-plan threshold rather than a
# door, and bridging it would invent a wall that is not there.
MAX_DOOR_WIDTH_M = 1.1

# Scale-free equivalents of the two constants above, in units of WALL THICKNESS.
# Preferred for raster input because wall thickness is measured off the image
# while the metre scale is only guessed.
#
# 7.0 was chosen by sweeping 4.5-9.0 over every sample, synthetic and real, and
# counting rooms recovered. It is a genuine optimum, not a round number: below
# it, doorways in real plans stay open and whole floors leak into the exterior
# component; above it, the closing kernel swallows small rooms outright and
# low-resolution plans lose every room at 9.0.
DOOR_WIDTH_IN_WALL_THICKNESSES = 7.0

# A room is at least ~6 wall thicknesses across (1.2m for a 0.2m wall), which
# matches MIN_ROOM_AREA_M2 on a conventionally drawn plan but survives a bad
# scale estimate.
MIN_ROOM_SIDE_IN_WALL_THICKNESSES = 6.0


class OpenCVRecognizer:
    """Implements the `Recognizer` protocol for raster images."""

    name = NAME
    version = VERSION
    accepts = frozenset({SourceKind.RASTER})

    def recognize(
        self,
        document: SourceDocument,
        *,
        pixels_per_meter: float | None = None,
    ) -> FloorPlanIR:
        started = time.perf_counter()
        warnings: list[Warning] = []

        image = cv2.imdecode(np.frombuffer(document.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RecognizerError(
                "Could not decode that file as an image. Supported raster formats "
                "are PNG, JPEG, WebP, BMP and TIFF."
            )
        filename = document.filename

        original_height, original_width = image.shape[:2]
        working, downscale = _fit_working_size(image)

        ink = _binarize(working)
        if ink.mean() < 1.0:
            raise RecognizerError(
                "Image is almost entirely blank after thresholding - "
                "it does not look like a floor plan."
            )
        if ink.mean() > 200:
            raise RecognizerError(
                "Image is almost entirely dark after thresholding - "
                "try a cleaner scan or a higher-contrast export."
            )

        # Thickness is measured twice, and the second measurement is the one
        # that counts. The first runs on raw ink and only has to be good enough
        # to separate walls from text so the wall mask can be built. The second
        # runs on that mask, where the thin clutter is gone, and yields the
        # typical wall thickness that scale and geometry are derived from.
        rough_half_width_px = _estimate_wall_half_width(ink, percentile=ROUGH_THICKNESS_PERCENTILE)
        if rough_half_width_px <= HAIRLINE_HALF_WIDTH_PX:
            # A 1px half-width is not a measurement of anything: at that size the
            # low percentile is reading anti-aliasing on hairline linework, not
            # wall thickness. Fall back to the conservative estimate, which at
            # least suppresses some of the furniture and annotation. Drawings
            # with genuinely filled walls never reach this branch.
            rough_half_width_px = max(
                rough_half_width_px, _estimate_wall_half_width(ink, percentile=80)
            )
        if rough_half_width_px < 1.0:
            raise RecognizerError(
                "No wall-like structures found. The drawing may be too low "
                "resolution, or the walls too faint."
            )

        wall_mask = _extract_wall_mask(ink, rough_half_width_px)
        if cv2.countNonZero(wall_mask) == 0:
            raise RecognizerError("Wall extraction produced an empty mask.")

        # The 25th percentile, not the median. Half-widths form two clear
        # populations -- thin interior partitions and thick exterior walls --
        # and on most plans the exterior perimeter is the LONGER run, so the
        # median lands on the exterior and over-estimates the typical wall by
        # around 50%. A low percentile sits inside the interior population,
        # which is both the more standardized dimension and the one the
        # assumed-thickness constant below refers to.
        half_width_px = _estimate_wall_half_width(wall_mask, percentile=25)
        if half_width_px < 1.0:
            half_width_px = rough_half_width_px

        if half_width_px <= HAIRLINE_HALF_WIDTH_PX:
            # Two very different drawings land here: an unfilled CAD plot, where
            # walls are drawn as two hairlines with nothing between them, and a
            # low-resolution scan. In both, wall thickness cannot be measured,
            # so scale, wall geometry and the room thresholds derived from it
            # are all built on a number at its floor. Say so plainly rather than
            # emitting confident nonsense.
            warnings.append(
                Warning(
                    code="hairline_linework",
                    severity=Severity.ERROR,
                    message=(
                        "Walls in this drawing are thin outlines, not filled strokes, "
                        "so their thickness could not be measured. Expect spurious "
                        "walls from furniture and annotation, and unreliable rooms. "
                        "If this plan came from AutoCAD or Revit, upload the DXF "
                        "instead - it carries exact wall thicknesses."
                    ),
                )
            )

        # --- scale ---------------------------------------------------------
        # Resolve scale before converting anything, so there is exactly one
        # pixels -> metres conversion factor in the whole function.
        if pixels_per_meter is not None:
            scale = Scale(
                pixels_per_meter=pixels_per_meter,
                source=ScaleSource.MANUAL,
                confidence=1.0,
                note="Supplied by the caller.",
            )
            working_ppm = pixels_per_meter * downscale
        else:
            working_ppm = (half_width_px * 2) / ASSUMED_WALL_THICKNESS_M
            scale = Scale(
                pixels_per_meter=working_ppm / downscale,
                source=ScaleSource.HEURISTIC,
                confidence=0.25,
                note=(
                    f"Inferred by assuming a typical interior wall is "
                    f"{ASSUMED_WALL_THICKNESS_M:.2f}m thick. Expect 15-40% error."
                ),
            )
            warnings.append(
                Warning(
                    code="scale_is_a_guess",
                    severity=Severity.ERROR,
                    message=(
                        "Scale was guessed from wall thickness, not measured. Room "
                        "proportions are reliable but absolute sizes may be off by "
                        "up to 40%. Calibrate against a known distance before this "
                        "model is used for anything dimensional."
                    ),
                )
            )

        # --- walls ---------------------------------------------------------
        skeleton = skeletonize(wall_mask > 0)
        segments_px = skeleton_to_segments(
            skeleton.astype(np.uint8),
            simplify_tolerance=max(2.0, half_width_px * 0.5),
            snap_tolerance=max(4.0, half_width_px * 1.5),
            min_length=max(8.0, MIN_WALL_LENGTH_M * working_ppm),
        )
        if not segments_px:
            raise RecognizerError(
                "Found wall-like pixels but could not trace any wall lines from them."
            )

        thickness_m = min(2.0, max(0.05, (half_width_px * 2) / working_ppm))
        walls = [
            Wall(
                start=[round(x0 / working_ppm, 4), round(y0 / working_ppm, 4)],
                end=[round(x1 / working_ppm, 4), round(y1 / working_ppm, 4)],
                thickness=round(thickness_m, 4),
                # Exterior vs interior needs the building outline, which we only
                # reliably have after room detection. Everything starts interior
                # and gets promoted below.
                type=WallType.INTERIOR,
                confidence=0.6,
            )
            for (x0, y0), (x1, y1) in segments_px
        ]

        # --- rooms ---------------------------------------------------------
        rooms = detect_rooms(wall_mask, working_ppm, wall_half_width_px=half_width_px)
        if not rooms:
            warnings.append(
                Warning(
                    code="no_rooms_detected",
                    severity=Severity.WARN,
                    message=(
                        "No enclosed rooms were found. This usually means walls are "
                        "broken by door gaps that were not bridged - the 3D model "
                        "will have walls but no floors."
                    ),
                )
            )

        _promote_exterior_walls(walls, rooms)

        # --- openings ------------------------------------------------------
        # Deliberately not attempted. Reliable door and window detection needs
        # symbol recognition, which is the ML recognizer's job. Phase 1 of the
        # proposal is explicitly "walls only, empty", so this matches scope --
        # and a half-working door detector would create more review work than
        # it saves. The IR and extruder fully support openings already; they
        # arrive via the review editor or a later recognizer.
        warnings.append(
            Warning(
                code="no_openings_detected",
                severity=Severity.INFO,
                message=(
                    "Doors and windows are not detected by the classical recognizer. "
                    "Add them in the review editor, or switch to a recognizer that "
                    "supports symbol detection."
                ),
            )
        )

        low_confidence = sum(1 for w in walls if w.confidence < 0.7)
        if low_confidence:
            warnings.append(
                Warning(
                    code="walls_need_review",
                    severity=Severity.INFO,
                    message=f"{low_confidence} of {len(walls)} walls should be checked by a human.",
                )
            )

        level = Level(name="Ground Floor", walls=walls, rooms=rooms)

        return FloorPlanIR(
            name=filename.rsplit(".", 1)[0],
            source=SourceImage(
                filename=filename, width=original_width, height=original_height
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
# Stages
# --------------------------------------------------------------------------


def _fit_working_size(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale to a sane working resolution.

    Returns the image and the factor by which it was reduced, so pixel
    measurements can be mapped back onto the original raster.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= MAX_WORKING_EDGE:
        return image, 1.0

    factor = MAX_WORKING_EDGE / longest
    resized = cv2.resize(
        image, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA
    )
    return resized, factor


def _binarize(image: np.ndarray) -> np.ndarray:
    """Produce an ink mask: 255 where the drawing has marks, 0 for paper.

    Otsu alone is not enough, and the reason is worth stating: it assumes the
    histogram is BIMODAL, and an architectural drawing is not. A real plan has
    three populations -- white paper, mid-grey secondary linework (railings,
    glazing, hatching, and on many drawings the exterior walls themselves), and
    black primary linework. Otsu splits the two extremes and throws the middle
    away with the paper.

    That is not cosmetic. On a real plan it deleted the light-grey outer walls,
    leaving gaps in the building envelope far wider than any doorway, so the
    entire interior merged with the outdoors and every large room was discarded:
    the recognizer returned five closets and no bedrooms.

    So the threshold is also pinned relative to the PAPER: anything meaningfully
    darker than the page is ink, whatever Otsu thinks. Taking the more inclusive
    of the two keeps Otsu's protection on genuinely dark drawings while
    recovering grey linework on pale ones.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Otsu over a lightly blurred image. Adaptive thresholding is tempting but
    # it turns large solid-black poche walls into hollow outlines, which is
    # exactly the signal the distance transform depends on.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_threshold, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # The paper is the brightest dominant mode. Searching only the top half
    # avoids mistaking a large solid-black poche for the page.
    histogram = cv2.calcHist([blurred], [0], None, [256], [0, 256]).ravel()
    paper_level = int(np.argmax(histogram[128:]) + 128)

    threshold = max(float(otsu_threshold), float(paper_level - PAPER_INK_MARGIN))
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # Close single-pixel gaps from JPEG ringing or thin anti-aliasing.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def _estimate_wall_half_width(mask: np.ndarray, *, percentile: float) -> float:
    """Estimate half the wall thickness, in pixels.

    Every pixel on a shape's medial axis carries, in the distance transform, its
    own distance to the nearest edge -- that is, the local half-thickness. So
    sampling the distance transform along the skeleton gives a distribution of
    half-thicknesses for every stroke in the mask.

    Which percentile to take depends on what the mask contains, so the caller
    chooses. On raw ink -- walls mixed in with text and furniture -- a high
    percentile is needed to land inside the wall population at all. On an
    already-cleaned wall mask the median is right, because it picks the TYPICAL
    wall rather than the thickest, and the typical wall is what scale should be
    calibrated against.
    """
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    medial = skeletonize(mask > 0)
    samples = distance[medial]
    samples = samples[samples > 0]
    if samples.size == 0:
        return 0.0
    return float(np.percentile(samples, percentile))


def _extract_wall_mask(ink: np.ndarray, half_width_px: float) -> np.ndarray:
    """Keep only strokes thick enough to be walls.

    This is a morphological opening by reconstruction, done via the distance
    transform: keep the core of anything at least a wall's thickness, then grow
    that core back out to its original width, clipped to the original ink.

    Compared with a plain `morphologyEx(MORPH_OPEN)` this preserves wall ends
    and corners rather than rounding them off, which matters because the corners
    are precisely where the skeleton needs to form clean junctions.
    """
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)

    core_radius = max(1.0, half_width_px * 0.6)
    core = (distance >= core_radius).astype(np.uint8) * 255
    if cv2.countNonZero(core) == 0:
        return np.zeros_like(ink)

    grow = int(round(core_radius)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
    grown = cv2.dilate(core, kernel)

    mask = cv2.bitwise_and(grown, ink)

    # Close hairline breaks from anti-aliasing only. Door-sized gaps are left
    # alone here on purpose: they are real, and the wall tracing downstream
    # should see them. Room detection bridges them separately for its own use.
    hairline = max(3, int(round(half_width_px * 0.8)) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hairline, hairline))
    )

    return _drop_small_components(mask, min_area=int(half_width_px**2 * 12))


def _bridge_door_gaps(wall_mask: np.ndarray, gap_px: int) -> np.ndarray:
    """Temporarily seal doorways so rooms come out as enclosed regions.

    A single unbridged doorway leaks a room into the corridor, the corridor into
    the next room, and eventually the whole interior into one blob -- which is
    exactly the "1 room detected" failure this function exists to fix.

    The trick is that the closing kernels are LINES, not disks. A horizontal
    line only bridges gaps along horizontal walls, and a vertical line only
    along vertical ones. A disk of the same size would also close the gap
    ACROSS a corridor, merging its two side walls and deleting the corridor from
    the plan. Directional closing bridges a 1m doorway while leaving a 1.2m
    corridor open, which a symmetric kernel cannot do at any size.

    The result is used only for room detection. Wall geometry keeps the real,
    unbridged mask.
    """
    span = max(3, gap_px | 1)
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, span))

    bridged = cv2.bitwise_or(
        cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, horizontal),
        cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, vertical),
    )
    return cv2.bitwise_or(bridged, wall_mask)


def _drop_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def detect_rooms(
    wall_mask: np.ndarray,
    pixels_per_meter: float,
    *,
    wall_half_width_px: float | None = None,
) -> list[Room]:
    """Find enclosed floor regions as the connected components of free space.

    Free space is everything that is not wall. Components touching the image
    border are outside the building, not rooms -- that single test removes the
    surrounding page, and it is why the wall mask had its door gaps bridged
    first: one unbridged doorway on the outer wall leaks the whole interior into
    the exterior component and every room disappears at once.

    Two thresholds decide everything here: how wide a gap counts as a doorway,
    and how small a region counts as a room. Expressing them in METRES makes
    both hostage to the scale estimate -- and on a real drawing that estimate can
    be off by 2-3x, which scales the area threshold by its SQUARE and silently
    discards every genuine room. That is exactly how this returned zero rooms on
    real-world plans while looking fine on synthetics.

    So when the caller can supply a MEASURED wall half-width, both thresholds are
    derived from it instead. Wall thickness is read directly off the image and is
    reliable; the metre scale is not. A doorway is ~4-10x the wall thickness and
    a room is at least a few thicknesses across, whatever the drawing's scale.

    `wall_half_width_px` should be omitted only when `pixels_per_meter` is known
    exactly -- from a DXF's stated units, say -- where metres are the better unit.
    """
    if wall_half_width_px and wall_half_width_px > 0:
        thickness_px = wall_half_width_px * 2
        gap_px = int(round(DOOR_WIDTH_IN_WALL_THICKNESSES * thickness_px))
        min_area_px = (MIN_ROOM_SIDE_IN_WALL_THICKNESSES * thickness_px) ** 2
    else:
        gap_px = int(round(MAX_DOOR_WIDTH_M * pixels_per_meter))
        min_area_px = MIN_ROOM_AREA_M2 * (pixels_per_meter**2)

    sealed = _bridge_door_gaps(wall_mask, gap_px=gap_px)
    free_space = cv2.bitwise_not(sealed)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=4)

    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])

    rooms: list[Room] = []
    for label in range(1, count):
        if label in border_labels:
            continue
        if stats[label, cv2.CC_STAT_AREA] < min_area_px:
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(contour, closed=True)
        approximated = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approximated) < 3:
            continue

        polygon = [
            [round(float(x) / pixels_per_meter, 4), round(float(y) / pixels_per_meter, 4)]
            for [[x, y]] in approximated
        ]
        area_m2 = stats[label, cv2.CC_STAT_AREA] / (pixels_per_meter**2)

        rooms.append(
            Room(
                type=RoomType.UNKNOWN,
                polygon=polygon,
                area=round(float(area_m2), 3),
                # Geometry is trustworthy; the label is not. Room TYPE needs
                # OCR of the room name or a classifier, so it stays unknown and
                # the confidence reflects that honestly.
                confidence=0.5,
            )
        )

    return sorted(rooms, key=lambda r: r.area or 0, reverse=True)


def _promote_exterior_walls(walls: list[Wall], rooms: list[Room]) -> None:
    """Mark walls on the building perimeter as exterior, in place.

    Cheap convex-hull test: a wall whose midpoint sits near the hull of all room
    geometry is on the outside. Wrong for genuinely concave footprints -- an
    L-shaped plan will miss walls in the notch -- but it costs nothing and only
    drives material defaults, which a human overrides in the configurator.
    """
    if not rooms:
        return

    points = np.array(
        [point for room in rooms for point in room.polygon], dtype=np.float32
    )
    if len(points) < 3:
        return

    hull = cv2.convexHull(points)
    tolerance = 0.5  # metres

    for wall in walls:
        midpoint = (
            (wall.start[0] + wall.end[0]) / 2,
            (wall.start[1] + wall.end[1]) / 2,
        )
        # Negative distance means outside the hull; near zero means on it.
        distance = cv2.pointPolygonTest(hull, midpoint, measureDist=True)
        if distance < tolerance:
            wall.type = WallType.EXTERIOR
            wall.thickness = max(wall.thickness, 0.25)

