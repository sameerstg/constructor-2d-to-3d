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

# A room smaller than this is a cupboard, a hatch pattern, or noise.
MIN_ROOM_AREA_M2 = 1.2

# Walls shorter than this are almost always fragments of a door symbol.
MIN_WALL_LENGTH_M = 0.35

# Widest gap in a wall still treated as a doorway when sealing rooms shut.
# Above roughly this, a gap is an intentional open-plan threshold rather than a
# door, and bridging it would invent a wall that is not there.
MAX_DOOR_WIDTH_M = 1.1


class OpenCVRecognizer:
    """Implements the `Recognizer` protocol."""

    name = NAME
    version = VERSION

    def recognize(
        self,
        image: np.ndarray,
        *,
        filename: str = "upload",
        pixels_per_meter: float | None = None,
    ) -> FloorPlanIR:
        started = time.perf_counter()
        warnings: list[Warning] = []

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
        rough_half_width_px = _estimate_wall_half_width(ink, percentile=80)
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
        rooms = _detect_rooms(wall_mask, working_ppm)
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
    """Produce an ink mask: 255 where the drawing has marks, 0 for paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Otsu over a lightly blurred image. Adaptive thresholding is tempting but
    # it turns large solid-black poche walls into hollow outlines, which is
    # exactly the signal the distance transform depends on.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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


def _detect_rooms(wall_mask: np.ndarray, pixels_per_meter: float) -> list[Room]:
    """Find enclosed floor regions as the connected components of free space.

    Free space is everything that is not wall. Components touching the image
    border are outside the building, not rooms -- that single test removes the
    surrounding page, and it is why the wall mask had its door gaps bridged
    first: one unbridged doorway on the outer wall leaks the whole interior into
    the exterior component and every room disappears at once.
    """
    sealed = _bridge_door_gaps(wall_mask, gap_px=int(round(MAX_DOOR_WIDTH_M * pixels_per_meter)))
    free_space = cv2.bitwise_not(sealed)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=4)

    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    min_area_px = MIN_ROOM_AREA_M2 * (pixels_per_meter**2)

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

