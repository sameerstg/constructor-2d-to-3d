"""Pair CAD double-lines into wall centrelines.

Architectural CAD does not draw a wall as a centreline. It draws its two FACES
as a pair of parallel lines a wall-thickness apart, and the centreline -- which
is what the IR wants and what extrusion needs -- is implied. Recovering it is
the main geometric problem in reading a DXF floor plan.

The payoff is large: the gap between a matched pair IS the wall thickness,
measured rather than assumed. That is one of the two things raster recognition
cannot recover at all (the other being scale), and it comes out exact here.

Two lines pair up when they are:

1. parallel, within a small angular tolerance,
2. separated by a plausible wall thickness,
3. overlapping when projected onto their shared direction -- two parallel walls
   at opposite ends of the building are not a pair,

and the pairing is greedy over the best-scoring candidates, so each face is
consumed once. Lines left unpaired are reported as single-line walls rather than
dropped, because a missing wall is a hole in the building and a spurious one is
a visible, fixable defect.

All distances are in METRES; the DXF reader converts units before calling in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass(frozen=True, slots=True)
class PairedWall:
    """A recovered centreline plus the thickness measured between its faces."""

    start: Point
    end: Point
    thickness: float
    #: True when this came from a matched pair; False for an unpaired single
    #: line, whose thickness is a default rather than a measurement.
    measured: bool

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


def _direction(line: Line) -> tuple[float, float, float]:
    """Unit direction and length of a line."""
    (x0, y0), (x1, y1) = line
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0, 0.0
    return dx / length, dy / length, length


def _are_parallel(a: Line, b: Line, *, angle_tolerance_deg: float) -> bool:
    ax, ay, _ = _direction(a)
    bx, by, _ = _direction(b)
    # |cross| of two unit vectors is |sin(angle)|. Direction-agnostic, so a line
    # drawn right-to-left still pairs with its left-to-right neighbour.
    return abs(ax * by - ay * bx) <= math.sin(math.radians(angle_tolerance_deg))


def _perpendicular_distance(a: Line, b: Line) -> float:
    """Distance from line b's midpoint to the infinite line through a."""
    (ax0, ay0), _ = a
    ux, uy, length = _direction(a)
    if length == 0:
        return math.inf

    (bx0, by0), (bx1, by1) = b
    mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
    return abs((mx - ax0) * uy - (my - ay0) * ux)


def _overlap_fraction(a: Line, b: Line) -> float:
    """How much b overlaps a when both are projected onto a's direction.

    Returned as a fraction of the shorter line, so a short stub against a long
    wall still scores 1.0 when fully covered -- which is what we want, since
    walls are routinely drawn as one long face against several short ones.
    """
    (ax0, ay0), _ = a
    ux, uy, a_len = _direction(a)
    if a_len == 0:
        return 0.0

    def project(p: Point) -> float:
        return (p[0] - ax0) * ux + (p[1] - ay0) * uy

    a_lo, a_hi = 0.0, a_len
    b_projected = sorted((project(b[0]), project(b[1])))
    b_lo, b_hi = b_projected

    overlap = min(a_hi, b_hi) - max(a_lo, b_lo)
    if overlap <= 0:
        return 0.0
    return overlap / min(a_len, b_hi - b_lo) if (b_hi - b_lo) > 0 else 0.0


def _centreline(a: Line, b: Line) -> Line:
    """Centreline of a matched pair, trimmed to their common extent.

    Built along a's direction so the result inherits the drawing's own axis
    alignment rather than drifting by however much the two faces disagree.
    """
    (ax0, ay0), _ = a
    ux, uy, a_len = _direction(a)

    def project(p: Point) -> float:
        return (p[0] - ax0) * ux + (p[1] - ay0) * uy

    b_projected = sorted((project(b[0]), project(b[1])))
    lo = max(0.0, b_projected[0])
    hi = min(a_len, b_projected[1])
    if hi <= lo:  # no common extent; fall back to a's full span
        lo, hi = 0.0, a_len

    # Offset half way towards b, perpendicular to a.
    (bx0, by0), (bx1, by1) = b
    mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
    signed = (mx - ax0) * uy - (my - ay0) * ux
    half = signed / 2
    ox, oy = uy * half, -ux * half

    return (
        (ax0 + ux * lo + ox, ay0 + uy * lo + oy),
        (ax0 + ux * hi + ox, ay0 + uy * hi + oy),
    )


def pair_walls(
    lines: list[Line],
    *,
    min_thickness: float = 0.05,
    max_thickness: float = 0.60,
    angle_tolerance_deg: float = 2.0,
    min_overlap: float = 0.35,
    default_thickness: float = 0.10,
    min_length: float = 0.30,
) -> tuple[list[PairedWall], int]:
    """Match parallel line pairs into centrelines.

    Returns the walls and the number of lines left unpaired, so the caller can
    warn honestly about how much of the drawing it actually understood.
    """
    candidates = [line for line in lines if math.dist(*line) >= min_length]

    # Score every plausible pairing first, then take them greedily in quality
    # order. Pairing in input order instead would let an early mediocre match
    # consume a face that was a much better partner for something else.
    scored: list[tuple[float, int, int, float]] = []
    for i, a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if not _are_parallel(a, b, angle_tolerance_deg=angle_tolerance_deg):
                continue

            separation = _perpendicular_distance(a, b)
            if not (min_thickness <= separation <= max_thickness):
                continue

            overlap = _overlap_fraction(a, b)
            if overlap < min_overlap:
                continue

            # Prefer long, well-overlapped pairs. Thinner separations break ties,
            # so a face pairs with its own wall rather than across a corridor.
            quality = overlap * min(math.dist(*a), math.dist(*b)) / separation
            scored.append((quality, i, j, separation))

    scored.sort(reverse=True)

    used: set[int] = set()
    walls: list[PairedWall] = []
    for _, i, j, separation in scored:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        start, end = _centreline(candidates[i], candidates[j])
        if math.dist(start, end) < min_length:
            continue
        walls.append(
            PairedWall(start=start, end=end, thickness=separation, measured=True)
        )

    unpaired = 0
    for index, line in enumerate(candidates):
        if index in used:
            continue
        unpaired += 1
        walls.append(
            PairedWall(
                start=line[0], end=line[1], thickness=default_thickness, measured=False
            )
        )

    return walls, unpaired
