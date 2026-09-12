"""Skeleton -> line segments.

Turning a one-pixel-wide skeleton into a small set of clean straight segments is
the least glamorous and most important step in the classical pipeline. Do it
naively and you get four thousand one-pixel segments; do it well and a typical
apartment comes out as forty walls that a human can actually review.

The stages, in order:

1. trace  - walk the skeleton into polylines between nodes (endpoints/junctions)
2. simplify - Douglas-Peucker, to drop pixel-stair noise
3. snap   - pull near-axis segments onto exact axes, then cluster the shared
            coordinates so collinear walls agree to the millimetre
4. merge  - join collinear neighbours and drop slivers

Everything here works in PIXELS. Conversion to metres happens once, later, in
the recognizer, so there is exactly one place where scale can be got wrong.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString

# 8-connected neighbourhood offsets
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

Pixel = tuple[int, int]  # (row, col)
Segment = tuple[tuple[float, float], tuple[float, float]]  # ((x0,y0),(x1,y1))


# --------------------------------------------------------------------------
# 1. Trace
# --------------------------------------------------------------------------


def _neighbor_map(skeleton: np.ndarray) -> dict[Pixel, list[Pixel]]:
    """Adjacency for every set pixel in the skeleton."""
    rows, cols = np.nonzero(skeleton)
    occupied = set(zip(rows.tolist(), cols.tolist(), strict=True))
    height, width = skeleton.shape

    adjacency: dict[Pixel, list[Pixel]] = {}
    for pixel in occupied:
        r, c = pixel
        neighbors = []
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) in occupied:
                neighbors.append((nr, nc))
        adjacency[pixel] = neighbors
    return adjacency


def trace_polylines(skeleton: np.ndarray) -> list[list[Pixel]]:
    """Walk a binary skeleton into polylines running between nodes.

    A node is any pixel that is not a simple pass-through: an endpoint (one
    neighbour) or a junction (three or more). Paths run node to node. Closed
    loops with no node at all -- a skeletonised ring corridor, say -- would
    otherwise be skipped entirely, so they get an arbitrary start pixel
    promoted to a node.
    """
    adjacency = _neighbor_map(skeleton)
    if not adjacency:
        return []

    nodes = {p for p, neighbors in adjacency.items() if len(neighbors) != 2}

    polylines: list[list[Pixel]] = []
    visited_edges: set[frozenset[Pixel]] = set()

    def walk(start: Pixel, first_step: Pixel) -> list[Pixel]:
        path = [start, first_step]
        previous, current = start, first_step
        # Follow the corridor until it opens out at another node.
        while current not in nodes:
            options = [p for p in adjacency[current] if p != previous]
            if not options:
                break
            previous, current = current, options[0]
            path.append(current)
            if current == start:  # closed the loop
                break
        return path

    for node in nodes:
        for neighbor in adjacency[node]:
            edge = frozenset((node, neighbor))
            if edge in visited_edges:
                continue
            path = walk(node, neighbor)
            # Mark every edge along the path so we do not retrace it from the
            # far end and end up with each wall twice.
            for a, b in zip(path, path[1:], strict=False):
                visited_edges.add(frozenset((a, b)))
            if len(path) >= 2:
                polylines.append(path)

    # Pure loops: no node was ever promoted, so nothing above touched them.
    unvisited = {
        p
        for p, neighbors in adjacency.items()
        if len(neighbors) == 2
        and all(frozenset((p, n)) not in visited_edges for n in neighbors)
    }
    while unvisited:
        start = unvisited.pop()
        path = walk(start, adjacency[start][0])
        for a, b in zip(path, path[1:], strict=False):
            visited_edges.add(frozenset((a, b)))
        unvisited -= set(path)
        if len(path) >= 2:
            polylines.append(path)

    return polylines


# --------------------------------------------------------------------------
# 2 + 3. Simplify and snap
# --------------------------------------------------------------------------


def polyline_to_segments(
    polyline: list[Pixel], *, simplify_tolerance: float
) -> list[Segment]:
    """Douglas-Peucker a traced pixel path down to straight segments.

    Input is (row, col); output is (x, y) == (col, row), because everything
    downstream of here is image/plan space with x to the right.
    """
    if len(polyline) < 2:
        return []

    points = [(float(c), float(r)) for r, c in polyline]
    # A two-point path is already a segment; shapely.simplify would be a no-op.
    if len(points) > 2:
        simplified = LineString(points).simplify(simplify_tolerance, preserve_topology=False)
        points = list(simplified.coords)

    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def snap_to_axes(segments: list[Segment], *, angle_tolerance_deg: float = 8.0) -> list[Segment]:
    """Pull near-horizontal and near-vertical segments onto exact axes.

    Residential plans are overwhelmingly axis-aligned, and a wall that is 0.4
    degrees off looks fine in 2D but produces visibly skewed geometry and ugly
    wall junctions once extruded. Segments further off axis than the tolerance
    are left alone -- genuine diagonal walls exist and must survive.
    """
    snapped: list[Segment] = []
    for (x0, y0), (x1, y1) in segments:
        dx, dy = x1 - x0, y1 - y0
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))

        if angle <= angle_tolerance_deg:  # near-horizontal
            y = (y0 + y1) / 2
            snapped.append(((x0, y), (x1, y)))
        elif angle >= 90 - angle_tolerance_deg:  # near-vertical
            x = (x0 + x1) / 2
            snapped.append(((x, y0), (x, y1)))
        else:
            snapped.append(((x0, y0), (x1, y1)))
    return snapped


def cluster_coordinates(segments: list[Segment], *, tolerance: float) -> list[Segment]:
    """Make walls that are meant to line up actually line up.

    Two parallel walls detected at y=101.2 and y=102.8 are almost certainly the
    same architectural line. Cluster all x values and all y values
    independently, then snap each to its cluster's mean. This closes the
    hairline gaps at corners that would otherwise leave the extruded mesh
    non-watertight.
    """
    if not segments:
        return []

    def build_clusters(values: list[float]) -> dict[float, float]:
        ordered = sorted(set(values))
        mapping: dict[float, float] = {}
        group = [ordered[0]]
        for value in ordered[1:]:
            if value - group[-1] <= tolerance:
                group.append(value)
            else:
                center = sum(group) / len(group)
                mapping.update({v: center for v in group})
                group = [value]
        center = sum(group) / len(group)
        mapping.update({v: center for v in group})
        return mapping

    xs = [p[0] for seg in segments for p in seg]
    ys = [p[1] for seg in segments for p in seg]
    x_map, y_map = build_clusters(xs), build_clusters(ys)

    return [
        ((x_map[x0], y_map[y0]), (x_map[x1], y_map[y1]))
        for (x0, y0), (x1, y1) in segments
    ]


# --------------------------------------------------------------------------
# 4. Merge
# --------------------------------------------------------------------------


def _segment_length(segment: Segment) -> float:
    (x0, y0), (x1, y1) = segment
    return math.hypot(x1 - x0, y1 - y0)


def merge_collinear(segments: list[Segment], *, tolerance: float) -> list[Segment]:
    """Join axis-aligned segments that share a line and touch or overlap.

    Tracing splits a single long wall at every doorway stub and T-junction. Left
    alone that turns one wall into six, which makes the review editor's wall
    list useless. Only exactly-axis-aligned runs are merged; diagonals are rare
    enough that merging them is not worth the risk of fusing two real walls.
    """
    horizontal: dict[float, list[Segment]] = defaultdict(list)
    vertical: dict[float, list[Segment]] = defaultdict(list)
    others: list[Segment] = []

    for segment in segments:
        (x0, y0), (x1, y1) = segment
        if abs(y1 - y0) < 1e-6:
            horizontal[y0].append(segment)
        elif abs(x1 - x0) < 1e-6:
            vertical[x0].append(segment)
        else:
            others.append(segment)

    merged: list[Segment] = list(others)

    def sweep(groups: dict[float, list[Segment]], axis: int) -> None:
        """Interval-union along one axis. axis=0 merges horizontals by x."""
        for fixed, group in groups.items():
            intervals = sorted(
                (min(a[axis], b[axis]), max(a[axis], b[axis])) for a, b in group
            )
            current_lo, current_hi = intervals[0]
            runs: list[tuple[float, float]] = []
            for lo, hi in intervals[1:]:
                if lo <= current_hi + tolerance:  # touching or overlapping
                    current_hi = max(current_hi, hi)
                else:
                    runs.append((current_lo, current_hi))
                    current_lo, current_hi = lo, hi
            runs.append((current_lo, current_hi))

            for lo, hi in runs:
                if axis == 0:
                    merged.append(((lo, fixed), (hi, fixed)))
                else:
                    merged.append(((fixed, lo), (fixed, hi)))

    sweep(horizontal, axis=0)
    sweep(vertical, axis=1)
    return merged


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def skeleton_to_segments(
    skeleton: np.ndarray,
    *,
    simplify_tolerance: float = 2.0,
    snap_tolerance: float = 6.0,
    min_length: float = 12.0,
) -> list[Segment]:
    """Full skeleton -> clean segment list, all distances in pixels."""
    segments: list[Segment] = []
    for polyline in trace_polylines(skeleton):
        segments.extend(polyline_to_segments(polyline, simplify_tolerance=simplify_tolerance))

    segments = [s for s in segments if _segment_length(s) >= 1.0]
    segments = snap_to_axes(segments)
    segments = cluster_coordinates(segments, tolerance=snap_tolerance)
    # Clustering can collapse a short segment to zero length; drop those before
    # merging so they do not create bogus zero-width intervals.
    segments = [s for s in segments if _segment_length(s) > 1e-6]
    segments = merge_collinear(segments, tolerance=snap_tolerance)

    return [s for s in segments if _segment_length(s) >= min_length]
