"""
Floor Plan Intermediate Representation (IR) - the canonical data contract.

This module is the SINGLE SOURCE OF TRUTH for the IR. The JSON Schema at
packages/schema/ir.schema.json and the TypeScript types at
packages/schema/src/ir.ts are both GENERATED from these models by
`python -m app.export_schema`. Do not hand-edit those files.

Design notes that matter:

* Coordinates are METRES in plan space, X right and Y DOWN. Y-down matches
  image space, so the recognizer never has to flip an axis and we never chase
  mirrored-plan bugs. The extruder is the single place that converts to a
  Y-up 3D frame (plan x -> world x, plan y -> world z).

* Openings are anchored to a wall by (wall_id, offset) rather than by absolute
  position. This is deliberate: when a human drags a wall in the review editor,
  its doors and windows travel with it instead of being orphaned in space.

* Every recognized entity carries a `confidence`. The review editor sorts by it
  so a human looks at the five things the recognizer was unsure about, not the
  two hundred it was fine with. That is what makes imperfect recognition
  commercially usable.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"

# A point in plan space, in metres. (x, y) with y pointing DOWN.
Point = Annotated[list[float], Field(min_length=2, max_length=2)]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------


class ScaleSource(StrEnum):
    """How we arrived at the pixels-per-metre figure, worst to best."""

    ASSUMED = "assumed"  # fell back to a default; probably wrong
    HEURISTIC = "heuristic"  # inferred from typical door widths etc.
    OCR = "ocr"  # read from dimension text on the drawing
    MANUAL = "manual"  # a human typed it in; trust absolutely


class Scale(Base):
    """Pixel-to-metre calibration. A wrong scale ruins everything downstream."""

    pixels_per_meter: float = Field(gt=0, alias="pixelsPerMeter")
    source: ScaleSource = ScaleSource.ASSUMED
    confidence: Confidence = 0.0
    note: str | None = None


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


class WallType(StrEnum):
    EXTERIOR = "exterior"
    INTERIOR = "interior"
    PARTITION = "partition"


class Wall(Base):
    """A wall segment, stored as a CENTRELINE plus a thickness.

    Centreline rather than two faces because it makes junctions tractable:
    walls sharing an endpoint miter cleanly, and dragging a corner in the
    editor moves every wall that meets there.
    """

    id: str = Field(default_factory=lambda: _uid("w"))
    start: Point
    end: Point
    thickness: float = Field(default=0.20, gt=0, le=2.0)
    height: float | None = Field(default=None, gt=0, le=10.0)
    type: WallType = WallType.INTERIOR
    confidence: Confidence = 1.0

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @model_validator(mode="after")
    def _reject_degenerate(self) -> Wall:
        if self.length < 1e-6:
            raise ValueError(f"wall {self.id} is zero-length")
        return self


class OpeningType(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"  # a hole with no leaf, e.g. a cased opening


class SwingDirection(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class Opening(Base):
    """A door or window, positioned ALONG a wall rather than in world space."""

    id: str = Field(default_factory=lambda: _uid("o"))
    wall_id: str = Field(alias="wallId")
    type: OpeningType = OpeningType.DOOR
    # Distance in metres from the wall's `start` to the opening's centre.
    offset: float = Field(ge=0)
    width: float = Field(gt=0, le=10.0)
    height: float = Field(default=2.1, gt=0, le=10.0)
    # Height of the opening's bottom edge above the floor. 0 for doors.
    sill: float = Field(default=0.0, ge=0)
    swing: SwingDirection | None = None
    confidence: Confidence = 1.0


class RoomType(StrEnum):
    UNKNOWN = "unknown"
    LIVING = "living"
    KITCHEN = "kitchen"
    DINING = "dining"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    HALL = "hall"
    CORRIDOR = "corridor"
    STORAGE = "storage"
    GARAGE = "garage"
    BALCONY = "balcony"
    STAIRS = "stairs"


class Room(Base):
    """An enclosed floor region. Polygon is a simple ring, no self-intersection."""

    id: str = Field(default_factory=lambda: _uid("r"))
    name: str | None = None
    type: RoomType = RoomType.UNKNOWN
    polygon: Annotated[list[Point], Field(min_length=3)]
    area: float | None = Field(default=None, ge=0)
    confidence: Confidence = 1.0


# --------------------------------------------------------------------------
# Level and document
# --------------------------------------------------------------------------


class Level(Base):
    """One storey. The MVP emits exactly one, but the shape is already
    multi-storey ready so Phase 3's multi-floor work is additive rather than
    a schema migration."""

    id: str = Field(default_factory=lambda: _uid("lvl"))
    name: str = "Ground Floor"
    elevation: float = 0.0
    wall_height: float = Field(default=2.7, gt=0, le=10.0, alias="wallHeight")
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Warning(Base):
    """Something the recognizer wants a human to look at.

    These drive the review editor's worklist. A recognizer that emits honest
    warnings is far more useful than one that silently guesses.
    """

    code: str
    message: str
    severity: Severity = Severity.WARN
    # id of the wall/room/opening this concerns, when applicable
    ref: str | None = None


class SourceImage(Base):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str | None = None


class Provenance(Base):
    recognizer: str
    recognizer_version: str = Field(alias="recognizerVersion")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), alias="generatedAt"
    )
    duration_ms: float | None = Field(default=None, alias="durationMs")


class FloorPlanIR(Base):
    """The complete intermediate representation of one floor plan.

    This is the "shared scene data package" from the platform proposal: the web
    configurator edits it, the extruder turns it into geometry, and the VR
    client reads from it. Everything downstream depends on this shape.
    """

    schema_version: Literal["1.0.0"] = Field(
        default=SCHEMA_VERSION, alias="schemaVersion"
    )
    id: str = Field(default_factory=lambda: _uid("plan"))
    name: str = "Untitled Plan"
    source: SourceImage | None = None
    scale: Scale
    levels: Annotated[list[Level], Field(min_length=1)]
    provenance: Provenance | None = None
    warnings: list[Warning] = Field(default_factory=list)

    # -- convenience -------------------------------------------------------

    def wall_by_id(self, wall_id: str) -> Wall | None:
        for level in self.levels:
            for wall in level.walls:
                if wall.id == wall_id:
                    return wall
        return None

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> FloorPlanIR:
        """An opening pointing at a missing wall would otherwise crash the
        extruder deep inside a mesh boolean, where the error is unreadable.
        Catch it here at the boundary instead, where we can name the culprit."""
        for level in self.levels:
            walls_by_id = {w.id: w for w in level.walls}
            for opening in level.openings:
                wall = walls_by_id.get(opening.wall_id)
                if wall is None:
                    raise ValueError(
                        f"opening {opening.id} references unknown wall "
                        f"{opening.wall_id!r} on level {level.id}"
                    )
                if opening.offset + opening.width / 2 > wall.length + 1e-6:
                    raise ValueError(
                        f"opening {opening.id} (offset {opening.offset:.2f}m, "
                        f"width {opening.width:.2f}m) overruns wall "
                        f"{wall.id} of length {wall.length:.2f}m"
                    )
        return self
