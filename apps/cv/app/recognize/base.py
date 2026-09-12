"""The recognizer interface.

Everything that turns a floor plan into IR implements this one Protocol. That is
the whole point: the proposal commits to trialing and finalizing an AI
recognition model during Phase 1, so the recognizer has to be a swappable part
from day one. Classical CV on a scan, a vector CAD parser, a segmentation
network and a paid third-party API all sit behind this same signature, and the
rest of the system cannot tell them apart.

Recognizers take a `SourceDocument`, not a decoded image, because the input is
not always a raster. A DXF exported from AutoCAD or Revit carries layers, real
units and block references; handing a recognizer pre-rasterized pixels would
throw all of that away before it could be used. A raster recognizer simply
decodes the bytes itself.

Registering a new recognizer is a two-line change in `app.recognize.registry`;
nothing in the API layer, the editor, or the extruder needs to know about it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.ir import FloorPlanIR
from app.sources import SourceDocument, SourceKind


class RecognizerError(RuntimeError):
    """Raised when a document cannot be interpreted as a floor plan at all.

    Distinct from low-confidence output: a recognizer that produced a poor but
    plausible result should return it WITH warnings so a human can fix it in the
    review editor. Raise this only when there is nothing usable to hand over.
    """


@runtime_checkable
class Recognizer(Protocol):
    """Document in, IR out."""

    name: str
    version: str
    #: Source kinds this recognizer can handle. Used to route uploads.
    accepts: frozenset[SourceKind]

    def recognize(
        self,
        document: SourceDocument,
        *,
        pixels_per_meter: float | None = None,
    ) -> FloorPlanIR:
        """Convert a floor plan document into the intermediate representation.

        Args:
            document: The uploaded file and its detected kind.
            pixels_per_meter: Caller-supplied scale override. When provided this
                is authoritative (ScaleSource.MANUAL) and the recognizer must
                not infer scale itself. A human who has measured the drawing
                always beats our guess. Ignored by recognizers whose input
                carries real units, such as DXF - a stated unit is already
                exact, and overriding it would only introduce error.

        Returns:
            A valid FloorPlanIR. May be imperfect, but must be structurally
            valid and should carry warnings describing its own uncertainty.

        Raises:
            RecognizerError: if the input is not usable as a floor plan.
        """
        ...
