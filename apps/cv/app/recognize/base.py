"""The recognizer interface.

Everything that turns a floor plan image into IR implements this one Protocol.
That is the whole point: the proposal commits to trialing and finalizing an AI
recognition model during Phase 1, so the recognizer has to be a swappable part
from day one. Classical CV, a segmentation network, and a paid third-party API
all sit behind this same signature, and the rest of the system cannot tell them
apart.

Registering a new recognizer is a two-line change in `app.recognize.registry`;
nothing in the API layer, the editor, or the extruder needs to know about it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.ir import FloorPlanIR


class RecognizerError(RuntimeError):
    """Raised when an image cannot be interpreted as a floor plan at all.

    Distinct from low-confidence output: a recognizer that produced a poor but
    plausible result should return it WITH warnings so a human can fix it in the
    review editor. Raise this only when there is nothing usable to hand over.
    """


@runtime_checkable
class Recognizer(Protocol):
    """Image in, IR out."""

    name: str
    version: str

    def recognize(
        self,
        image: np.ndarray,
        *,
        filename: str = "upload",
        pixels_per_meter: float | None = None,
    ) -> FloorPlanIR:
        """Convert a floor plan raster into the intermediate representation.

        Args:
            image: BGR or grayscale image as read by OpenCV.
            filename: Original filename, recorded in IR provenance.
            pixels_per_meter: Caller-supplied scale override. When provided this
                is authoritative (ScaleSource.MANUAL) and the recognizer must
                not try to infer scale itself. A human who has measured the
                drawing always beats our guess.

        Returns:
            A valid FloorPlanIR. May be imperfect, but must be structurally
            valid and should carry warnings describing its own uncertainty.

        Raises:
            RecognizerError: if the input is not usable as a floor plan.
        """
        ...
