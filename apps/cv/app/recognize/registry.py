"""Recognizer registry.

Adding a recognizer -- a segmentation model, a vendor API wrapper -- means
importing it and adding one line to `_RECOGNIZERS`. Nothing else in the system
changes. The API exposes the keys here as the `recognizer` request parameter,
which is what makes an honest A/B against real client floor plans a
one-parameter experiment rather than a branch.
"""

from __future__ import annotations

from app.recognize.base import Recognizer
from app.recognize.opencv_v1 import OpenCVRecognizer

DEFAULT_RECOGNIZER = "opencv"

_RECOGNIZERS: dict[str, Recognizer] = {
    "opencv": OpenCVRecognizer(),
}


def get_recognizer(name: str | None = None) -> Recognizer:
    key = name or DEFAULT_RECOGNIZER
    try:
        return _RECOGNIZERS[key]
    except KeyError:
        available = ", ".join(sorted(_RECOGNIZERS))
        raise KeyError(f"Unknown recognizer {key!r}. Available: {available}") from None


def available_recognizers() -> list[dict[str, str]]:
    return [
        {"key": key, "name": r.name, "version": r.version}
        for key, r in sorted(_RECOGNIZERS.items())
    ]
