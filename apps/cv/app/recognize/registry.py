"""Recognizer registry and routing.

Adding a recognizer -- a segmentation model, an IFC reader, a vendor API
wrapper -- means importing it and adding one line to `_RECOGNIZERS`. Nothing
else in the system changes. The API exposes the keys here as the `recognizer`
request parameter, which makes an honest A/B against real client floor plans a
one-parameter experiment rather than a branch.

Uploads are routed by their detected kind, so a DXF and a scan of the same
building both go to the right place without the caller choosing.
"""

from __future__ import annotations

from app.recognize.base import Recognizer
from app.recognize.dxf_v1 import DxfRecognizer
from app.recognize.opencv_v1 import OpenCVRecognizer
from app.sources import CONVERSION_ADVICE, SourceDocument, SourceKind


class UnsupportedSourceError(Exception):
    """Raised for a file we cannot read but can advise about."""

    def __init__(self, kind: SourceKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


_RECOGNIZERS: dict[str, Recognizer] = {
    "opencv": OpenCVRecognizer(),
    "dxf": DxfRecognizer(),
}

#: Preferred recognizer per source kind. A kind absent here has no reader.
_ROUTES: dict[SourceKind, str] = {
    SourceKind.RASTER: "opencv",
    SourceKind.DXF: "dxf",
}


def get_recognizer(name: str | None = None) -> Recognizer:
    if name is None:
        raise KeyError("No recognizer specified and no document to route.")
    try:
        return _RECOGNIZERS[name]
    except KeyError:
        available = ", ".join(sorted(_RECOGNIZERS))
        raise KeyError(f"Unknown recognizer {name!r}. Available: {available}") from None


def route(document: SourceDocument, name: str | None = None) -> Recognizer:
    """Pick the recognizer for a document, or explain why we cannot read it.

    An explicit `name` wins, but is still checked against what that recognizer
    accepts -- sending a DXF to the raster pipeline produces a confusing decode
    failure rather than a useful message.
    """
    if name is not None:
        recognizer = get_recognizer(name)
        if document.kind not in recognizer.accepts:
            accepted = ", ".join(sorted(recognizer.accepts))
            raise UnsupportedSourceError(
                document.kind,
                f"Recognizer {name!r} accepts {accepted}, but this file was "
                f"detected as {document.kind}.",
            )
        return recognizer

    key = _ROUTES.get(document.kind)
    if key is None:
        raise UnsupportedSourceError(
            document.kind,
            CONVERSION_ADVICE.get(
                document.kind, CONVERSION_ADVICE[SourceKind.UNKNOWN]
            ),
        )
    return _RECOGNIZERS[key]


def available_recognizers() -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "name": r.name,
            "version": r.version,
            "accepts": sorted(r.accepts),
        }
        for key, r in sorted(_RECOGNIZERS.items())
    ]


def supported_kinds() -> list[str]:
    return sorted(_ROUTES)
