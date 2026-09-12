"""Source documents: what a floor plan can arrive as.

The pipeline originally assumed a raster image. That assumption is wrong for
the drawings this platform will actually receive. A floor plan exported from
AutoCAD or Revit is **vector data with semantics** - layers named for their
trade, real world units, dimension entities, and block references for doors and
windows. Rasterizing it and running computer vision over the pixels throws away
precisely the information that makes the hard problem easy.

So the recognizer interface takes a SourceDocument rather than a numpy array,
and each recognizer declares which kinds it accepts. Raster CV remains the
fallback for scans and photographs, where there genuinely is nothing but pixels.

Format landscape
----------------
- **DXF** - the open interchange format. Both AutoCAD and Revit export it, and
  ezdxf (MIT) reads it. This is the one worth supporting properly.
- **DWG** - AutoCAD's proprietary native format. No permissively licensed
  reader exists; convert to DXF first.
- **RVT** - Revit's proprietary native format. Same story; export IFC or DXF.
- **IFC** - the open BIM standard, and the richest possible input: walls, doors
  and spaces are first-class objects with real dimensions. Needs IfcOpenShell
  (LGPL-3.0), so it belongs out-of-process.
- **PDF** - common delivery format. Vector PDFs retain line geometry.

See docs/CAD-INPUT.md for how to get from each of these to something we read.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Magic bytes, checked before file extensions. Extensions lie: plenty of
# "floorplan.dxf" files are actually DWG saved with the wrong suffix, and the
# resulting parse error is far more confusing than an honest rejection.
_DWG_MAGIC = b"AC10"  # AC1012, AC1015, AC1018, AC1021, AC1024, AC1027, AC1032
_PDF_MAGIC = b"%PDF"
_RVT_MAGIC = b"\xd0\xcf\x11\xe0"  # OLE2 compound document

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class SourceKind(StrEnum):
    RASTER = "raster"
    DXF = "dxf"
    IFC = "ifc"
    PDF = "pdf"

    # Recognised so we can refuse them with a useful message rather than a
    # generic decode failure. Both are proprietary and have no permissively
    # licensed reader; the fix is always "export an interchange format".
    DWG = "dwg"
    RVT = "rvt"

    UNKNOWN = "unknown"


#: What to tell a user who uploads a format we cannot read directly.
CONVERSION_ADVICE: dict[SourceKind, str] = {
    SourceKind.DWG: (
        "DWG is AutoCAD's proprietary format and cannot be read directly. "
        "In AutoCAD use SAVEAS and choose DXF, or run the free ODA File "
        "Converter over the folder. DXF keeps the units, layers and blocks "
        "this pipeline relies on."
    ),
    SourceKind.RVT: (
        "RVT is Revit's proprietary format and cannot be read directly. "
        "From Revit, export the plan view as DXF (File > Export > CAD Formats), "
        "or export IFC for full building semantics."
    ),
    SourceKind.UNKNOWN: (
        "Unrecognised file type. Supported: DXF, and raster images "
        "(PNG, JPEG, WebP, BMP, TIFF)."
    ),
}


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """An uploaded floor plan, before anything has been made of it."""

    filename: str
    data: bytes
    kind: SourceKind

    @property
    def stem(self) -> str:
        return Path(self.filename).stem or "Untitled Plan"

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def detect_kind(filename: str, data: bytes) -> SourceKind:
    """Identify a floor plan file from its contents, falling back to its name.

    Content first, extension second. DXF is the awkward case: it is plain text
    with no magic number, so it is identified by its section structure instead.
    """
    head = data[:1024]

    if head.startswith(_PDF_MAGIC):
        return SourceKind.PDF
    if head.startswith(_DWG_MAGIC):
        return SourceKind.DWG
    if head.startswith(_RVT_MAGIC):
        # Revit files are OLE2 containers. So are .doc and .xls, but nobody is
        # uploading those to a floor plan tool, and the advice we give for a
        # misidentified one is still broadly correct.
        return SourceKind.RVT

    # IFC is a STEP physical file; the header is ISO-10303-21.
    if b"ISO-10303-21" in head:
        return SourceKind.IFC

    # DXF has no magic bytes. Every DXF begins with a SECTION group, and the
    # HEADER section follows almost immediately. Binary DXF starts with a
    # sentinel string instead.
    if head.startswith(b"AutoCAD Binary DXF"):
        return SourceKind.DXF
    probe = head.lstrip()[:512].upper()
    if b"SECTION" in probe and (b"HEADER" in probe or b"ENTITIES" in probe):
        return SourceKind.DXF

    suffix = Path(filename).suffix.lower()
    if suffix in _RASTER_SUFFIXES:
        return SourceKind.RASTER
    if suffix == ".dxf":
        return SourceKind.DXF
    if suffix == ".dwg":
        return SourceKind.DWG
    if suffix == ".rvt":
        return SourceKind.RVT
    if suffix in {".ifc", ".ifczip"}:
        return SourceKind.IFC

    # Raster formats all have magic bytes, but there are many of them and
    # OpenCV will tell us definitively when it tries to decode.
    if data[:2] in {b"\x89P", b"\xff\xd8", b"BM", b"II", b"MM"} or head[:4] == b"RIFF":
        return SourceKind.RASTER

    return SourceKind.UNKNOWN


def load(filename: str, data: bytes) -> SourceDocument:
    return SourceDocument(filename=filename, data=data, kind=detect_kind(filename, data))
