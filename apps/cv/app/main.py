"""FastAPI service: floor plan recognition and 3D extrusion.

Two endpoints carry the product:

    POST /recognize   image  -> IR      (the hard, fuzzy, AI-shaped half)
    POST /extrude     IR     -> GLB     (the easy, deterministic half)

Keeping them separate is the whole architecture in miniature. The IR sits
between them, so a human can edit the plan between recognition and geometry --
which is the "manual confirmation" stage the platform proposal builds its
quality guarantees on -- and so the recognizer can be swapped without the
extruder, the editor, or the VR client noticing.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.extrude.gltf import ExtrusionError, export_glb
from app.ir import FloorPlanIR
from app.recognize.base import RecognizerError
from app.recognize.registry import (
    UnsupportedSourceError,
    available_recognizers,
    route,
    supported_kinds,
)
from app.sources import load as load_source

logger = logging.getLogger(__name__)

# Generous but finite. A 50MP architectural scan is a legitimate input; a 2GB
# upload is not, and decoding one would take the service down.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

app = FastAPI(
    title="Constructor CV",
    version="0.1.0",
    summary="Floor plan recognition and 3D extrusion for the XR Home platform",
)

# The Next.js app calls this service directly in development. In production it
# should sit behind the web app's own API routes and not be publicly reachable.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "recognizers": available_recognizers(),
        "supportedKinds": supported_kinds(),
    }


@app.get("/schema")
def schema() -> dict:
    """The IR JSON Schema, generated from the Pydantic models."""
    return FloorPlanIR.model_json_schema(by_alias=True)


@app.post("/recognize", response_model=FloorPlanIR, response_model_by_alias=True)
async def recognize(
    file: UploadFile = File(
        ..., description="Floor plan: DXF, or a raster image (PNG, JPEG, WebP, BMP, TIFF)"
    ),
    recognizer: str | None = Form(
        default=None, description="Recognizer key; see GET /health. Omit to auto-route."
    ),
    pixels_per_meter: float | None = Form(
        default=None,
        description="Known scale for raster input. Ignored for DXF, which states its units.",
    ),
) -> FloorPlanIR:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw) / 1e6:.1f}MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f}MB.",
        )

    if pixels_per_meter is not None and pixels_per_meter <= 0:
        raise HTTPException(status_code=400, detail="pixels_per_meter must be positive.")

    document = load_source(file.filename or "upload", raw)

    try:
        engine = route(document, recognizer)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedSourceError as exc:
        # 415: the file is intact and identifiable, we just cannot read that
        # format. The message says how to convert it, which is the only useful
        # thing we can offer for a proprietary DWG or RVT.
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    try:
        return engine.recognize(document, pixels_per_meter=pixels_per_meter)
    except RecognizerError as exc:
        # The image decoded but is not usable as a floor plan. That is a problem
        # with the input, so report it as one and say what would help.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("recognizer %r crashed", recognizer)
        raise HTTPException(
            status_code=500, detail=f"Recognition failed unexpectedly: {exc}"
        ) from exc


@app.post(
    "/extrude",
    responses={200: {"content": {"model/gltf-binary": {}}, "description": "GLB model"}},
)
def extrude(ir: FloorPlanIR, level: int = 0) -> Response:
    """IR -> GLB. Deterministic: the same IR always produces the same bytes."""
    try:
        glb = export_glb(ir, level_index=level)
    except ExtrusionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("extrusion failed for plan %s", ir.id)
        raise HTTPException(status_code=500, detail=f"Extrusion failed: {exc}") from exc

    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f'attachment; filename="{ir.id}.glb"'},
    )
