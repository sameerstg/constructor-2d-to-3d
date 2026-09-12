"""Score a recognizer against ground truth.

The single most useful thing to have before choosing a recognition model. Once
this exists, "should we license an AI recognition API?" stops being a matter of
opinion and becomes a table of numbers on the client's own floor plans.

    python -m app.evaluate --samples ../../samples
    python -m app.evaluate --samples ../../samples --recognizer opencv --glb out/

Every supported source kind is scored, DXF included, so the vector and raster
paths can be compared on the same building rather than argued about.

Metrics are deliberately blunt. Wall-length ratio and room count catch the
failures that actually matter -- missing half the building, or leaking every
room into one -- without pretending to a precision the inputs do not support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.extrude.gltf import ExtrusionError, export_glb
from app.recognize.base import RecognizerError
from app.recognize.registry import UnsupportedSourceError, route
from app.sources import load as load_source

#: Everything the pipeline can read. DXF is scored alongside the rasters
#: deliberately: the whole build-vs-buy question is "how much better is the
#: vector path", and that needs both in one table.
SCORABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".dxf"}

#: Ground-truth manifests, merged. Each generator writes its own.
TRUTH_FILES = ("ground_truth.json", "ground_truth_dxf.json", "ground_truth_cad_raster.json")


def evaluate_one(
    image_path: Path, truth: dict | None, recognizer_key: str | None, glb_dir: Path | None
) -> dict:
    try:
        document = load_source(image_path.name, image_path.read_bytes())
    except OSError as exc:
        return {"file": image_path.name, "status": "unreadable", "reason": str(exc)}

    try:
        engine = route(document, recognizer_key)
    except UnsupportedSourceError as exc:
        return {"file": image_path.name, "status": "unsupported", "reason": str(exc)}

    try:
        ir = engine.recognize(document)
    except RecognizerError as exc:
        return {"file": image_path.name, "status": "rejected", "reason": str(exc)}

    level = ir.levels[0]
    wall_length = sum(w.length for w in level.walls)

    result: dict = {
        "file": image_path.name,
        "status": "ok",
        "kind": str(document.kind),
        "recognizer": engine.name,
        "walls": len(level.walls),
        "rooms": len(level.rooms),
        "openings": len(level.openings),
        "wall_length_m": round(wall_length, 2),
        "scale_ppm": round(ir.scale.pixels_per_meter, 2),
        "scale_source": ir.scale.source.value,
        "duration_ms": ir.provenance.duration_ms if ir.provenance else None,
        "warnings": [w.code for w in ir.warnings],
    }

    if truth:
        expected_length = truth.get("total_wall_length_m")
        expected_rooms = truth.get("room_count")
        expected_ppm = truth.get("pixels_per_meter")
        result["truth"] = {
            "rooms": expected_rooms,
            "wall_length_m": expected_length,
            "scale_ppm": expected_ppm,
        }
        result["scores"] = {
            # 1.0 is perfect; below 1 means under-detection, above means noise.
            "wall_length_ratio": round(wall_length / expected_length, 3)
            if expected_length
            else None,
            "room_recall": round(len(level.rooms) / expected_rooms, 3)
            if expected_rooms
            else None,
            "scale_error_pct": round(
                abs(ir.scale.pixels_per_meter - expected_ppm) / expected_ppm * 100, 1
            )
            if expected_ppm
            else None,
        }

    if glb_dir is not None:
        glb_dir.mkdir(parents=True, exist_ok=True)
        try:
            glb = export_glb(ir)
            out = glb_dir / f"{image_path.stem}.glb"
            out.write_bytes(glb)
            result["glb"] = {"path": str(out), "bytes": len(glb)}
        except ExtrusionError as exc:
            result["glb"] = {"error": str(exc)}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a recognizer against ground truth.")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument(
        "--recognizer",
        default=None,
        help="Force a recognizer. Default: route each file by its detected kind.",
    )
    parser.add_argument("--glb", type=Path, default=None, help="Also write GLB files here")
    args = parser.parse_args()

    truths: dict[str, dict] = {}
    for name in TRUTH_FILES:
        path = args.samples / name
        if path.exists():
            truths.update(json.loads(path.read_text()))

    images = sorted(
        p
        for p in args.samples.iterdir()
        if p.is_file() and p.suffix.lower() in SCORABLE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No images found in {args.samples}")

    results = [
        evaluate_one(path, truths.get(path.name), args.recognizer, args.glb) for path in images
    ]

    header = (
        f"{'file':<46}{'via':<8}{'walls':>6}{'rooms':>6}{'open':>6}"
        f"{'len m':>9}{'len ratio':>11}{'ppm err':>9}{'ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "ok":
            reason = (r.get("reason", "") or "").split(".")[0]
            print(f"{r['file']:<46}{r['status'].upper()}: {reason}")
            continue
        scores = r.get("scores", {})
        print(
            f"{r['file']:<46}"
            f"{r['recognizer']:<8}"
            f"{r['walls']:>6}"
            f"{r['rooms']:>6}"
            f"{r['openings']:>6}"
            f"{r['wall_length_m']:>9.1f}"
            f"{_fmt(scores.get('wall_length_ratio')):>11}"
            f"{_fmt(scores.get('scale_error_pct'), '%'):>9}"
            f"{r['duration_ms'] or 0:>8.0f}"
        )

    print()
    print(json.dumps(results, indent=2))


def _fmt(value: float | None, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.2f}{suffix}"


if __name__ == "__main__":
    main()
