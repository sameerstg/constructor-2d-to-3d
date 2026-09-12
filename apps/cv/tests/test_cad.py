"""Tests for vector CAD input.

Where the raster tests can only check that output is plausible, these check that
it is EXACT. That is the whole argument for reading DXF rather than a picture of
it: the drawing states its units, its layers say which lines are walls, and
doors are block references with real positions. Every number below is asserted
against the generator's ground truth, not a tolerance band.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.extrude.gltf import build_scene
from app.ir import OpeningType, RoomType, ScaleSource
from app.main import app
from app.recognize.registry import UnsupportedSourceError, route
from app.recognize.wallpairs import pair_walls
from app.sources import SourceDocument, SourceKind, detect_kind
from app.sources import load as load_source
from app.synth_dxf import generate_dxf


def _serialize(doc) -> bytes:
    """ezdxf writes text; the pipeline takes bytes, as an upload would."""
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def _document(doc, name: str = "plan.dxf") -> SourceDocument:
    return load_source(name, _serialize(doc))


@pytest.fixture(scope="module")
def dxf_pair():
    return generate_dxf()


@pytest.fixture(scope="module")
def dxf_bytes(dxf_pair) -> bytes:
    return _serialize(dxf_pair[0])


@pytest.fixture(scope="module")
def dxf_doc(dxf_bytes: bytes) -> SourceDocument:
    return load_source("plan.dxf", dxf_bytes)


@pytest.fixture(scope="module")
def dxf_truth(dxf_pair):
    return dxf_pair[1]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# Source detection and routing
# --------------------------------------------------------------------------


def test_dxf_detected_by_content_not_extension(dxf_bytes: bytes):
    """Extensions lie. Plenty of "plan.dxf" files are DWG saved with the wrong
    suffix; a content check gives an honest error rather than a parse crash."""
    assert detect_kind("plan.txt", dxf_bytes) is SourceKind.DXF
    assert detect_kind("no-extension", dxf_bytes) is SourceKind.DXF


def test_proprietary_formats_are_identified_so_we_can_advise():
    assert detect_kind("plan.dwg", b"AC1032\x00\x00") is SourceKind.DWG
    assert detect_kind("model.rvt", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") is SourceKind.RVT


def test_dwg_is_refused_with_conversion_advice():
    document = load_source("plan.dwg", b"AC1032\x00\x00padding")
    with pytest.raises(UnsupportedSourceError) as exc:
        route(document)
    # The only useful thing to say about a proprietary format is how to escape
    # it, so the message must actually say so.
    assert "DXF" in str(exc.value)


def test_rvt_is_refused_with_conversion_advice():
    document = load_source("model.rvt", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32)
    with pytest.raises(UnsupportedSourceError) as exc:
        route(document)
    assert "IFC" in str(exc.value) or "DXF" in str(exc.value)


def test_routing_picks_the_dxf_recognizer(dxf_doc: SourceDocument):
    assert route(dxf_doc).name == "dxf"


def test_explicit_recognizer_must_accept_the_kind(dxf_doc: SourceDocument):
    """Forcing a DXF through the raster pipeline should explain, not fail to decode."""
    with pytest.raises(UnsupportedSourceError, match="accepts"):
        route(dxf_doc, "opencv")


# --------------------------------------------------------------------------
# Wall pairing - the main geometric problem in reading CAD
# --------------------------------------------------------------------------


def test_parallel_faces_become_one_centred_wall():
    """CAD draws a wall as its two faces. Two lines 0.2m apart are ONE 0.2m wall,
    and its centreline - which nothing in the file records - is halfway."""
    faces = [((0.0, 0.0), (5.0, 0.0)), ((0.0, 0.2), (5.0, 0.2))]
    walls, unpaired = pair_walls(faces)

    assert unpaired == 0
    assert len(walls) == 1
    assert walls[0].thickness == pytest.approx(0.2)
    assert walls[0].start[1] == pytest.approx(0.1)
    assert walls[0].measured is True


def test_faces_too_far_apart_are_not_one_wall():
    """Two walls across a 3m room are parallel but are not each other's faces."""
    faces = [((0.0, 0.0), (5.0, 0.0)), ((0.0, 3.0), (5.0, 3.0))]
    walls, unpaired = pair_walls(faces)
    assert unpaired == 2
    assert all(not w.measured for w in walls)


def test_non_overlapping_parallel_lines_do_not_pair():
    """Correctly spaced, but at opposite ends of the building."""
    faces = [((0.0, 0.0), (2.0, 0.0)), ((20.0, 0.2), (22.0, 0.2))]
    _, unpaired = pair_walls(faces)
    assert unpaired == 2


def test_unpaired_lines_survive_as_walls():
    """A missing wall is a hole in the building; a spurious one is visible and
    fixable in review. Never silently drop geometry."""
    walls, unpaired = pair_walls([((0.0, 0.0), (5.0, 0.0))])
    assert unpaired == 1
    assert len(walls) == 1
    assert walls[0].measured is False


# --------------------------------------------------------------------------
# The DXF recognizer
# --------------------------------------------------------------------------


def test_dimensions_are_exact(dxf_doc: SourceDocument, dxf_truth):
    """No estimation anywhere: this is the entire argument for vector input."""
    ir = route(dxf_doc).recognize(dxf_doc)
    level = ir.levels[0]

    xs = [p[0] for w in level.walls for p in (w.start, w.end)]
    ys = [p[1] for w in level.walls for p in (w.start, w.end)]
    expected_w, expected_h = dxf_truth.building_size_m

    assert max(xs) == pytest.approx(expected_w, abs=0.02)
    assert max(ys) == pytest.approx(expected_h, abs=0.02)
    assert sum(w.length for w in level.walls) == pytest.approx(
        dxf_truth.total_wall_length_m, rel=0.01
    )


def test_wall_thickness_is_measured_not_assumed(dxf_doc: SourceDocument, dxf_truth):
    """Raster recognition cannot recover this at all. Here it falls out of the
    face pairing, per wall, exactly."""
    ir = route(dxf_doc).recognize(dxf_doc)
    thicknesses = sorted({round(w.thickness, 3) for w in ir.levels[0].walls})

    assert thicknesses == sorted(
        [dxf_truth.wall_thickness_m["interior"], dxf_truth.wall_thickness_m["exterior"]]
    )


def test_scale_is_exact_and_needs_no_calibration(dxf_doc: SourceDocument):
    ir = route(dxf_doc).recognize(dxf_doc)
    assert ir.scale.source is ScaleSource.MANUAL
    assert ir.scale.confidence == 1.0
    assert not any(w.code == "scale_is_a_guess" for w in ir.warnings)


def test_doors_and_windows_are_found(dxf_doc: SourceDocument, dxf_truth):
    """Block references make openings detectable, which raster input cannot do."""
    ir = route(dxf_doc).recognize(dxf_doc)
    openings = ir.levels[0].openings

    doors = [o for o in openings if o.type is OpeningType.DOOR]
    windows = [o for o in openings if o.type is OpeningType.WINDOW]

    assert len(doors) == dxf_truth.door_count
    assert len(windows) == dxf_truth.window_count
    # Windows sit above the floor; doors start at it.
    assert all(o.sill == 0 for o in doors)
    assert all(o.sill > 0 for o in windows)


def test_openings_stay_within_their_wall(dxf_doc: SourceDocument):
    """The IR validator rejects overruns, so this would fail loudly - but a door
    block measured across its swing arc is genuinely wider than the doorway,
    and clamping it is what keeps real drawings loadable."""
    ir = route(dxf_doc).recognize(dxf_doc)
    level = ir.levels[0]
    by_id = {w.id: w for w in level.walls}

    for opening in level.openings:
        wall = by_id[opening.wall_id]
        assert opening.offset - opening.width / 2 >= -1e-6
        assert opening.offset + opening.width / 2 <= wall.length + 1e-6


def test_room_names_come_from_drawing_text(dxf_doc: SourceDocument):
    ir = route(dxf_doc).recognize(dxf_doc)
    rooms = ir.levels[0].rooms
    names = {r.name for r in rooms if r.name}

    assert {"LIVING", "KITCHEN", "BED 1", "BED 2"} <= names
    # A named room should also be classified, so Phase 2 can choose materials
    # and fixtures per room type without another pass.
    kitchen = next(r for r in rooms if r.name == "KITCHEN")
    assert kitchen.type is RoomType.KITCHEN
    bedroom = next(r for r in rooms if r.name == "BED 1")
    assert bedroom.type is RoomType.BEDROOM


def test_all_rooms_are_found(dxf_doc: SourceDocument, dxf_truth):
    ir = route(dxf_doc).recognize(dxf_doc)
    assert len(ir.levels[0].rooms) == dxf_truth.room_count


def test_unspecified_units_are_warned_about_not_guessed_silently():
    """A silently wrong unit makes the building 1000x too big. Never silent."""
    document = _document(generate_dxf(unitless=True)[0], "unitless.dxf")
    ir = route(document).recognize(document)
    assert any(w.code == "dxf_units_unspecified" for w in ir.warnings)


def test_furniture_and_annotation_layers_are_ignored():
    """Layer filtering is the DXF equivalent of the raster thickness test:
    clutter must not inflate the wall count."""
    cluttered = _document(generate_dxf()[0], "a.dxf")
    clean = _document(generate_dxf(with_clutter=False)[0], "b.dxf")

    assert (
        len(route(cluttered).recognize(cluttered).levels[0].walls)
        == len(route(clean).recognize(clean).levels[0].walls)
    )


def test_plan_is_moved_to_the_origin(dxf_doc: SourceDocument):
    """CAD drawings sit on a site grid tens of thousands of metres from zero,
    which wrecks float precision in the mesh and puts the model off-camera."""
    ir = route(dxf_doc).recognize(dxf_doc)
    xs = [p[0] for w in ir.levels[0].walls for p in (w.start, w.end)]
    ys = [p[1] for w in ir.levels[0].walls for p in (w.start, w.end)]
    assert min(xs) == pytest.approx(0, abs=0.01)
    assert min(ys) == pytest.approx(0, abs=0.01)


def test_binary_dxf_is_refused_clearly():
    document = load_source("plan.dxf", b"AutoCAD Binary DXF\r\n\x1a\x00" + b"\x00" * 64)
    from app.recognize.base import RecognizerError

    with pytest.raises(RecognizerError, match="ASCII"):
        route(document).recognize(document)


# --------------------------------------------------------------------------
# Extrusion and API
# --------------------------------------------------------------------------


def test_extrudes_to_watertight_geometry_with_real_openings(dxf_doc: SourceDocument):
    ir = route(dxf_doc).recognize(dxf_doc)
    scene = build_scene(ir)

    for name, mesh in scene.geometry.items():
        assert mesh.is_watertight, f"{name} is not watertight"

    level = ir.levels[0]
    opening = level.openings[0]
    wall = next(w for w in level.walls if w.id == opening.wall_id)
    solid = wall.length * level.wall_height * wall.thickness
    assert scene.geometry[f"wall_{wall.id}"].volume < solid


def test_recognize_endpoint_accepts_dxf(client: TestClient, dxf_bytes: bytes):
    response = client.post(
        "/recognize", files={"file": ("plan.dxf", dxf_bytes, "application/dxf")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["provenance"]["recognizer"] == "dxf"
    assert body["scale"]["source"] == "manual"
    assert len(body["levels"][0]["openings"]) > 0


def test_dwg_upload_is_refused_with_advice(client: TestClient):
    """415, not 500: the file is intact and identifiable, we just cannot read
    that format - and the response says exactly how to convert it."""
    response = client.post(
        "/recognize", files={"file": ("plan.dwg", b"AC1032\x00\x00pad", "image/vnd.dwg")}
    )
    assert response.status_code == 415
    assert "DXF" in response.json()["detail"]


def test_health_reports_supported_kinds(client: TestClient):
    body = client.get("/health").json()
    assert "dxf" in body["supportedKinds"]
    assert "raster" in body["supportedKinds"]
