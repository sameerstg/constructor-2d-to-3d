"""End-to-end and unit tests for the recognition and extrusion pipeline.

Synthetic plans with known ground truth do most of the work here. Testing a
recognizer against images whose true wall count and scale are known is the only
way to catch a regression that leaves output still looking plausible -- which is
the characteristic way a CV pipeline breaks.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from app.extrude.gltf import ExtrusionError, build_scene, export_glb
from app.ir import (
    FloorPlanIR,
    Level,
    Opening,
    OpeningType,
    Room,
    Scale,
    ScaleSource,
    Severity,
    Wall,
    WallType,
)
from app.main import app
from app.recognize.base import RecognizerError
from app.recognize.opencv_v1 import _binarize, detect_rooms
from app.recognize.registry import get_recognizer
from app.recognize.skeleton import cluster_coordinates, merge_collinear, snap_to_axes
from app.sources import SourceDocument
from app.sources import load as load_source
from app.synth import generate_plan


def _as_png(image: np.ndarray, name: str = "plan.png") -> SourceDocument:
    encoded = cv2.imencode(".png", image)[1].tobytes()
    return load_source(name, encoded)


@pytest.fixture(scope="module")
def plan_image() -> np.ndarray:
    image, _ = generate_plan(with_clutter=True)
    return image


@pytest.fixture(scope="module")
def plan_doc(plan_image: np.ndarray) -> SourceDocument:
    return _as_png(plan_image)


@pytest.fixture(scope="module")
def truth():
    _, ground_truth = generate_plan(with_clutter=True)
    return ground_truth


@pytest.fixture
def simple_ir() -> FloorPlanIR:
    """A 5m x 4m room with one door. Hand-built, so every number is known."""
    walls = [
        Wall(id="w_n", start=[0, 0], end=[5, 0], thickness=0.2, type=WallType.EXTERIOR),
        Wall(id="w_e", start=[5, 0], end=[5, 4], thickness=0.2, type=WallType.EXTERIOR),
        Wall(id="w_s", start=[5, 4], end=[0, 4], thickness=0.2, type=WallType.EXTERIOR),
        Wall(id="w_w", start=[0, 4], end=[0, 0], thickness=0.2, type=WallType.EXTERIOR),
    ]
    return FloorPlanIR(
        name="test",
        scale=Scale(pixels_per_meter=100, source=ScaleSource.MANUAL, confidence=1.0),
        levels=[
            Level(
                walls=walls,
                openings=[
                    Opening(id="o1", wall_id="w_n", type=OpeningType.DOOR, offset=2.5, width=0.9)
                ],
                rooms=[Room(id="r1", polygon=[[0, 0], [5, 0], [5, 4], [0, 4]], area=20.0)],
            )
        ],
    )


# --------------------------------------------------------------------------
# IR validation
# --------------------------------------------------------------------------


def test_opening_on_missing_wall_is_rejected():
    """Referential integrity is checked at the boundary, not deep in the mesh code."""
    with pytest.raises(ValueError, match="unknown wall"):
        FloorPlanIR(
            scale=Scale(pixels_per_meter=100),
            levels=[
                Level(
                    walls=[Wall(id="w1", start=[0, 0], end=[3, 0])],
                    openings=[Opening(wall_id="nope", offset=1, width=0.9)],
                )
            ],
        )


def test_opening_overrunning_its_wall_is_rejected():
    with pytest.raises(ValueError, match="overruns wall"):
        FloorPlanIR(
            scale=Scale(pixels_per_meter=100),
            levels=[
                Level(
                    walls=[Wall(id="w1", start=[0, 0], end=[2, 0])],
                    openings=[Opening(wall_id="w1", offset=1.9, width=0.9)],
                )
            ],
        )


def test_zero_length_wall_is_rejected():
    with pytest.raises(ValueError, match="zero-length"):
        Wall(start=[1, 1], end=[1, 1])


def test_ir_round_trips_through_camel_case_json(simple_ir: FloorPlanIR):
    """The web app sees camelCase; Python uses snake_case. Both must reload."""
    payload = simple_ir.model_dump(mode="json", by_alias=True)
    assert "wallHeight" in payload["levels"][0]
    assert payload["levels"][0]["openings"][0]["wallId"] == "w_n"
    assert FloorPlanIR.model_validate(payload).levels[0].walls[0].id == "w_n"


# --------------------------------------------------------------------------
# Skeleton vectorization
# --------------------------------------------------------------------------


def test_near_axis_segments_snap_flat():
    # 3 degrees off horizontal over 100 units.
    segments = [((0.0, 0.0), (100.0, 5.0))]
    (a, b), *_ = snap_to_axes(segments)
    assert a[1] == b[1] == pytest.approx(2.5)


def test_genuinely_diagonal_segments_survive_snapping():
    segments = [((0.0, 0.0), (100.0, 100.0))]
    (a, b), *_ = snap_to_axes(segments, angle_tolerance_deg=8.0)
    assert (a, b) == ((0.0, 0.0), (100.0, 100.0))


def test_nearby_coordinates_cluster_to_one_line():
    """Two walls meant to be collinear must end up exactly collinear, or the
    extruded mesh is left with hairline gaps at the junction."""
    segments = [((0.0, 100.0), (50.0, 100.0)), ((60.0, 103.0), (110.0, 103.0))]
    clustered = cluster_coordinates(segments, tolerance=5.0)
    assert clustered[0][0][1] == clustered[1][0][1]


def test_collinear_touching_segments_merge():
    segments = [((0.0, 10.0), (50.0, 10.0)), ((50.0, 10.0), (120.0, 10.0))]
    merged = merge_collinear(segments, tolerance=2.0)
    assert len(merged) == 1
    assert merged[0] == ((0.0, 10.0), (120.0, 10.0))


def test_segments_separated_by_a_doorway_do_not_merge():
    """A door-sized gap is real geometry and must not be closed by the merge."""
    segments = [((0.0, 10.0), (50.0, 10.0)), ((130.0, 10.0), (200.0, 10.0))]
    merged = merge_collinear(segments, tolerance=6.0)
    assert len(merged) == 2


# --------------------------------------------------------------------------
# Recognizer
# --------------------------------------------------------------------------


def test_recognizer_finds_walls_and_rooms(plan_doc):
    ir = get_recognizer("opencv").recognize(plan_doc)
    level = ir.levels[0]

    assert len(level.walls) >= 8, "should find the envelope plus the internal partitions"
    assert len(level.rooms) >= 3
    assert ir.provenance is not None
    assert ir.provenance.recognizer == "opencv"


def test_recognizer_ignores_furniture_and_text(plan_doc):
    """Clutter is thin; walls are thick. Detection must not be inflated by it."""
    clean, _ = generate_plan(with_clutter=False)
    engine = get_recognizer("opencv")

    cluttered_walls = len(engine.recognize(plan_doc).levels[0].walls)
    clean_walls = len(engine.recognize(_as_png(clean)).levels[0].walls)

    assert cluttered_walls == clean_walls


def test_recognizer_recovers_proportions(plan_doc, truth):
    """Absolute scale is a guess, but SHAPE should be close to ground truth."""
    ir = get_recognizer("opencv").recognize(plan_doc)
    detected = sum(w.length for w in ir.levels[0].walls)

    # Normalise out the known scale error to compare proportions alone.
    corrected = detected * (ir.scale.pixels_per_meter / truth.pixels_per_meter)
    assert 0.75 <= corrected / truth.total_wall_length_m <= 1.25


def test_supplied_scale_is_authoritative(plan_doc):
    ir = get_recognizer("opencv").recognize(plan_doc, pixels_per_meter=80.0)
    assert ir.scale.source is ScaleSource.MANUAL
    assert ir.scale.pixels_per_meter == 80.0
    assert ir.scale.confidence == 1.0
    # A supplied scale is certain, so the "this is a guess" warning must go.
    assert not any(w.code == "scale_is_a_guess" for w in ir.warnings)


def test_guessed_scale_is_flagged_loudly(plan_doc):
    """A silently wrong scale is the worst failure mode: everything downstream
    looks right and is the wrong size. It must always be reported."""
    ir = get_recognizer("opencv").recognize(plan_doc)
    assert ir.scale.source is ScaleSource.HEURISTIC
    assert ir.scale.confidence < 0.5
    assert any(w.code == "scale_is_a_guess" for w in ir.warnings)


def test_blank_image_is_rejected_not_guessed_at():
    blank = np.full((600, 800, 3), 255, dtype=np.uint8)
    with pytest.raises(RecognizerError):
        get_recognizer("opencv").recognize(_as_png(blank, "blank.png"))


def test_unknown_recognizer_names_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        get_recognizer("does-not-exist")


# --------------------------------------------------------------------------
# Extrusion
# --------------------------------------------------------------------------


def test_walls_and_floors_become_named_meshes(simple_ir: FloorPlanIR):
    scene = build_scene(simple_ir)
    names = set(scene.geometry)
    # Names are the addressing scheme the configurator uses to retarget
    # materials, so they are part of the contract.
    assert "wall_w_n" in names
    assert "floor_r1" in names


def test_extruded_geometry_is_watertight(simple_ir: FloorPlanIR):
    """VR needs solids. A non-watertight wall renders with holes from inside."""
    scene = build_scene(simple_ir)
    for name, mesh in scene.geometry.items():
        assert mesh.is_watertight, f"{name} is not watertight"


def test_plan_maps_to_world_with_y_up(simple_ir: FloorPlanIR):
    scene = build_scene(simple_ir)
    bounds = scene.bounds
    # 5m x 4m plan, 2.7m walls: X spans 5, Z spans 4, Y is height.
    assert bounds[1][0] - bounds[0][0] == pytest.approx(5.2, abs=0.01)  # +thickness
    assert bounds[1][2] - bounds[0][2] == pytest.approx(4.2, abs=0.01)
    assert bounds[1][1] == pytest.approx(2.7, abs=0.01)


def test_a_door_actually_removes_material(simple_ir: FloorPlanIR):
    """The boolean must cut, not just be attempted."""
    with_door = build_scene(simple_ir).geometry["wall_w_n"].volume

    no_openings = simple_ir.model_copy(deep=True)
    no_openings.levels[0].openings = []
    solid = build_scene(no_openings).geometry["wall_w_n"].volume

    door = simple_ir.levels[0].openings[0]
    expected = door.width * door.height * 0.2
    assert solid - with_door == pytest.approx(expected, rel=0.05)


def test_export_produces_a_valid_glb(simple_ir: FloorPlanIR):
    glb = export_glb(simple_ir)
    assert glb[:4] == b"glTF"
    reloaded = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")
    assert len(reloaded.geometry) == len(simple_ir.levels[0].walls) + 1


def test_extruding_a_missing_level_fails_clearly(simple_ir: FloorPlanIR):
    with pytest.raises(ExtrusionError, match="does not exist"):
        build_scene(simple_ir, level_index=7)


def test_wall_yaw_matches_plan_direction():
    """Guards the plan(x, y) -> world(x, elevation, y) mapping. Getting this
    wrong mirrors the whole building, which is easy to miss on a symmetric plan."""
    from app.extrude.gltf import _wall_frame

    # A wall running +y in plan space should run +z in world space.
    wall = Wall(start=[0, 0], end=[0, 3])
    length, yaw, unit, _ = _wall_frame(wall)
    assert length == pytest.approx(3.0)
    assert unit == pytest.approx((0.0, 1.0))
    assert math.cos(yaw) == pytest.approx(0.0, abs=1e-9)
    assert -math.sin(yaw) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_lists_recognizers(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert any(r["key"] == "opencv" for r in body["recognizers"])


def test_recognize_endpoint_returns_camel_case_ir(client: TestClient, plan_image):
    encoded = cv2.imencode(".png", plan_image)[1].tobytes()
    response = client.post(
        "/recognize", files={"file": ("plan.png", encoded, "image/png")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["schemaVersion"] == "1.0.0"
    assert "wallHeight" in body["levels"][0]
    assert body["source"]["filename"] == "plan.png"


def test_recognize_rejects_an_unrecognised_file(client: TestClient):
    """415, not 400: the upload is intact, we just have no reader for it. The
    message lists what we DO accept, which is the only actionable thing."""
    response = client.post(
        "/recognize", files={"file": ("notes.txt", b"this is not a png", "text/plain")}
    )
    assert response.status_code == 415
    assert "DXF" in response.json()["detail"]


def test_recognize_rejects_an_unusable_image(client: TestClient):
    blank = np.full((400, 400, 3), 255, dtype=np.uint8)
    encoded = cv2.imencode(".png", blank)[1].tobytes()
    response = client.post(
        "/recognize", files={"file": ("blank.png", encoded, "image/png")}
    )
    # 422: the file decoded fine, it just is not a floor plan.
    assert response.status_code == 422


def test_extrude_endpoint_returns_glb(client: TestClient, simple_ir: FloorPlanIR):
    response = client.post(
        "/extrude", json=simple_ir.model_dump(mode="json", by_alias=True)
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"
    assert response.content[:4] == b"glTF"


def test_extrude_rejects_a_malformed_ir(client: TestClient):
    response = client.post("/extrude", json={"nonsense": True})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Robustness on real-world drawings
#
# These lock in fixes for failures found by running real floor plans through
# the pipeline. Each one returned plausible-looking output while being badly
# wrong, which is the failure mode synthetic fixtures do not catch.
# --------------------------------------------------------------------------


def _room_mask(width=400, height=300, thickness=8, door_px=36):
    """Two rooms separated by a wall with a doorway gap.

    `door_px` is a realistic door at the scale the test passes in: 36px at
    40 px/m is 0.9m. A wider gap is correctly read as an open-plan threshold
    rather than a door, and is deliberately NOT bridged.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (width - 20, height - 20), 255, thickness)
    mid, half = width // 2, door_px // 2
    cv2.line(mask, (mid, 20), (mid, height // 2 - half), 255, thickness)
    cv2.line(mask, (mid, height // 2 + half), (mid, height - 20), 255, thickness)
    return mask


def test_room_detection_survives_a_wrong_scale_estimate():
    """Room thresholds must not be hostage to the scale guess.

    Deriving the minimum room area from pixels-per-metre scales the error by its
    SQUARE, so a 3x scale over-estimate discards every real room. Real plans
    returned zero rooms for exactly this reason while synthetics looked fine.
    """
    mask = _room_mask()
    half_width = 4.0

    honest = detect_rooms(mask, 40.0, wall_half_width_px=half_width)
    # Same drawing, same measured thickness, wildly wrong metre scale.
    overestimated = detect_rooms(mask, 400.0, wall_half_width_px=half_width)

    assert len(honest) >= 2
    assert len(overestimated) == len(honest)


def test_metre_thresholds_still_used_when_scale_is_exact():
    """DXF states its units, so metres are the better unit there."""
    rooms = detect_rooms(_room_mask(), 40.0)
    assert len(rooms) >= 2


def test_hairline_linework_is_reported_not_silently_guessed():
    """An unfilled CAD plot draws walls as two hairlines with nothing between.

    Wall thickness is then unmeasurable, so scale and geometry are built on a
    number at its floor. The recognizer may still emit something, but it must
    say the output is untrustworthy and point at the DXF.
    """
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    # Hollow double-line walls: an outline, not a filled stroke.
    for inset in (0, 6):
        cv2.rectangle(image, (100 + inset, 100 + inset), (1100 - inset, 800 - inset), (0, 0, 0), 1)
    cv2.line(image, (600, 100), (600, 800), (0, 0, 0), 1)
    cv2.line(image, (606, 100), (606, 800), (0, 0, 0), 1)

    ir = get_recognizer("opencv").recognize(_as_png(image, "hollow.png"))
    codes = [w.code for w in ir.warnings]
    assert "hairline_linework" in codes

    warning = next(w for w in ir.warnings if w.code == "hairline_linework")
    assert warning.severity is Severity.ERROR
    assert "DXF" in warning.message


def test_grey_linework_is_not_thrown_away_with_the_paper():
    """Otsu assumes a bimodal histogram; a real plan is trimodal.

    White paper, mid-grey secondary linework, black primary linework. Otsu
    splits the extremes and discards the middle — which on a real drawing
    deleted the light-grey OUTER walls, leaving gaps far wider than a doorway so
    the whole interior merged with the outdoors and every large room vanished.
    """
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    grey = (165, 165, 165)
    black = (0, 0, 0)

    # Envelope drawn in grey, interior partition in black — the exact
    # combination that used to lose the envelope.
    cv2.rectangle(image, (80, 80), (720, 520), grey, 14)
    cv2.line(image, (400, 80), (400, 260), black, 14)
    cv2.line(image, (400, 340), (400, 520), black, 14)

    ink = _binarize(image)

    # The grey envelope must survive thresholding...
    assert ink[80, 400] > 0, "grey exterior wall was thresholded away"
    # ...and with it closed, the interior must resolve into rooms rather than
    # leaking into the exterior component.
    ir = get_recognizer("opencv").recognize(_as_png(image, "grey.png"))
    assert len(ir.levels[0].rooms) >= 2


def test_solid_black_poche_is_not_mistaken_for_paper():
    """The paper level is found in the upper half of the histogram on purpose:
    a plan with heavy filled walls must not have the poche taken for the page."""
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (740, 540), (0, 0, 0), 60)

    ink = _binarize(image)
    assert ink[60, 400] > 0          # the wall is ink
    assert ink[300, 400] == 0        # the room interior is not
