"""End-to-end and unit tests for the recognition and extrusion pipeline.

Synthetic plans with known ground truth do most of the work here. Testing a
recognizer against images whose true wall count and scale are known is the only
way to catch a regression that leaves output still looking plausible -- which is
the characteristic way a CV pipeline breaks.
"""

from __future__ import annotations

import math

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
    Wall,
    WallType,
)
from app.main import app
from app.recognize.base import RecognizerError
from app.recognize.registry import get_recognizer
from app.recognize.skeleton import cluster_coordinates, merge_collinear, snap_to_axes
from app.synth import generate_plan


@pytest.fixture(scope="module")
def plan_image() -> np.ndarray:
    image, _ = generate_plan(with_clutter=True)
    return image


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


def test_recognizer_finds_walls_and_rooms(plan_image):
    ir = get_recognizer("opencv").recognize(plan_image, filename="synthetic.png")
    level = ir.levels[0]

    assert len(level.walls) >= 8, "should find the envelope plus the internal partitions"
    assert len(level.rooms) >= 3
    assert ir.provenance is not None
    assert ir.provenance.recognizer == "opencv"


def test_recognizer_ignores_furniture_and_text(plan_image):
    """Clutter is thin; walls are thick. Detection must not be inflated by it."""
    clean, _ = generate_plan(with_clutter=False)
    engine = get_recognizer("opencv")

    cluttered_walls = len(engine.recognize(plan_image).levels[0].walls)
    clean_walls = len(engine.recognize(clean).levels[0].walls)

    assert cluttered_walls == clean_walls


def test_recognizer_recovers_proportions(plan_image, truth):
    """Absolute scale is a guess, but SHAPE should be close to ground truth."""
    ir = get_recognizer("opencv").recognize(plan_image)
    detected = sum(w.length for w in ir.levels[0].walls)

    # Normalise out the known scale error to compare proportions alone.
    corrected = detected * (ir.scale.pixels_per_meter / truth.pixels_per_meter)
    assert 0.75 <= corrected / truth.total_wall_length_m <= 1.25


def test_supplied_scale_is_authoritative(plan_image):
    ir = get_recognizer("opencv").recognize(plan_image, pixels_per_meter=80.0)
    assert ir.scale.source is ScaleSource.MANUAL
    assert ir.scale.pixels_per_meter == 80.0
    assert ir.scale.confidence == 1.0
    # A supplied scale is certain, so the "this is a guess" warning must go.
    assert not any(w.code == "scale_is_a_guess" for w in ir.warnings)


def test_guessed_scale_is_flagged_loudly(plan_image):
    """A silently wrong scale is the worst failure mode: everything downstream
    looks right and is the wrong size. It must always be reported."""
    ir = get_recognizer("opencv").recognize(plan_image)
    assert ir.scale.source is ScaleSource.HEURISTIC
    assert ir.scale.confidence < 0.5
    assert any(w.code == "scale_is_a_guess" for w in ir.warnings)


def test_blank_image_is_rejected_not_guessed_at():
    blank = np.full((600, 800, 3), 255, dtype=np.uint8)
    with pytest.raises(RecognizerError):
        get_recognizer("opencv").recognize(blank)


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
    import cv2

    encoded = cv2.imencode(".png", plan_image)[1].tobytes()
    response = client.post(
        "/recognize", files={"file": ("plan.png", encoded, "image/png")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["schemaVersion"] == "1.0.0"
    assert "wallHeight" in body["levels"][0]
    assert body["source"]["filename"] == "plan.png"


def test_recognize_rejects_a_non_image(client: TestClient):
    response = client.post(
        "/recognize", files={"file": ("notes.txt", b"this is not a png", "text/plain")}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_recognize_rejects_an_unusable_image(client: TestClient):
    import cv2

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
