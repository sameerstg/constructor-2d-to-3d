# Licensing constraints

Read this before adding any model, dataset or library to the recognition
pipeline. The platform's endpoint is a **white-label, multi-tenant SaaS sold to
other builder companies** (Phase 5 of the proposal), which is the most
restrictive possible use. A dependency that is fine for a prototype can be fatal
at that stage, and by then it is load-bearing.

Two rules follow from that:

1. Judge every dependency against Phase 5, not against today.
2. Weights inherit their training data's licence. "The code is Apache-2.0" says
   nothing about whether you can ship the model.

---

## Landmines

### CubiCasa5K — CC BY-NC 4.0 — NON-COMMERCIAL

[CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) is the default floor plan
dataset: 5,000 annotated plans, and the benchmark nearly every published
floor-plan paper reports against. Its licence is **Creative Commons
Attribution-NonCommercial 4.0**.

This matters more than it first appears, because **most pretrained floor-plan
weights you will find are trained on it**. Those weights are derived works. Using
them in a commercial product is a licence violation even though the model code
is usually permissively licensed.

What is safe:

- Using it to **benchmark** recognizers during development.
- Reading the papers and reimplementing the architectures.

What is not:

- Shipping weights trained on it, in any product that is sold.

Paths forward, in order of preference:

1. **Fine-tune on the client's own floor plans.** The client is supplying sample
   plans in Phase 1 anyway, and a builder company has thousands of drawings in
   the same house style. A model fitted to one customer's drawing conventions
   will beat a general model on that customer's drawings regardless of licence.
2. **License CubiCasa commercially.** They sell a conversion API; talk to them.
3. **Use a permissively licensed dataset**, accepting weaker coverage.

### Ultralytics YOLO — AGPL-3.0

YOLOv8/v11 are the obvious reach for door and window symbol detection and are
**AGPL-3.0**. The AGPL's network clause is triggered by exactly what this
platform is: software offered to users over a network. It would oblige releasing
the platform's source to every tenant.

Use **RT-DETR** or **MMDetection** (both Apache-2.0) instead. They are
competitive for this task. Alternatively Ultralytics sells a commercial licence.

### Anything where weights and code are licensed separately

Check both, every time. A permissive repository routinely ships weights under a
research-only or non-commercial licence.

---

## Current dependencies

Everything presently in the pipeline is permissively licensed and safe for
commercial and white-label use.

| Package | Licence | Role |
| --- | --- | --- |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Image processing |
| NumPy | BSD-3-Clause | Arrays |
| Shapely | BSD-3-Clause | Polygon geometry |
| scikit-image | BSD-3-Clause | Skeletonization |
| Pillow | MIT-CMU | Image IO |
| trimesh | MIT | Mesh construction, glTF export |
| manifold3d | Apache-2.0 | Mesh booleans for openings |
| mapbox-earcut | ISC | Polygon triangulation |
| FastAPI / Starlette / Pydantic | MIT / BSD-3 / MIT | API |
| three.js | MIT | 3D rendering |
| react-three-fiber, drei | MIT | React renderer for three.js |
| Next.js, React | MIT | Web app |

**No model weights ship today.** The classical recognizer is pure algorithm,
which is precisely why it is a good baseline: it has no licence surface at all.

---

## Watch list for later phases

| Phase | Component | Concern |
| --- | --- | --- |
| 1 | AI recognition model or API | The subject of this document. Settle it before the model is load-bearing. |
| 3 | Furniture library (100+ models) | The proposal has the client purchasing these. Most stock-3D licences permit use in a product but **not** redistribution to sub-licensees — which is what white-labelling in Phase 5 does. Verify the terms cover it **before purchase**, not after. |
| 4 | Multiplayer networking | Usually a commercial SaaS; check per-CCU pricing and any attribution terms. |
| 5 | Billing provider | Standard commercial terms; low risk. |
| Any | Blender for asset optimization | GPL. Fine as a **tool** you run; do not link `bpy` into distributed code. |
| Any | IfcOpenShell, if BIM/IFC export is wanted | LGPL-3.0. Usable, but keep it in a separate process rather than linking it into proprietary code. |

The Phase 3 asset-library row is the one most likely to be missed, because
purchasing happens long before Phase 5 makes the redistribution question real.
