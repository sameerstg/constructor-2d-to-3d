# Constructor — floor plan to 3D

Upload a 2D floor plan image, get an editable 3D model of the home.

This is the first slice of the XR Home Visualization & Design Platform: the
Phase 1 pipeline from the proposal, end to end, with the recognition step built
as a swappable part so the AI model can be chosen on evidence rather than up
front.

```
 image  ──▶  recognizer  ──▶  IR  ──▶  human review  ──▶  extruder  ──▶  GLB
            (swappable)     (JSON)      (2D editor)      (parametric)    (VR)
```

---

## The one idea worth understanding

**"2D image to 3D model" is not one AI model.** It is two problems with
completely different natures, and conflating them is how this kind of project
fails:

| | Problem | Difficulty | Solved by |
| --- | --- | --- | --- |
| **A** | image → **vector floor plan** | hard, fuzzy | AI + CV + a human |
| **B** | vector floor plan → **3D mesh** | easy, deterministic | parametric extrusion |

Generative image-to-3D models (TripoSR, InstantMesh, Hunyuan3D) solve neither.
They produce unstructured blobs with no metric scale and no addressable parts,
so you could never swap the material on one named wall — which is exactly what
Phase 2 requires.

Between A and B sits the **IR**: a JSON document of walls, openings, rooms and
scale. It is the "shared scene data package" from section 5 of the proposal. The
web configurator edits it, the extruder consumes it, and the VR client reads it.
Get this schema right and Phases 2–5 are mostly plumbing.

It is also where the human fits. Recognition is never perfect, so the IR is
reviewed and corrected **before** any geometry exists — which is what makes an
imperfect recognizer into a usable product.

---

## Running it

Two services. Both must be running.

**CV service** (Python 3.11, port 8000):

```bash
cd apps/cv && uv venv && uv pip install -e ".[dev]" && .venv/Scripts/python -m uvicorn app.main:app --reload
```

**Web app** (port 3000):

```bash
cd apps/web && npm install && npm run dev
```

Open http://localhost:3000 and click **Sample plan**, or drop in your own.

> Use `localhost`, not `127.0.0.1`. Next treats them as different origins and
> silently refuses to load client chunks on the mismatch — the page renders but
> never hydrates. `allowedDevOrigins` in `next.config.ts` covers the loopback IP.

### Generating test data

The client's sample plans arrive at the start of Phase 1. Until then, synthetic
plans with **known ground truth** are more useful anyway, because they let the
recognizer be scored rather than eyeballed:

```bash
cd apps/cv && .venv/Scripts/python -m app.synth --out ../../samples
```

### Scoring a recognizer

```bash
cd apps/cv && .venv/Scripts/python -m app.evaluate --samples ../../samples
```

This is the tool that turns "should we license an AI recognition API?" into a
table of numbers on the client's own drawings. Run it before that decision.

---

## Layout

```
apps/cv/                  Python recognition + extrusion service
  app/ir.py               ** the IR contract — source of truth **
  app/recognize/base.py   the Recognizer protocol (image -> IR)
  app/recognize/opencv_v1.py   classical baseline recognizer
  app/recognize/skeleton.py    skeleton -> clean wall segments
  app/extrude/gltf.py     IR -> watertight GLB
  app/synth.py            synthetic plans with ground truth
  app/evaluate.py         score a recognizer against ground truth
apps/web/                 Next.js app
  lib/ir.ts               TypeScript view of the IR + rescale()
  components/PlanOverlay.tsx   2D review surface + scale calibration
  components/Viewer3D.tsx      live 3D view (react-three-fiber)
packages/schema/          generated JSON Schema for the IR
docs/LICENSING.md         ** read before adding any model or dataset **
```

## Adding a recognizer

Implement `image -> FloorPlanIR`, then add one line to
`app/recognize/registry.py`. Nothing else changes — not the API, not the editor,
not the extruder. That is the point: the proposal commits to trialing and
finalizing a recognition model during Phase 1, so the recognizer is a swappable
part from day one, and the whole product can be built and tested before anyone
commits to a vendor.

---

## Where it actually stands

Measured against synthetic plans with known ground truth
(`python -m app.evaluate`):

| | Result |
| --- | --- |
| Walls | traced cleanly; furniture, text and dimension lines correctly ignored |
| Rooms | 4–5 of 5 detected |
| Wall length | within ~11% once scale is corrected |
| Scale (inferred) | **15–40% error** |
| Scale (calibrated) | ~0.1% error |
| Openings | not detected — see below |
| Runtime | 30–140 ms per plan |

### Scale is the weak link, and it is not fixable by better image processing

The recognizer recovers correct **proportions** but cannot recover absolute
**size**. It infers scale by assuming a typical interior wall is 0.20 m thick,
and when the drawing disagrees the whole model is uniformly wrong. The error is
in the assumption, not the measurement.

Evidence: half-width measurements form two clean populations — interior
partitions and thicker exterior walls — and on most plans the exterior perimeter
is the longer run, so a median lands on the exterior and over-estimates by ~50%.
Correcting that bias (using a low percentile) helps, but a plan drawn with 0.12 m
partitions is still 25% off, and nothing in the image says which case you are in.

So scale comes from outside the wall geometry. Three sources, in order of
increasing effort:

1. **Calibrate by clicking** — implemented. The user drags across a dimension
   they can read off the drawing and types the real distance. Measured at 0.1%
   error against ground truth. Because proportions are already right, this is a
   uniform multiply over the IR, applied instantly client-side.
2. **OCR the dimension text** — the natural automation of (1). Use PaddleOCR or
   Tesseract, both Apache-2.0.
3. **Read the stated drawing scale** ("SCALE 1:100") plus the source DPI.

### Openings are deliberately not detected

Reliable door and window detection needs symbol recognition, which is the ML
recognizer's job. Phase 1 of the proposal is explicitly *"walls only, empty"*, so
this matches the agreed scope — and a half-working door detector would create
more review work than it saves.

The IR and the extruder both **fully support** openings, including real
boolean-cut geometry, and this is covered by tests. They arrive through the
review editor or a later recognizer without any structural change.

---

## What comes next

| | Work | Why |
| --- | --- | --- |
| **M3** | Wall editing in the 2D review surface — drag endpoints, split, delete, add openings | The highest-value remaining component. It is what makes imperfect recognition commercially acceptable, and it is the proposal's stage-3 manual confirmation. |
| **M4** | ML recognizer behind the same interface, A/B'd against the classical baseline on the client's plans | Only now is the build-vs-buy decision on an AI recognition API answerable with evidence. Read `docs/LICENSING.md` first. |
| **M4** | OCR for scale and room names | Removes the biggest source of error and the main manual step. |
| **M5** | Persistence, projects, auth | Currently everything is in-memory and per-session. |
| **M5** | Materials and furniture on named parts | Phase 2 of the proposal. Every mesh is already named for its IR entity (`wall_w_1a2b3c`), which is the addressing scheme this needs. |

## Testing

```bash
cd apps/cv && .venv/Scripts/python -m pytest    # 29 tests
cd apps/web && npm run lint && npx tsc --noEmit
```
