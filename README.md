# Constructor — floor plan to 3D

Upload a floor plan — **DXF** from AutoCAD or Revit, or a raster scan — and get
an editable 3D model of the home.

This is the first slice of the XR Home Visualization & Design Platform: the
Phase 1 pipeline from the proposal, end to end, with the recognition step built
as a swappable part so the AI model can be chosen on evidence rather than up
front.

```
 DXF   ─┐
        ├──▶  recognizer  ──▶  IR  ──▶  human review  ──▶  extruder  ──▶  GLB
 image ─┘    (swappable)     (JSON)      (2D editor)      (parametric)    (VR)
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

Open http://localhost:3000 and click a sample — **DXF (CAD)**, **Revit plot**,
**AutoCAD plot**, or one of the real plans — or drop in your own.

> If port 3000 is taken, Next picks the next free port and prints it. Check the
> dev server's output rather than assuming 3000.

> Use `localhost`, not `127.0.0.1`. Next treats them as different origins and
> silently refuses to load client chunks on the mismatch — the page renders but
> never hydrates. `allowedDevOrigins` in `next.config.ts` covers the loopback IP.

### Generating test data

The client's sample plans arrive at the start of Phase 1. Until then, synthetic
plans with **known ground truth** are more useful anyway, because they let the
recognizer be scored rather than eyeballed:

```bash
cd apps/cv && .venv/Scripts/python -m app.synth --out ../../samples             # raster
cd apps/cv && .venv/Scripts/python -m app.synth_dxf --out ../../samples         # CAD vector
cd apps/cv && .venv/Scripts/python -m app.synth_cad_raster --out ../../samples  # CAD plotted to raster
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
  app/recognize/base.py   the Recognizer protocol (document -> IR)
  app/recognize/opencv_v1.py   classical raster recognizer (scans, photos)
  app/recognize/skeleton.py    skeleton -> clean wall segments
  app/recognize/dxf_v1.py      vector CAD recognizer (AutoCAD/Revit DXF)
  app/recognize/wallpairs.py   CAD double-lines -> wall centrelines
  app/sources.py               format detection and routing
  app/extrude/gltf.py     IR -> watertight GLB
  app/synth.py            synthetic raster plans with ground truth
  app/synth_dxf.py        synthetic CAD plans, drawn to AIA conventions
  app/synth_cad_raster.py AutoCAD/Revit-style rasters, rendered from that DXF
  app/evaluate.py         score a recognizer against ground truth
apps/web/                 Next.js app
  lib/ir.ts               TypeScript view of the IR + rescale()
  components/PlanOverlay.tsx   2D review surface + scale calibration
  components/Viewer3D.tsx      live 3D view (react-three-fiber)
packages/schema/          generated JSON Schema for the IR
docs/CAD-INPUT.md         getting DXF out of AutoCAD, Revit, and the rest
docs/LICENSING.md         ** read before adding any model or dataset **
```

## Adding a recognizer

Implement `SourceDocument -> FloorPlanIR`, declare which source kinds it
accepts, then add one line to
`app/recognize/registry.py`. Nothing else changes — not the API, not the editor,
not the extruder. That is the point: the proposal commits to trialing and
finalizing a recognition model during Phase 1, so the recognizer is a swappable
part from day one, and the whole product can be built and tested before anyone
commits to a vendor.

---

## Where it actually stands

Two recognizers, measured with `python -m app.evaluate --samples ../../samples`.
The gap between them is the argument for sending CAD rather than a picture of it:

| | Raster scan (`opencv`) | **DXF (`dxf`)** |
| --- | --- | --- |
| Walls | traced cleanly; clutter ignored | **14 of 14, exact** |
| Wall thickness | one global estimate | **measured per wall, exact** |
| Building size | ±11% after calibration | **exact** |
| Rooms | 4–5 of 5 | **5 of 5** |
| Room names | not detected | **read from drawing text** |
| Scale | guessed, **15–40% error** | **exact, from the file's units** |
| Openings | not detected | **10 of 10** (doors + windows) |
| Runtime | 30–500 ms | 25–90 ms |

### On real drawings

The synthetic plans flatter the raster recognizer. These are what it does on
real floor plans and on CAD plots rendered to raster:

| Sample | Walls | Rooms | Verdict |
| --- | --- | --- | --- |
| `synthetic_cad.dxf` | **14** (exact) | **5** (exact) | perfect — vector input |
| `revit_style.png` | **14** (exact) | 1 | walls right, rooms leak |
| `images.png` (real) | 45 | 5 | usable after review |
| `cedreo-…webp` (real) | 56 | 11 | usable after review |
| `Black-and-White-…webp` (real) | 21 | 1 | walls usable, rooms poor |
| `autocad_style.png` | 57 (truth 14) | 0 | **fails** — see below |

Two failures found by running real plans, both now fixed, and both worth
recording because each produced *plausible-looking* output while being wrong:

1. **Rooms inherited the scale guess, squared.** The minimum-room-area threshold
   was derived from pixels-per-metre. A 2–3× scale error scales an *area*
   threshold by 4–9×, silently discarding every real room — real plans returned
   **zero** rooms. Both room thresholds are now derived from *measured* wall
   thickness, which is scale-free. `images.png` went 0 → 5 rooms.
2. **Solid-walled plans lost every interior partition.** The rough thickness
   estimate used the 80th percentile, which on a drawing with a thick filled
   perimeter lands inside the *exterior* population; the wall mask was then cut
   at 0.6× that and deleted every thinner partition. A Revit-style plot returned
   **4** walls instead of 14. Now measured at the 60th percentile, falling back
   to the 80th only when that bottoms out at hairline width. Revit: 4 → 14.

### The one case that does not work: unfilled CAD line-work

`autocad_style.png` is a plotted AutoCAD drawing where walls are **two hairlines
with nothing between them**. The whole raster approach rests on walls being the
*thick* dark structures, and here nothing is thick — the wall lines are the same
weight as the furniture. It returns 57 walls against a true 14.

This is not a tuning problem, it is the wrong input. A recognizer that only
works on filled walls should be seen to only work on filled walls, so the
drawing is kept as a sample rather than quietly dropped, and the recognizer now
emits a `hairline_linework` **error** telling the user to upload the DXF — where
the same building comes out exactly right. Recovering hollow walls from a raster
means detecting parallel line pairs in pixels, which is the vector
[`wallpairs.py`](apps/cv/app/recognize/wallpairs.py) problem all over again;
worth doing only if clients cannot supply DXF.

### Why DXF is so much better, concretely

A DXF *states* what the raster pipeline has to infer. Units give exact scale.
Layer names (`A-WALL`) say which lines are walls. Walls are drawn as two
parallel faces, so the gap between a matched pair **is** the thickness, measured
rather than assumed. Doors and windows are named block references with real
positions. Room labels are text placed inside the room.

Recovering the wall centreline from those face pairs is the one genuinely hard
geometric step, and it lives in
[`wallpairs.py`](apps/cv/app/recognize/wallpairs.py).

Rooms are the exception: DXF has no concept of an enclosed room unless the
drawing happens to carry one. Rather than write a second room finder, the
recovered centrelines are rasterized and handed to the existing raster room
detection, so door-gap bridging exists once and improves both paths.

### Scale is the weak link *for raster input*, and better image processing cannot fix it

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

Scale inference only ever applies to raster input. **DXF states its units, so
none of this arises** — the calibration control is hidden and the scale is
marked exact.

### Openings: detected from CAD, not from rasters

DXF gives doors and windows directly, as named block references: 10 of 10 on the
test plan, sized from the block's own geometry and anchored to the nearest wall.

The raster recognizer deliberately does **not** attempt them. Reliable symbol
detection in pixels needs a trained model, Phase 1 of the proposal is explicitly
*"walls only, empty"*, and a half-working door detector creates more review work
than it saves. The IR and extruder support openings fully either way, including
real boolean-cut geometry.

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
cd apps/cv && .venv/Scripts/python -m pytest    # 57 tests
cd apps/web && npm run lint && npx tsc --noEmit
```
