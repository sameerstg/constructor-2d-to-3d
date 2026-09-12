# Getting a CAD floor plan in

**Short version: export DXF.** Both AutoCAD and Revit do it in one step, it is
an open format, and it carries everything the pipeline needs.

## Why this matters more than it sounds

A raster scan and a DXF of the same building are not the same input. The DXF
*states* what the raster pipeline has to guess:

| | Raster scan | DXF |
| --- | --- | --- |
| Scale | guessed from wall thickness, **15–40% error** | **exact**, from the file's units |
| Wall thickness | one global estimate for every wall | **measured per wall**, face to face |
| Which lines are walls | inferred from stroke weight | **layer name** |
| Doors and windows | not detected at all | **block references**, with position and size |
| Room names | not detected | **text**, placed inside the room |

Measured on the same synthetic building, the DXF path returns exact dimensions,
exact per-wall thicknesses, every room with its name, and all ten openings. The
raster path returns approximate proportions and no openings.

So if the drawing exists in CAD, **do not send a screenshot of it**.

---

## From your tool

### AutoCAD → DXF

`SAVEAS` → *AutoCAD DXF (\*.dxf)*. Choose **ASCII**, not Binary.

Or for a whole folder, the free [ODA File
Converter](https://www.opendesign.com/guestfiles/oda_file_converter) batch
converts DWG → DXF.

### Revit → DXF

*File → Export → CAD Formats → DXF*. Export the **plan view** you want, not a
3D view. In the export dialog's *Layers* tab, the default AIA layer mapping is
what this pipeline expects — leave it.

### Revit → IFC (richer, not yet supported)

*File → Export → IFC*. IFC is the better input in principle: `IfcWall`,
`IfcDoor` and `IfcSpace` are first-class objects with real dimensions and
relationships, so there is nothing to infer at all. It needs
[IfcOpenShell](https://github.com/ifcopenshell/ifcopenshell) (LGPL-3.0), which
per [LICENSING.md](LICENSING.md) should run out-of-process. Worth adding when a
client works in Revit natively.

### SketchUp, ArchiCAD, Vectorworks, Chief Architect

All export DXF. Same advice.

---

## What we do and do not read

| Format | Status | Notes |
| --- | --- | --- |
| **DXF** (ASCII) | ✅ supported | R12 through R2018+. The recommended input. |
| **PNG / JPEG / WebP / BMP / TIFF** | ✅ supported | Scans and photographs. Scale must be calibrated. |
| DXF (binary) | ❌ | Re-export as ASCII DXF. |
| **DWG** | ❌ refused with advice | Proprietary. No permissively licensed reader exists. Convert to DXF. |
| **RVT** | ❌ refused with advice | Proprietary. Export DXF or IFC from Revit. |
| **IFC** | 🔜 detected, no reader yet | The natural next addition. |
| **PDF** | 🔜 detected, no reader yet | Vector PDFs retain line geometry; `pdfplumber` (MIT) is the permissive route. |

Refusals return **415** with the conversion instructions above, not a generic
error. A proprietary format is not a bug to debug — it is a one-step export the
user needs to be told about.

---

## Making your drawings read well

The recognizer follows CAD convention. The closer your drawing is to standard
practice, the better it does.

**Layers.** Walls are found by layer name — `A-WALL` (AIA/NCS), `WALL`, `MUR`,
`WAND` and similar, in several languages. If no layer looks like a wall layer,
**every** line in the drawing is considered and furniture will show up as walls.
You will get a `no_wall_layer` warning when this happens. Naming the wall layer
`A-WALL` is the single highest-value fix.

**Walls as double lines.** Draw walls as their two faces, as CAD normally does.
The gap between a matched pair becomes the wall thickness. A wall drawn as a
single centreline still works but gets a default thickness and a lower
confidence, and you will see an `unpaired_wall_lines` warning.

**Units.** Set them. `$INSUNITS = 0` (unitless) means we assume millimetres and
warn — and if the drawing was actually in metres, the building comes out 1000×
too large. This is the one error that is catastrophic rather than cosmetic.

**Doors and windows as blocks.** Named block references containing "door" or
"window" (or their equivalents in several languages) are detected, sized from
the block's own geometry, and attached to the nearest wall. Openings drawn as
loose lines are invisible to us and must be added in review.

**Room names as text.** A TEXT or MTEXT entity placed inside a room becomes that
room's name, and known words ("KITCHEN", "BED 1") also set the room *type* —
which is what lets Phase 2 pick materials and fixtures per room automatically.

**Model space, not paper space.** Export the model view. A drawing whose
geometry lives only in a layout tab reads as empty.

---

## Warnings you may see

| Code | Meaning |
| --- | --- |
| `dxf_units_unspecified` | `$INSUNITS` is 0; millimetres assumed. Check the model is not 1000× off. |
| `no_wall_layer` | No wall-like layer name; every line was considered. |
| `unpaired_wall_lines` | Some lines had no parallel partner; default thickness used. |
| `unattached_openings` | Door/window blocks too far from any wall to attach. |
| `no_openings_detected` | No door or window blocks found — they may be loose geometry. |
| `no_rooms_detected` | Walls do not quite close; the model will have no floors. |
