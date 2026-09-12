/**
 * Floor Plan IR - TypeScript view of the contract.
 *
 * The Pydantic models in `apps/cv/app/ir.py` are the source of truth; the
 * generated JSON Schema lives at `packages/schema/ir.schema.json`. These types
 * mirror it in its serialized (camelCase) form. When the Python models change,
 * re-run `python -m app.export_schema` and update this file to match - the
 * schema diff tells you exactly what moved.
 *
 * Plan coordinates are METRES, X right and Y DOWN. Y-down matches image space,
 * so the 2D overlay can draw IR directly over the uploaded floor plan with no
 * conversion at all. The 3D layer is the only place that flips to Y-up.
 */

export type Point = [number, number];

export type ScaleSource = "assumed" | "heuristic" | "ocr" | "manual";

export interface Scale {
  pixelsPerMeter: number;
  source: ScaleSource;
  confidence: number;
  note?: string | null;
}

export type WallType = "exterior" | "interior" | "partition";

export interface Wall {
  id: string;
  start: Point;
  end: Point;
  thickness: number;
  height?: number | null;
  type: WallType;
  confidence: number;
}

export type OpeningType = "door" | "window" | "opening";

export interface Opening {
  id: string;
  wallId: string;
  type: OpeningType;
  /** Metres from the wall's `start` to the opening's centre. */
  offset: number;
  width: number;
  height: number;
  /** Height of the opening's bottom edge above the floor. 0 for doors. */
  sill: number;
  swing?: "left" | "right" | null;
  confidence: number;
}

export type RoomType =
  | "unknown"
  | "living"
  | "kitchen"
  | "dining"
  | "bedroom"
  | "bathroom"
  | "hall"
  | "corridor"
  | "storage"
  | "garage"
  | "balcony"
  | "stairs";

export interface Room {
  id: string;
  name?: string | null;
  type: RoomType;
  polygon: Point[];
  area?: number | null;
  confidence: number;
}

export interface Level {
  id: string;
  name: string;
  elevation: number;
  wallHeight: number;
  walls: Wall[];
  openings: Opening[];
  rooms: Room[];
}

export type Severity = "info" | "warn" | "error";

export interface IRWarning {
  code: string;
  message: string;
  severity: Severity;
  ref?: string | null;
}

export interface SourceImage {
  filename: string;
  width: number;
  height: number;
  sha256?: string | null;
}

export interface Provenance {
  recognizer: string;
  recognizerVersion: string;
  generatedAt: string;
  durationMs?: number | null;
}

export interface FloorPlanIR {
  schemaVersion: "1.0.0";
  id: string;
  name: string;
  source?: SourceImage | null;
  scale: Scale;
  levels: Level[];
  provenance?: Provenance | null;
  warnings: IRWarning[];
}

// ---------------------------------------------------------------------------
// Derived values
// ---------------------------------------------------------------------------

export function wallLength(wall: Wall): number {
  return Math.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]);
}

export function planBounds(level: Level): {
  min: Point;
  max: Point;
  center: Point;
  size: Point;
} {
  const xs: number[] = [];
  const ys: number[] = [];

  for (const wall of level.walls) {
    xs.push(wall.start[0], wall.end[0]);
    ys.push(wall.start[1], wall.end[1]);
  }
  for (const room of level.rooms) {
    for (const [x, y] of room.polygon) {
      xs.push(x);
      ys.push(y);
    }
  }
  if (xs.length === 0) {
    return { min: [0, 0], max: [0, 0], center: [0, 0], size: [0, 0] };
  }

  const min: Point = [Math.min(...xs), Math.min(...ys)];
  const max: Point = [Math.max(...xs), Math.max(...ys)];
  return {
    min,
    max,
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2],
    size: [max[0] - min[0], max[1] - min[1]],
  };
}

export function totalFloorArea(level: Level): number {
  return level.rooms.reduce((sum, room) => sum + (room.area ?? 0), 0);
}

/**
 * Rescale a plan in place-ish (returns a new IR) to a corrected pixels-per-metre.
 *
 * Recognition produces correct PROPORTIONS even when absolute scale is a guess,
 * so correcting the scale is a uniform multiply over every coordinate rather
 * than a re-run of recognition. That is what makes the calibrate-by-clicking
 * flow instant: the user drags a line across a known dimension and the whole
 * model resizes without touching the server.
 */
export function rescale(ir: FloorPlanIR, newPixelsPerMeter: number): FloorPlanIR {
  const factor = ir.scale.pixelsPerMeter / newPixelsPerMeter;
  if (!Number.isFinite(factor) || factor <= 0) return ir;

  const scalePoint = ([x, y]: Point): Point => [x * factor, y * factor];

  return {
    ...ir,
    scale: {
      pixelsPerMeter: newPixelsPerMeter,
      source: "manual",
      confidence: 1,
      note: "Calibrated against a known distance on the drawing.",
    },
    levels: ir.levels.map((level) => ({
      ...level,
      walls: level.walls.map((wall) => ({
        ...wall,
        start: scalePoint(wall.start),
        end: scalePoint(wall.end),
        thickness: wall.thickness * factor,
      })),
      openings: level.openings.map((opening) => ({
        ...opening,
        offset: opening.offset * factor,
        width: opening.width * factor,
      })),
      rooms: level.rooms.map((room) => ({
        ...room,
        polygon: room.polygon.map(scalePoint),
        area: room.area == null ? null : room.area * factor * factor,
      })),
    })),
    // The guess is no longer in force, so its warning should not still be shown.
    warnings: ir.warnings.filter((w) => w.code !== "scale_is_a_guess"),
  };
}
