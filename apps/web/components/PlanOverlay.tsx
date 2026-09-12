"use client";

/**
 * The 2D review surface: recognised geometry drawn on top of the original
 * drawing, plus scale calibration.
 *
 * This is the beginning of the "manual confirmation" stage the platform
 * proposal builds its quality guarantees on. Seeing the IR superimposed on the
 * source image is what turns "the AI got it wrong somewhere" into "that wall is
 * in the wrong place", and it is the reason imperfect recognition is
 * commercially usable at all.
 *
 * Calibration earns its place here specifically. Recognition recovers correct
 * proportions but cannot recover absolute scale from wall thickness alone - it
 * is off by 15-40%, and nothing in the image says by how much. Two clicks
 * across a dimension the user can read off the drawing fixes it exactly.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import { type FloorPlanIR, type Point, wallLength } from "@/lib/ir";

type CalibrationPoint = { x: number; y: number };

export default function PlanOverlay({
  ir,
  imageUrl,
  levelIndex = 0,
  selectedWallId,
  onSelectWall,
  onCalibrate,
}: {
  ir: FloorPlanIR;
  /**
   * The source drawing to superimpose on. Null for vector input (DXF), which
   * has no raster to show — the recognised geometry IS the drawing there, and
   * it is exact rather than something to be checked against a background.
   */
  imageUrl: string | null;
  levelIndex?: number;
  selectedWallId?: string | null;
  onSelectWall?: (id: string) => void;
  /** Called with the corrected pixels-per-metre once the user confirms. */
  onCalibrate?: (pixelsPerMeter: number) => void;
}) {
  const level = ir.levels[levelIndex];
  const source = ir.source;

  const [showWalls, setShowWalls] = useState(true);
  const [showRooms, setShowRooms] = useState(true);
  const [calibrating, setCalibrating] = useState(false);
  const [points, setPoints] = useState<CalibrationPoint[]>([]);
  const [realDistance, setRealDistance] = useState("");

  const svgRef = useRef<SVGSVGElement>(null);

  // IR is in metres; the image is in pixels. One multiply converts between them,
  // so the overlay's viewBox can just be the source image's pixel dimensions.
  const ppm = ir.scale.pixelsPerMeter;
  const toPx = useCallback((p: Point): CalibrationPoint => ({ x: p[0] * ppm, y: p[1] * ppm }), [ppm]);

  const width = source?.width ?? 1000;
  const height = source?.height ?? 1000;

  const pixelDistance = useMemo(() => {
    if (points.length < 2) return 0;
    return Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
  }, [points]);

  const handleClick = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!calibrating || !svgRef.current) return;

    // Map the click through the SVG's own coordinate system so it stays correct
    // at any display size or zoom level.
    const svg = svgRef.current;
    const rect = svg.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * width;
    const y = ((event.clientY - rect.top) / rect.height) * height;

    setPoints((previous) => (previous.length >= 2 ? [{ x, y }] : [...previous, { x, y }]));
  };

  const applyCalibration = () => {
    const metres = parseFloat(realDistance);
    if (!Number.isFinite(metres) || metres <= 0 || pixelDistance <= 0) return;
    onCalibrate?.(pixelDistance / metres);
    setCalibrating(false);
    setPoints([]);
    setRealDistance("");
  };

  const scaleIsGuessed = ir.scale.source === "assumed" || ir.scale.source === "heuristic";
  // Calibration needs a drawing to measure against. It is also pointless when
  // the source stated its own units, where "correcting" the scale could only
  // introduce error.
  const canCalibrate = imageUrl !== null && scaleIsGuessed;

  return (
    <div className="flex h-full flex-col">
      {/* toolbar */}
      <div className="flex flex-wrap items-center gap-3 border-b border-white/10 px-4 py-2.5 text-sm">
        <Toggle checked={showWalls} onChange={setShowWalls} label="Walls" swatch="#38bdf8" />
        <Toggle checked={showRooms} onChange={setShowRooms} label="Rooms" swatch="#a78bfa" />

        <div className="ml-auto flex flex-wrap items-center gap-2">
          {scaleIsGuessed && !calibrating && (
            <span className="rounded bg-amber-500/15 px-2 py-1 text-xs text-amber-300">
              Scale is estimated
            </span>
          )}
          {!scaleIsGuessed && (
            <span className="rounded bg-emerald-500/15 px-2 py-1 text-xs text-emerald-300">
              Scale is exact
            </span>
          )}
          {canCalibrate && (
            <button
              onClick={() => {
                setCalibrating((c) => !c);
                setPoints([]);
              }}
              className={`rounded px-3 py-1.5 text-xs font-medium transition ${
                calibrating
                  ? "bg-amber-500 text-black"
                  : "bg-white/10 text-white hover:bg-white/20"
              }`}
            >
              {calibrating ? "Cancel" : "Calibrate scale"}
            </button>
          )}
        </div>
      </div>

      {calibrating && (
        <div className="flex flex-wrap items-center gap-3 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm">
          <span className="text-amber-200">
            {points.length === 0 && "Click one end of a known dimension."}
            {points.length === 1 && "Now click the other end."}
            {points.length === 2 && `Measured ${pixelDistance.toFixed(0)} px. How long is it?`}
          </span>
          {points.length === 2 && (
            <>
              <input
                type="number"
                step="0.01"
                min="0.01"
                autoFocus
                value={realDistance}
                onChange={(e) => setRealDistance(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && applyCalibration()}
                placeholder="metres"
                className="w-28 rounded border border-white/20 bg-black/40 px-2 py-1 text-white outline-none focus:border-amber-400"
              />
              <button
                onClick={applyCalibration}
                disabled={!realDistance}
                className="rounded bg-amber-500 px-3 py-1 text-xs font-medium text-black disabled:opacity-40"
              >
                Apply
              </button>
            </>
          )}
        </div>
      )}

      {/* drawing */}
      <div className="relative flex-1 overflow-auto bg-neutral-900 p-4">
        <div className="relative mx-auto w-full max-w-4xl">
          {imageUrl ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={imageUrl} alt="Uploaded floor plan" className="block w-full" />
          ) : (
            /* Vector source: nothing to superimpose on, so the SVG below sizes
               itself from the plan's own aspect ratio over a blank sheet. */
            <div
              className="w-full bg-neutral-100"
              style={{ aspectRatio: `${width} / ${height}` }}
            />
          )}

          <svg
            ref={svgRef}
            viewBox={`0 0 ${width} ${height}`}
            onClick={handleClick}
            className={`absolute inset-0 h-full w-full ${
              calibrating ? "cursor-crosshair" : ""
            }`}
          >
            {showRooms &&
              level.rooms.map((room) => {
                const d = room.polygon.map(toPx).map((p) => `${p.x},${p.y}`).join(" ");
                return (
                  <polygon
                    key={room.id}
                    points={d}
                    fill="rgba(167,139,250,0.18)"
                    stroke="rgba(167,139,250,0.75)"
                    strokeWidth={Math.max(1, width / 600)}
                  />
                );
              })}

            {showWalls &&
              level.walls.map((wall) => {
                const a = toPx(wall.start);
                const b = toPx(wall.end);
                const isSelected = wall.id === selectedWallId;
                return (
                  <line
                    key={wall.id}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={isSelected ? "#f59e0b" : "#38bdf8"}
                    strokeWidth={Math.max(2, wall.thickness * ppm * 0.5)}
                    strokeOpacity={isSelected ? 1 : 0.8}
                    strokeLinecap="round"
                    className={calibrating ? "" : "cursor-pointer"}
                    onClick={(event) => {
                      if (calibrating) return;
                      event.stopPropagation();
                      onSelectWall?.(wall.id);
                    }}
                  />
                );
              })}

            {/* calibration ruler */}
            {points.length > 0 && (
              <g>
                {points.length === 2 && (
                  <line
                    x1={points[0].x}
                    y1={points[0].y}
                    x2={points[1].x}
                    y2={points[1].y}
                    stroke="#fbbf24"
                    strokeWidth={Math.max(2, width / 400)}
                    strokeDasharray={`${width / 90} ${width / 140}`}
                  />
                )}
                {points.map((p, index) => (
                  <circle
                    key={index}
                    cx={p.x}
                    cy={p.y}
                    r={Math.max(4, width / 180)}
                    fill="#fbbf24"
                    stroke="#000"
                    strokeWidth={1}
                  />
                ))}
              </g>
            )}
          </svg>
        </div>
      </div>

      {/* footer stats */}
      <div className="flex flex-wrap gap-x-6 gap-y-1 border-t border-white/10 px-4 py-2 text-xs text-neutral-400">
        <span>{level.walls.length} walls</span>
        <span>{level.rooms.length} rooms</span>
        <span>
          {level.walls.reduce((sum, w) => sum + wallLength(w), 0).toFixed(1)} m of wall
        </span>
        <span className={scaleIsGuessed ? "text-amber-400" : "text-emerald-400"}>
          {ppm.toFixed(1)} px/m &middot; {ir.scale.source}
        </span>
      </div>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  swatch,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  swatch: string;
}) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-1.5 text-neutral-300">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5 accent-sky-400"
      />
      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: swatch }} />
      {label}
    </label>
  );
}
