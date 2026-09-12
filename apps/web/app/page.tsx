"use client";

/**
 * Floor plan -> IR -> 3D, end to end.
 *
 * The flow is deliberately three distinct steps rather than one magic button,
 * because the middle step is where the quality comes from:
 *
 *   upload  ->  REVIEW the recognised plan  ->  3D model
 *
 * Dropping the review step is what makes automated floor plan conversion
 * unreliable in practice. Keeping it visible - with the recognised geometry
 * drawn over the original and the scale flagged when it is a guess - is what
 * makes an imperfect recogniser into a usable product.
 */

import dynamic from "next/dynamic";
import { useCallback, useRef, useState } from "react";
import PlanOverlay from "@/components/PlanOverlay";
import { type FloorPlanIR, rescale, totalFloorArea } from "@/lib/ir";

// three.js has no business in the server bundle, and the viewer touches
// document on mount.
const Viewer3D = dynamic(() => import("@/components/Viewer3D"), {
  ssr: false,
  loading: () => <Centered>Loading 3D view…</Centered>,
});

type Status = "idle" | "working" | "ready" | "error";
type Tab = "review" | "model";

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [ir, setIr] = useState<FloorPlanIR | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("review");
  const [selectedWallId, setSelectedWallId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setStatus("working");
    setError(null);
    setSelectedWallId(null);

    // Only rasters have something to show behind the recognised geometry. A DXF
    // is vector data the browser cannot render as an image, so the review view
    // draws the IR on its own.
    const isRaster = file.type.startsWith("image/");

    // Revoke the previous object URL before replacing it, or every upload in a
    // session leaks its image.
    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return isRaster ? URL.createObjectURL(file) : null;
    });

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch("/api/recognize", { method: "POST", body: form });
      const payload = await response.json();

      if (!response.ok) {
        setError(payload?.detail ?? "Recognition failed.");
        setStatus("error");
        return;
      }

      setIr(payload as FloorPlanIR);
      setTab("review");
      setStatus("ready");
    } catch {
      setError("Could not reach the server.");
      setStatus("error");
    }
  }, []);

  const downloadGlb = useCallback(async () => {
    if (!ir) return;
    const response = await fetch("/api/extrude", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(ir),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      setError(payload?.detail ?? "Export failed.");
      return;
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${ir.name || "floorplan"}.glb`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [ir]);

  const level = ir?.levels[0];

  return (
    <main className="flex h-dvh flex-col bg-neutral-950 text-neutral-100">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-white/10 px-5 py-3">
        <div>
          <h1 className="text-sm font-semibold tracking-tight">Constructor</h1>
          <p className="text-xs text-neutral-500">Floor plan to 3D model</p>
        </div>

        {ir && (
          <nav className="flex gap-1 rounded-lg bg-white/5 p-1 lg:ml-4">
            <TabButton active={tab === "review"} onClick={() => setTab("review")}>
              1. Review
            </TabButton>
            <TabButton active={tab === "model"} onClick={() => setTab("model")}>
              2. 3D model
            </TabButton>
          </nav>
        )}

        <div className="ml-auto flex items-center gap-2">
          {ir && (
            <button
              onClick={downloadGlb}
              className="rounded-md bg-sky-500 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-sky-400"
            >
              Download .glb
            </button>
          )}
          <button
            onClick={() => inputRef.current?.click()}
            className="rounded-md bg-white/10 px-3 py-1.5 text-xs font-medium transition hover:bg-white/20"
          >
            {ir ? "New plan" : "Upload plan"}
          </button>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".dxf,image/png,image/jpeg,image/webp,image/bmp,image/tiff"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = "";
          }}
        />
      </header>

      {/* Below lg the side panel stacks under the drawing instead of squeezing
          it; a 288px fixed column next to a floor plan is unusable on a narrow
          window, and the plan is the thing worth the space. */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <section className="min-h-0 min-w-0 flex-1">
          {status === "idle" && <Dropzone onFile={handleFile} />}
          {status === "working" && <Centered>Reading the drawing…</Centered>}
          {status === "error" && (
            <Centered>
              <p className="max-w-md text-center text-sm text-red-400">{error}</p>
              <button
                onClick={() => setStatus("idle")}
                className="mt-4 rounded-md bg-white/10 px-3 py-1.5 text-xs hover:bg-white/20"
              >
                Try another file
              </button>
            </Centered>
          )}

          {status === "ready" && ir && tab === "review" && (
            <PlanOverlay
              ir={ir}
              imageUrl={imageUrl}
              selectedWallId={selectedWallId}
              onSelectWall={setSelectedWallId}
              onCalibrate={(ppm) => setIr((current) => (current ? rescale(current, ppm) : current))}
            />
          )}

          {status === "ready" && ir && tab === "model" && (
            <Viewer3D
              ir={ir}
              selectedWallId={selectedWallId}
              onSelectWall={setSelectedWallId}
            />
          )}
        </section>

        {ir && level && (
          <aside className="max-h-64 w-full shrink-0 overflow-y-auto border-t border-white/10 bg-neutral-900/60 p-4 text-sm lg:max-h-none lg:w-72 lg:border-l lg:border-t-0">
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500">
              Plan
            </h2>
            <dl className="space-y-1.5 text-xs">
              <Row label="Walls" value={String(level.walls.length)} />
              <Row label="Rooms" value={String(level.rooms.length)} />
              <Row label="Openings" value={String(level.openings.length)} />
              <Row label="Floor area" value={`${totalFloorArea(level).toFixed(1)} m²`} />
              <Row label="Wall height" value={`${level.wallHeight.toFixed(2)} m`} />
              <Row
                label="Scale"
                value={`${ir.scale.pixelsPerMeter.toFixed(1)} px/m`}
                tone={ir.scale.source === "manual" ? "good" : "warn"}
              />
              {ir.provenance && (
                <Row
                  label="Recognizer"
                  value={`${ir.provenance.recognizer} ${ir.provenance.recognizerVersion}`}
                />
              )}
              {ir.provenance?.durationMs != null && (
                <Row label="Took" value={`${ir.provenance.durationMs.toFixed(0)} ms`} />
              )}
            </dl>

            {ir.warnings.length > 0 && (
              <>
                <h2 className="mb-2 mt-6 text-xs font-semibold uppercase tracking-wider text-neutral-500">
                  Needs attention
                </h2>
                <ul className="space-y-2">
                  {ir.warnings.map((warning) => (
                    <li
                      key={warning.code}
                      className={`rounded border-l-2 px-2.5 py-2 text-xs leading-relaxed ${
                        warning.severity === "error"
                          ? "border-red-500 bg-red-500/10 text-red-200"
                          : warning.severity === "warn"
                            ? "border-amber-500 bg-amber-500/10 text-amber-200"
                            : "border-sky-500 bg-sky-500/10 text-sky-200"
                      }`}
                    >
                      {warning.message}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </aside>
        )}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------

/** Bundled demo plans, so the tool can be tried before any real plan exists.
 *
 * Ordered best-case first. The spread is deliberate: DXF is exact, a Revit plot
 * is recognisable because its walls are filled solid, an AutoCAD plot is the
 * hard case because its walls are hollow outlines, and the real plans are what
 * actually turns up in an inbox.
 */
const SAMPLES = [
  { label: "DXF (CAD)", path: "/samples/synthetic_cad.dxf", type: "" },
  { label: "Revit plot", path: "/samples/revit_style.png", type: "image/png" },
  { label: "AutoCAD plot", path: "/samples/autocad_style.png", type: "image/png" },
  { label: "Real plan", path: "/samples/images.png", type: "image/png" },
  {
    label: "Real plan (dense)",
    path: "/samples/cedreo-en_2d-floor-plan-WB_apartment-01.webp",
    type: "image/webp",
  },
];

function Dropzone({ onFile }: { onFile: (file: File) => void }) {
  const [dragging, setDragging] = useState(false);

  const loadSample = async (sample: (typeof SAMPLES)[number]) => {
    const response = await fetch(sample.path);
    const blob = await response.blob();
    const name = sample.path.split("/").pop() ?? "sample";
    // A dev server serves .dxf as octet-stream or text/plain; pin the type so
    // the raster-vs-vector check downstream is not at the mercy of that.
    onFile(new File([blob], name, { type: sample.type }));
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onFile(file);
      }}
      className="flex h-full items-center justify-center p-8"
    >
      <div
        className={`flex w-full max-w-lg flex-col items-center rounded-2xl border-2 border-dashed p-14 text-center transition ${
          dragging ? "border-sky-400 bg-sky-500/10" : "border-white/15"
        }`}
      >
        <p className="text-base font-medium">Drop a floor plan here</p>
        <p className="mt-2 max-w-sm text-xs leading-relaxed text-neutral-500">
          DXF from AutoCAD or Revit gives exact dimensions, doors and room names.
          Raster scans (PNG, JPEG, WebP) also work, with estimated scale.
        </p>
        <div className="mt-6 flex gap-2">
          {SAMPLES.map((sample) => (
            <button
              key={sample.path}
              onClick={() => loadSample(sample)}
              className="rounded-md bg-white/10 px-3 py-1.5 text-xs font-medium transition hover:bg-white/20"
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center text-sm text-neutral-400">
      {children}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-3 py-1 text-xs font-medium transition ${
        active ? "bg-white/15 text-white" : "text-neutral-400 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "good" | "warn";
}) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-neutral-500">{label}</dt>
      <dd
        className={
          tone === "warn" ? "text-amber-400" : tone === "good" ? "text-emerald-400" : "text-neutral-200"
        }
      >
        {value}
      </dd>
    </div>
  );
}
