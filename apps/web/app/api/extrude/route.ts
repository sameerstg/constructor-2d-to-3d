/**
 * Proxy: IR -> CV service -> GLB bytes.
 *
 * The viewer builds its own geometry locally for responsiveness, so this route
 * exists for the durable artefact: the GLB that gets handed to the Pico
 * headset. It is the server's watertight boolean-cut mesh, not the browser's
 * display approximation.
 */

import { NextRequest, NextResponse } from "next/server";

const CV_SERVICE_URL = process.env.CV_SERVICE_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  let ir: unknown;
  try {
    ir = await request.json();
  } catch {
    return NextResponse.json({ detail: "Expected a JSON floor plan IR." }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${CV_SERVICE_URL}/extrude`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(ir),
    });

    if (!upstream.ok) {
      const detail = await upstream.text();
      return new NextResponse(detail, {
        status: upstream.status,
        headers: { "content-type": "application/json" },
      });
    }

    return new NextResponse(await upstream.arrayBuffer(), {
      status: 200,
      headers: {
        "content-type": "model/gltf-binary",
        "content-disposition":
          upstream.headers.get("content-disposition") ?? 'attachment; filename="model.glb"',
      },
    });
  } catch {
    return NextResponse.json(
      { detail: `Could not reach the CV service at ${CV_SERVICE_URL}. Is it running?` },
      { status: 502 },
    );
  }
}
