/**
 * Proxy: browser -> Next -> CV service.
 *
 * The browser never talks to the Python service directly. In production that
 * service should not be publicly reachable at all: it decodes untrusted images
 * and runs heavy CPU work, so it belongs behind the web app's own auth,
 * rate limiting and request size caps rather than exposed on its own port.
 */

import { NextRequest, NextResponse } from "next/server";

const CV_SERVICE_URL = process.env.CV_SERVICE_URL ?? "http://127.0.0.1:8000";

// Recognition on a large scan is CPU-bound and genuinely slow. The default
// fetch timeout would abort a legitimate request.
const TIMEOUT_MS = 120_000;

export async function POST(request: NextRequest) {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ detail: "Expected multipart form data." }, { status: 400 });
  }

  const file = form.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return NextResponse.json({ detail: "No file was uploaded." }, { status: 400 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(`${CV_SERVICE_URL}/recognize`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });

    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    const aborted = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      {
        detail: aborted
          ? "Recognition timed out. The image may be very large - try a smaller export."
          : `Could not reach the CV service at ${CV_SERVICE_URL}. Is it running?`,
      },
      { status: aborted ? 504 : 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
