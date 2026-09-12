import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next blocks cross-origin requests for dev resources by default, and it
  // treats 127.0.0.1 and localhost as different origins. Without this, opening
  // the app on the loopback IP silently fails to load client chunks: the page
  // renders but never hydrates, so nothing is clickable and no error is shown.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
