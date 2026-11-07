import { setupDevPlatform } from '@cloudflare/next-on-pages/next-dev';

// Here we use the @cloudflare/next-on-pages next-dev module to allow us to use bindings during local development
// (when running the application with `next dev`), for more information see:
// https://github.com/cloudflare/next-on-pages/blob/5712c57ea7/internal-packages/next-dev/README.md
if (process.env.NODE_ENV === 'development') {
  await setupDevPlatform();
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone static projects live in public/project/<slug>/index.html.
  // Serve them at the clean URL /project/<slug>.
  async rewrites() {
    return [
      { source: "/project/:slug", destination: "/project/:slug/index.html" },
    ];
  },
};

export default nextConfig;
