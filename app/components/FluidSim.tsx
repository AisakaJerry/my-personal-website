"use client";

import { useEffect, useRef } from "react";

// 1-D wave equation on a horizontal height field.
// surf[x] = screen-Y of the water surface at column x  (0 = top, 1 = bottom).
// Resting position: 0.5 (mid-screen).  Wave propagates left/right.
const N    = 512;
const C2   = 0.42;   // wave-speed²  — must be < 0.5 for stability
const DAMP = 0.993;  // per-step energy loss

// ── Fullscreen quad ────────────────────────────────────────────────────────
const VERT = `
@vertex
fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[i], 0., 1.);
}`;

// ── Side-view water renderer ───────────────────────────────────────────────
// surf[] holds surface Y for each column; uv.y < surf[x] → air (white),
// uv.y > surf[x] → water (blue gradient + caustics + specular).
const FRAG = `
@group(0) @binding(0) var<storage,read> surf: array<f32>;
@group(0) @binding(1) var<uniform>      dim:  vec4<f32>; // (w, h, 0, 0)

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv  = pos.xy / dim.xy;          // (0,0)=top-left (1,1)=bot-right
  let N   = i32(${N});

  // Bilinear interpolate surface height at this x column.
  let fx  = uv.x * f32(N - 1);
  let xi  = i32(floor(fx));
  let xf  = fract(fx);
  let sy  = mix(surf[clamp(xi, 0, N-1)], surf[clamp(xi+1, 0, N-1)], xf);

  // Signed distance from surface: positive = below = water.
  let d   = uv.y - sy;

  // Surface slope → screen-space normal (for specular).
  let hl  = surf[max(xi - 1, 0)];
  let hr  = surf[min(xi + 2, N - 1)];
  let slp = (hr - hl) * f32(N) * 0.25 * (dim.y / dim.x);
  let nrm = normalize(vec2(-slp, 1.0));
  let lit = normalize(vec2(0.3, -1.0));
  let spec= pow(max(dot(nrm, lit), 0.0), 80.0);

  // ── Air ──────────────────────────────────────────────────────────────────
  if (d < -0.004) {
    return vec4(1., 1., 1., 1.);
  }

  // ── Surface transition band ───────────────────────────────────────────────
  if (d < 0.004) {
    let t  = (d + 0.004) / 0.008;
    let sc = vec3(0.50, 0.79, 1.00) + spec * vec3(0.85, 0.93, 1.00);
    return vec4(mix(vec3(1.), clamp(sc, vec3(0.), vec3(1.)), t), 1.);
  }

  // ── Water body ────────────────────────────────────────────────────────────
  let shallow = vec3(0.17, 0.54, 0.96);
  let deep    = vec3(0.02, 0.10, 0.42);
  var wc      = mix(shallow, deep, clamp(d * 2.8, 0., 1.));

  // Subsurface light scattering near the surface.
  wc += exp(-d * 22.) * vec3(0.07, 0.11, 0.17);

  // Caustic shimmer.
  let caus = sin(pos.x * 0.055 + sy * 30.) * sin(pos.y * 0.08 - sy * 22.);
  wc += clamp(caus, 0., 1.) * exp(-d * 16.) * 0.07;

  return vec4(clamp(wc, vec3(0.), vec3(1.)), 1.);
}`;

type D = ReturnType<typeof Object.create>;

export default function FluidSim({
  onWaterline,
}: {
  onWaterline?: (surfaceY: number) => void;
}) {
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const onWlRef        = useRef(onWaterline);
  const ptrRef         = useRef({ x: 0.5, down: false });
  const gyrRef         = useRef({ ax: 0 });

  // Keep callback ref fresh without re-running the effect.
  useEffect(() => { onWlRef.current = onWaterline; }, [onWaterline]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    let raf: number, stopped = false;

    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;

    // ── CPU wave buffers ─────────────────────────────────────────────────────
    const prev = new Float32Array(N).fill(0.5);
    const cur  = new Float32Array(N).fill(0.5);
    const nxt  = new Float32Array(N);

    function splat(x: number, str: number, r: number) {
      for (let i = 0; i < N; i++) {
        const dx = i / N - x;
        cur[i] = Math.max(0.05, Math.min(0.95, cur[i] + str * Math.exp(-(dx * dx) / (r * r))));
      }
    }

    function stepWave() {
      for (let x = 0; x < N; x++) {
        const l = cur[x > 0     ? x - 1 : 0];
        const r = cur[x < N - 1 ? x + 1 : N - 1];
        nxt[x] = Math.max(0.05, Math.min(0.95,
          DAMP * (2 * cur[x] - prev[x] + C2 * (l - 2 * cur[x] + r))
        ));
      }
      prev.set(cur);
      cur.set(nxt);
    }

    // ── WebGPU setup ─────────────────────────────────────────────────────────
    async function run() {
      const nav = navigator as D;
      if (!nav.gpu) return;
      const adapter: D = await nav.gpu.requestAdapter();
      if (!adapter) return;
      const dev: D = await adapter.requestDevice();
      const q: D   = dev.queue;
      const ctx: D = canvas.getContext("webgpu");
      const fmt: string = nav.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: dev, format: fmt, alphaMode: "opaque" });

      const surfBuf = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const dimBuf  = dev.createBuffer({ size: 16,    usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
      q.writeBuffer(dimBuf, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));

      const renPL = dev.createRenderPipeline({
        layout:    "auto",
        vertex:    { module: dev.createShaderModule({ code: VERT }), entryPoint: "main" },
        fragment:  { module: dev.createShaderModule({ code: FRAG }), entryPoint: "main", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-strip" },
      });

      // Seed a couple of gentle startup ripples.
      splat(0.30, 0.055, 0.040);
      splat(0.72, 0.045, 0.035);

      function frame() {
        if (stopped) return;

        const ptr = ptrRef.current;
        const gyr = gyrRef.current;
        if (ptr.down) splat(ptr.x, 0.05, 0.025);
        // Gyro: tilt left/right → wave on that side
        if (Math.abs(gyr.ax) > 0.6) splat(0.5 + Math.sign(gyr.ax) * 0.4, 0.04, 0.05);

        stepWave();

        // Report centre waterline Y to page for text-colour updates.
        onWlRef.current?.(cur[Math.floor(N / 2)]);

        q.writeBuffer(surfBuf, 0, cur);

        const enc = dev.createCommandEncoder();
        const bg  = dev.createBindGroup({
          layout: renPL.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: surfBuf } },
            { binding: 1, resource: { buffer: dimBuf  } },
          ],
        });
        const rp = enc.beginRenderPass({ colorAttachments: [{
          view:       ctx.getCurrentTexture().createView(),
          clearValue: { r:1, g:1, b:1, a:1 },
          loadOp:     "clear",
          storeOp:    "store",
        }]});
        rp.setPipeline(renPL);
        rp.setBindGroup(0, bg);
        rp.draw(4);
        rp.end();
        q.submit([enc.finish()]);

        raf = requestAnimationFrame(frame);
      }

      frame();
    }

    run().catch(console.error);

    // ── Input ────────────────────────────────────────────────────────────────
    const c  = canvas;
    const rc = () => c.getBoundingClientRect();
    const nx = (e: MouseEvent | Touch) => (e.clientX - rc().left) / rc().width;

    const onDown  = (e: MouseEvent)  => { ptrRef.current = { x: nx(e),          down: true }; };
    const onMove  = (e: MouseEvent)  => { if (ptrRef.current.down) ptrRef.current.x = nx(e); };
    const onUp    = ()               => { ptrRef.current.down = false; };
    const onTDown = (e: TouchEvent)  => { e.preventDefault(); ptrRef.current = { x: nx(e.touches[0]), down: true }; };
    const onTMove = (e: TouchEvent)  => { e.preventDefault(); if (ptrRef.current.down) ptrRef.current.x = nx(e.touches[0]); };
    const onTUp   = ()               => { ptrRef.current.down = false; };
    const onMot   = (e: DeviceMotionEvent) => { gyrRef.current.ax = e.accelerationIncludingGravity?.x ?? 0; };

    c.addEventListener("mousedown",  onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    c.addEventListener("touchstart", onTDown, { passive: false });
    c.addEventListener("touchmove",  onTMove, { passive: false });
    c.addEventListener("touchend",   onTUp);
    window.addEventListener("devicemotion", onMot);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      c.removeEventListener("mousedown",  onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
      c.removeEventListener("touchstart", onTDown);
      c.removeEventListener("touchmove",  onTMove);
      c.removeEventListener("touchend",   onTUp);
      window.removeEventListener("devicemotion", onMot);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full block"
      style={{ cursor: "crosshair" }}
    />
  );
}
