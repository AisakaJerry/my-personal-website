"use client";

import { useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// 1-D Shallow Water Equations — side-view water in a container
//
// Staggered (Arakawa-C) grid:
//   h[i]  — water depth at cell centers, i = 0..N-1   (rest depth = 1)
//   u[i]  — velocity at cell faces,      i = 0..N     (u[0] = u[N] = 0 → walls)
//
// Update (explicit, SUBSTEPS per frame):
//   u[i] += dt * (-G * (h[i]-h[i-1])/dx + gx - DAMP*u[i]) + viscosity
//   h[i] -= dt * (F[i+1] - F[i]) / dx,   F[face] = u * h_upwind
//
// The staggered grid avoids the odd/even decoupling and heavy numerical
// diffusion of collocated Lax-Friedrichs — waves actually propagate.
// Flux form + zero wall flux conserves total volume exactly.
// ─────────────────────────────────────────────────────────────────────────────

const N        = 256;   // number of columns
const DX       = 1.0;   // grid spacing
const DT       = 0.12;  // sub-step time step
const SUBSTEPS = 5;     // sub-steps per frame (wave crosses screen in ~2.3 s)
const G        = 9.8;   // gravity → wave speed c = sqrt(G*h) ≈ 3.1 cells/unit-time
const DAMP     = 0.06;  // linear velocity damping (per unit time)
const NU       = 0.3;   // velocity viscosity (kills grid-scale jaggies)
const U_MAX    = 2.5;   // safety clamp, keeps CFL = (|u|+c)*DT/DX < 1

// Gyro
const GYRO_SCALE      = 0.004; // body-force per unit of ax (full tilt ≈ half-depth slosh)
const GYRO_TILT_SCALE = 0.025; // visual slope offset (immediate feedback)
const GYRO_THRESHOLD  = 0.25;  // dead-zone for sensor noise

// Click / touch — a volume-neutral depression applied while pointer is down.
const SPLASH_RADIUS = 0.05; // Gaussian sigma as fraction of grid width
const SPLASH_AMP    = 0.06; // depth removed per frame at the center (rest depth = 1)
const SPLASH_FLOOR  = 0.35; // stop pushing once the column is this shallow

// ── Fullscreen quad ───────────────────────────────────────────────────────────
const VERT = `
@vertex
fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[i], 0., 1.);
}`;

// ── Side-view renderer ─────────────────────────────────────────────────────────
// surf[x] = screen-Y of the water surface (0=top, 1=bottom).
// Air pixels → alpha 0 (CSS bg shows through). Water pixels → opaque blue.
const FRAG = `
@group(0) @binding(0) var<storage,read> surf: array<f32>;
@group(0) @binding(1) var<uniform>      dim:  vec4<f32>;

fn sampleSurf(uv_x: f32) -> f32 {
  let fx = uv_x * f32(${N} - 1);
  let xi = i32(floor(fx));
  let xf = fract(fx);
  return mix(surf[clamp(xi,   0, ${N}-1)],
             surf[clamp(xi+1, 0, ${N}-1)], xf);
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv  = pos.xy / dim.xy;
  let sy  = sampleSurf(uv.x);
  let d   = uv.y - sy;

  // Surface slope → specular normal
  let e   = 1.5 / f32(${N});
  let sl  = (sampleSurf(clamp(uv.x+e,0.,1.)) - sampleSurf(clamp(uv.x-e,0.,1.))) / (2.*e) * (dim.y/dim.x) * 0.4;
  let nrm = normalize(vec2(-sl, 1.));
  let lit = normalize(vec2(0.3, -1.));
  let spec = pow(max(dot(nrm, lit), 0.), 80.);

  // Air → transparent
  if (d < -0.003) { return vec4(0.,0.,0.,0.); }

  // Surface band
  if (d < 0.003) {
    let t  = (d + 0.003) / 0.006;
    let sc = vec3(0.48, 0.76, 1.00) + spec * 0.9;
    return vec4(clamp(sc, vec3(0.), vec3(1.)), t);
  }

  // Water body
  let shallow = vec3(0.18, 0.55, 0.97);
  let deep    = vec3(0.02, 0.10, 0.44);
  var wc      = mix(shallow, deep, clamp(d * 3.0, 0., 1.));
  wc += exp(-d * 20.) * vec3(0.06, 0.10, 0.16);
  let caus = sin(pos.x*0.06 + sy*28.) * sin(pos.y*0.09 - sy*20.);
  wc += clamp(caus, 0., 1.) * exp(-d*14.) * 0.06;
  return vec4(clamp(wc, vec3(0.), vec3(1.)), 1.);
}`;

type D = ReturnType<typeof Object.create>;

export default function FluidSim({
  onWaterline,
}: {
  onWaterline?: (surfaceY: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onWlRef   = useRef(onWaterline);
  const ptrRef    = useRef({ x: 0.5, down: false });
  const gyrRef    = useRef({ ax: 0 });

  useEffect(() => { onWlRef.current = onWaterline; }, [onWaterline]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    let raf: number, stopped = false;

    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;

    // ── SWE state ─────────────────────────────────────────────────────────────
    const REST_H = 1.0;               // rest depth (normalised)
    const H0     = REST_H * N;        // total volume

    const h    = new Float64Array(N).fill(REST_H); // depth at cell centers
    const u    = new Float64Array(N + 1);          // velocity at cell faces
    const hNew = new Float64Array(N);
    const uNew = new Float64Array(N + 1);

    // surfBuf: screen-Y (0=top, 1=bottom) for each column, sent to GPU each frame
    const surfBuf = new Float32Array(N);

    // Smoothed gyro
    let smoothAx = 0;

    function applyForce(xFrac: number) {
      const xc  = xFrac * N;
      const sig = SPLASH_RADIUS * N;

      // Fade the push out as the column gets shallow — prevents carving the
      // water down to nothing while the pointer is held.
      const hc  = h[Math.max(0, Math.min(N - 1, Math.round(xc)))];
      const eff = SPLASH_AMP * Math.max(0, Math.min(1, (hc - SPLASH_FLOOR) / 0.5));
      if (eff <= 0) return;

      // Volume-neutral shape: narrow Gaussian down, twice-as-wide Gaussian up
      // (half amplitude → equal integrals), so displaced water piles up beside
      // the pit instead of being teleported across the whole tank.
      for (let x = 0; x < N; x++) {
        const d = x - xc;
        const narrow = Math.exp(-0.5 * (d / sig) * (d / sig));
        const wide   = Math.exp(-0.5 * (d / (2 * sig)) * (d / (2 * sig)));
        h[x] = Math.max(0.02, h[x] - eff * (narrow - 0.5 * wide));
      }
      // The min-clamp above can leak a little volume — put it back uniformly.
      let total = 0;
      for (let x = 0; x < N; x++) total += h[x];
      const add = (H0 - total) / N;
      for (let x = 0; x < N; x++) h[x] += add;
    }

    function stepSWE(gx: number) {
      // Momentum on faces: pressure gradient + body force + damping + viscosity
      for (let i = 1; i < N; i++) {
        const dhdx = (h[i] - h[i - 1]) / DX;
        const visc = NU * (u[i + 1] - 2 * u[i] + u[i - 1]) / (DX * DX);
        let v = u[i] + DT * (-G * dhdx + gx - DAMP * u[i] + visc);
        if (v >  U_MAX) v =  U_MAX;
        if (v < -U_MAX) v = -U_MAX;
        uNew[i] = v;
      }
      uNew[0] = 0;
      uNew[N] = 0;

      // Continuity: upwind flux through each face
      for (let i = 0; i < N; i++) {
        const ul = uNew[i];
        const ur = uNew[i + 1];
        const fl = ul > 0 ? (i > 0     ? h[i - 1] : h[i]) * ul : h[i] * ul;
        const fr = ur > 0 ? h[i] * ur : (i < N - 1 ? h[i + 1] : h[i]) * ur;
        hNew[i] = Math.max(0.02, h[i] - DT / DX * (fr - fl));
      }

      h.set(hNew);
      u.set(uNew);
    }

    function fillSurfBuf() {
      // delta = deviation from rest depth. tanh gives a soft ceiling so big
      // sloshes compress smoothly instead of slamming into a hard clamp.
      const DISP = 0.28; // max screen-fraction the surface moves from center

      for (let x = 0; x < N; x++) {
        const delta = h[x] - REST_H;
        const tilt  = smoothAx * GYRO_TILT_SCALE * (x / N - 0.5);
        surfBuf[x]  = Math.max(0.03, Math.min(0.97,
          0.5 - DISP * Math.tanh(delta * 1.6) + tilt
        ));
      }
    }

    // ── WebGPU ────────────────────────────────────────────────────────────────
    async function run() {
      const nav = navigator as D;
      if (!nav.gpu) return;
      const adapter: D = await nav.gpu.requestAdapter();
      if (!adapter) return;
      const dev: D = await adapter.requestDevice();
      const q: D   = dev.queue;
      const ctx: D = canvas.getContext("webgpu");
      const fmt: string = nav.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: dev, format: fmt, alphaMode: "premultiplied" });

      const surfGPU = dev.createBuffer({ size: N * 4,  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const dimBuf  = dev.createBuffer({ size: 16,     usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
      q.writeBuffer(dimBuf, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));

      const renPL = dev.createRenderPipeline({
        layout:    "auto",
        vertex:    { module: dev.createShaderModule({ code: VERT }), entryPoint: "main" },
        fragment:  { module: dev.createShaderModule({ code: FRAG }), entryPoint: "main", targets: [{ format: fmt, blend: { color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] },
        primitive: { topology: "triangle-strip" },
      });

      function frame() {
        if (stopped) return;

        const ptr = ptrRef.current;
        const gyr = gyrRef.current;

        smoothAx += 0.25 * (gyr.ax - smoothAx);
        const gx = Math.abs(smoothAx) < GYRO_THRESHOLD ? 0 : smoothAx * GYRO_SCALE;

        if (ptr.down) applyForce(ptr.x);

        for (let s = 0; s < SUBSTEPS; s++) stepSWE(gx);

        fillSurfBuf();
        onWlRef.current?.(surfBuf[Math.floor(N / 2)]);

        q.writeBuffer(surfGPU, 0, surfBuf);

        const enc = dev.createCommandEncoder();
        const bg  = dev.createBindGroup({
          layout: renPL.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: surfGPU } },
            { binding: 1, resource: { buffer: dimBuf  } },
          ],
        });
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0,g:0,b:0,a:0 }, loadOp:"clear", storeOp:"store" }] });
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

    // ── Input ─────────────────────────────────────────────────────────────────
    const c  = canvas;
    const rc = () => c.getBoundingClientRect();
    const nx = (e: MouseEvent | Touch) => (e.clientX - rc().left) / rc().width;

    const onDown   = (e: MouseEvent)  => { ptrRef.current = { x: nx(e), down: true }; };
    const onMove   = (e: MouseEvent)  => { if (ptrRef.current.down) ptrRef.current.x = nx(e); };
    const onUp     = ()               => { ptrRef.current.down = false; };
    const onTDown  = (e: TouchEvent)  => { e.preventDefault(); ptrRef.current = { x: nx(e.touches[0]), down: true }; };
    const onTMove  = (e: TouchEvent)  => { e.preventDefault(); if (ptrRef.current.down) ptrRef.current.x = nx(e.touches[0]); };
    const onTUp    = ()               => { ptrRef.current.down = false; };
    const onMot    = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      if (!a) return;
      const ax = a.x ?? 0, ay = a.y ?? 0;
      gyrRef.current.ax = Math.abs(ax) >= Math.abs(ay) ? ax : ay;
    };
    const onOrient = (e: DeviceOrientationEvent) => {
      gyrRef.current.ax = (e.gamma ?? 0) * 0.3;
    };

    c.addEventListener("mousedown",       onDown);
    window.addEventListener("mousemove",  onMove);
    window.addEventListener("mouseup",    onUp);
    c.addEventListener("touchstart",      onTDown,  { passive: false });
    c.addEventListener("touchmove",       onTMove,  { passive: false });
    c.addEventListener("touchend",        onTUp);
    window.addEventListener("devicemotion",      onMot);
    window.addEventListener("deviceorientation", onOrient);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      c.removeEventListener("mousedown",      onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
      c.removeEventListener("touchstart",     onTDown);
      c.removeEventListener("touchmove",      onTMove);
      c.removeEventListener("touchend",       onTUp);
      window.removeEventListener("devicemotion",      onMot);
      window.removeEventListener("deviceorientation", onOrient);
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
