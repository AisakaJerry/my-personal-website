"use client";

import { useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// 1-D Shallow Water Equations (SWE) — side-view water in a container
//
// State per column x:
//   h[x]  — water depth  (always ≥ 0)
//   u[x]  — depth-averaged horizontal velocity
//
// Update (explicit Euler, one step):
//   h_new[x]  = h[x]  - dt * (flux_right - flux_left) / dx
//   u_new[x]  = u[x]  - dt * (g * (h[x+1]-h[x-1])/(2*dx)   ← pressure gradient
//                            + u[x]*(u_right-u_left)/(2*dx)  ← advection
//                            + gx                            ← gravity body force
//                            + damping * u[x])               ← viscosity
//
// Flux = h[x] * u[x]  (volume flow rate per unit width)
//
// Boundary: reflective walls — u[0]=u[N-1]=0, h mirrors neighbor
// ─────────────────────────────────────────────────────────────────────────────

const N    = 256;    // number of columns
const DX   = 1.0;   // grid spacing (arbitrary units)
const DT   = 0.4;   // time step
const G    = 0.4;   // gravitational constant (controls wave speed)
const DAMP = 0.07;  // linear velocity damping

// Gyro
const GYRO_SCALE     = 0.012;  // body-force per unit of ax (tilt acceleration)
const GYRO_TILT_SCALE = 0.025; // visual slope offset (immediate feedback)
const GYRO_THRESHOLD  = 0.25;  // dead-zone for sensor noise

// Click / touch
const SPLASH_RADIUS = 0.07;  // fraction of grid width
const SPLASH_AMP    = 0.18;  // height displacement on click

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
    // Resting water level = 0.5 (half-full container).
    const H0 = 0.5 * N * DX;   // resting total depth in grid units

    const h    = new Float64Array(N).fill(H0 / N); // depth per column
    const u    = new Float64Array(N);               // velocity per column
    const hNew = new Float64Array(N);
    const uNew = new Float64Array(N);

    // surfBuf: screen-Y (0=top, 1=bottom) for each column, sent to GPU each frame
    const surfBuf = new Float32Array(N);

    // Smoothed gyro
    let smoothAx = 0;

    function applyForce(xFrac: number) {
      const xc  = xFrac * N;
      const sig = SPLASH_RADIUS * N;
      for (let x = 0; x < N; x++) {
        const dx = (x - xc) / sig;
        // Negative = push water down (stone drop) — displaces volume by raising neighbors
        h[x] = Math.max(0.001, h[x] - SPLASH_AMP * Math.exp(-0.5 * dx * dx));
      }
      // Conserve total volume: redistribute the removed water evenly
      const deficit = H0 - h.reduce((s, v) => s + v, 0);
      const add = deficit / N;
      for (let x = 0; x < N; x++) h[x] += add;
    }

    function stepSWE(gx: number) {
      // Lax-Friedrichs scheme — averages neighbors to suppress odd/even decoupling.
      // hNew[x] = 0.5*(h[x-1]+h[x+1]) - dt/(2dx) * (flux[x+1] - flux[x-1])
      // uNew[x] = 0.5*(u[x-1]+u[x+1]) - dt/(2dx) * (momentum terms)
      const restH = H0 / N;

      for (let x = 1; x < N - 1; x++) {
        const flux_l = h[x-1] * u[x-1];
        const flux_r = h[x+1] * u[x+1];

        // Continuity (Lax-Friedrichs)
        hNew[x] = 0.5 * (h[x-1] + h[x+1]) - 0.5 * DT / DX * (flux_r - flux_l);
        if (hNew[x] < 0.0001) hNew[x] = 0.0001;
        if (hNew[x] > restH * 3.5) hNew[x] = restH * 3.5;

        // Momentum (Lax-Friedrichs) with pressure gradient + body force + damping
        const dh_dx = (h[x+1] - h[x-1]) / (2 * DX);
        const uAvg  = 0.5 * (u[x-1] + u[x+1]);
        uNew[x] = uAvg + DT * (-G * dh_dx + gx - DAMP * u[x]);
        if (uNew[x] >  2.5) uNew[x] =  2.5;
        if (uNew[x] < -2.5) uNew[x] = -2.5;
      }

      // Reflective walls
      hNew[0]     = hNew[1];
      hNew[N - 1] = hNew[N - 2];
      uNew[0]     = 0;
      uNew[N - 1] = 0;

      // Extra damping near walls to prevent edge pile-up
      for (let i = 0; i < 8; i++) {
        const fade = (8 - i) / 8;
        uNew[i]         *= 1 - fade * 0.15;
        uNew[N - 1 - i] *= 1 - fade * 0.15;
      }

      h.set(hNew);
      u.set(uNew);

      // Spatial smoothing — blends each column with neighbors to kill high-frequency
      // checkerboard mode. σ=0.3 means 30% neighbor average, 70% self.
      const σ = 0.3;
      const h0 = h[0];
      for (let x = 1; x < N - 1; x++) {
        hNew[x] = (1 - σ) * h[x] + (σ / 2) * (h[x - 1] + h[x + 1]);
      }
      hNew[0]     = (1 - σ) * h[0]     + σ * h[1];
      hNew[N - 1] = (1 - σ) * h[N - 1] + σ * h[N - 2];
      h.set(hNew);

      // Enforce volume conservation — rescale h so total always equals H0
      let total = 0;
      for (let x = 0; x < N; x++) total += h[x];
      const scale = H0 / total;
      for (let x = 0; x < N; x++) h[x] *= scale;
    }

    function fillSurfBuf() {
      // h[x] is local water column height. Resting value = H0/N.
      // Deviation from rest: delta[x] = h[x] - H0/N
      // Map delta to screen offset: positive delta (more water) → surface rises → smaller Y.
      // Scale: DISP_SCALE controls how much 1 unit of delta shifts the surface on screen.
      const restH    = H0 / N;
      const DISP_SCALE = 0.8; // screen-fraction per unit of normalised deviation

      for (let x = 0; x < N; x++) {
        const delta = (h[x] - restH) / restH;   // normalised deviation from rest [-1..1]
        const tilt  = smoothAx * GYRO_TILT_SCALE * (x / N - 0.5);
        surfBuf[x]  = Math.max(0.05, Math.min(0.95,
          0.5 - delta * DISP_SCALE * 0.5 + tilt
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

        // Two SWE steps per frame for smoother propagation
        stepSWE(gx);
        stepSWE(gx);

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
