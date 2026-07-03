"use client";

import { useEffect, useRef, useState } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// 1-D Shallow Water Equations — side-view water in a container
//
// Staggered (Arakawa-C) grid:
//   h[i]  — water depth at cell centers, i = 0..N-1   (rest depth = 1)
//   u[i]  — velocity at cell faces,      i = 0..N     (u[0] = u[N] = 0 → walls)
//
// Update (explicit, wall-clock-paced sub-steps):
//   u[i] += dt * (-Geff * (h[i]-h[i-1])/dx + gx - DAMP*u[i]) + viscosity
//   h[i] -= dt * (F[i+1] - F[i]) / dx,   F[face] = u * h_upwind
//
// Tilt physics: with the screen tilted, only the gravity component along
// screen-y (ay) flattens the surface, while the component along screen-x (ax)
// drives water sideways. Geff = G*|ay| and the body force ∝ ax, so the
// equilibrium surface slope is ∝ ax/|ay| = tan(tilt) — the surface stays
// parallel to the real-world horizon. When gravity points toward the screen
// top (device upside down), the whole sim is mirrored 180° and water fills
// from the top of the screen.
//
// The staggered grid avoids the odd/even decoupling and heavy numerical
// diffusion of collocated Lax-Friedrichs — waves actually propagate.
// Flux form + zero wall flux conserves total volume exactly.
// ─────────────────────────────────────────────────────────────────────────────

const N        = 256;   // number of columns
const DX       = 1.0;   // grid spacing
const DT       = 0.12;  // sub-step time step (sim-time units)
const SIM_RATE = 36;    // sim-time units per real second (frame-rate independent)
const MAX_STEP = 15;    // max sub-steps per frame (cap catch-up after jank)
const G        = 9.8;   // gravity → wave speed c = sqrt(G*h) ≈ 3.1 cells/unit-time
const G_MIN_AY = 0.05;  // floor for |ay| so the pressure term never vanishes
const DAMP     = 0.06;  // linear velocity damping (per unit time)
const NU       = 0.3;   // velocity viscosity (kills grid-scale jaggies)
const U_MAX    = 2.5;   // safety clamp, keeps CFL = (|u|+c)*DT/DX < 1

// Display mapping: depth deviation → screen fraction (soft tanh ceiling)
const DISP   = 0.45;
const TANH_K = 1.5;

// Gyro — gyr holds the normalized gravity direction in the screen frame:
//   ax ∈ [-1,1] along screen-x (+ = right), ay ∈ [-1,1] along screen-y (+ = down)
const GYRO_TILT_SCALE = 0.08; // visual slope offset (immediate feedback)
const GYRO_THRESHOLD  = 0.03; // dead-zone for sensor noise (~2° of tilt)
const FLIP_HYST       = 0.2;  // |ay| hysteresis before mirroring upside down

// Click / touch — a volume-neutral depression applied while pointer is down.
const SPLASH_RADIUS = 0.05; // Gaussian sigma as fraction of grid width
const SPLASH_AMP    = 0.06; // depth removed per 0.6 sim-time at the center
const SPLASH_FLOOR  = 0.35; // stop pushing once the column is this shallow

// ── Fullscreen quad ───────────────────────────────────────────────────────────
const VERT = `
@vertex
fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[i], 0., 1.);
}`;

// ── Side-view renderer ─────────────────────────────────────────────────────────
// surf[x] = screen-Y of the water surface (0=top, 1=bottom) in the sim frame.
// dim = (width, height, flip, 0); flip=1 mirrors the scene 180° so water
// hangs from the top of the screen (device upside down).
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
  let raw = pos.xy / dim.xy;
  let uv  = mix(raw, vec2(1.) - raw, dim.z);
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
  onWaterline?: (surfaceY: number, flipped: boolean) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onWlRef   = useRef(onWaterline);
  const ptrRef    = useRef({ x: 0.5, down: false });
  const gyrRef    = useRef({ ax: 0, ay: 1 }); // assume upright until sensors speak
  const [needsGyroPerm, setNeedsGyroPerm] = useState(false);
  const askGyroRef = useRef<() => void>(() => {});

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

    // Smoothed gyro (screen frame) + upside-down mirroring state
    let smoothAx = 0;
    let smoothAy = 1;
    let flipped  = false;

    // Body force that makes the equilibrium surface slope equal the real
    // tilt angle on screen: solves DISP*TANH_K*N * (force/G) = aspect (see
    // fillSurfBuf mapping). Recomputed on resize since it depends on aspect.
    let tiltForce = 0;
    const computeTiltForce = () => {
      tiltForce = (G * (canvas.width / canvas.height)) / (DISP * TANH_K * N);
    };
    computeTiltForce();

    function applyForce(xFrac: number, amp: number) {
      const xc  = xFrac * N;
      const sig = SPLASH_RADIUS * N;

      // Fade the push out as the column gets shallow — prevents carving the
      // water down to nothing while the pointer is held.
      const hc  = h[Math.max(0, Math.min(N - 1, Math.round(xc)))];
      const eff = amp * Math.max(0, Math.min(1, (hc - SPLASH_FLOOR) / 0.5));
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

    function stepSWE(gx: number, gEff: number) {
      // Momentum on faces: pressure gradient + body force + damping + viscosity
      for (let i = 1; i < N; i++) {
        const dhdx = (h[i] - h[i - 1]) / DX;
        const visc = NU * (u[i + 1] - 2 * u[i] + u[i - 1]) / (DX * DX);
        let v = u[i] + DT * (-gEff * dhdx + gx - DAMP * u[i] + visc);
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

    // Mirror the sim 180° (upside-down transition) without a visual jump:
    // reverse columns and negate velocities.
    function mirrorState() {
      for (let i = 0; i < N; i++) hNew[i] = h[N - 1 - i];
      h.set(hNew);
      for (let i = 0; i <= N; i++) uNew[i] = -u[N - i];
      u.set(uNew);
    }

    function fillSurfBuf(axSim: number) {
      // delta = deviation from rest depth. tanh gives a soft ceiling so big
      // sloshes compress smoothly instead of slamming into a hard clamp.
      for (let x = 0; x < N; x++) {
        const delta = h[x] - REST_H;
        const tilt  = axSim * GYRO_TILT_SCALE * (x / N - 0.5);
        surfBuf[x]  = Math.max(0.03, Math.min(0.97,
          0.5 - DISP * Math.tanh(delta * TANH_K) + tilt
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
      const dimArr  = new Float32Array(4);

      const renPL = dev.createRenderPipeline({
        layout:    "auto",
        vertex:    { module: dev.createShaderModule({ code: VERT }), entryPoint: "main" },
        fragment:  { module: dev.createShaderModule({ code: FRAG }), entryPoint: "main", targets: [{ format: fmt, blend: { color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } } }] },
        primitive: { topology: "triangle-strip" },
      });

      let lastT = performance.now();
      let acc   = 0;

      function frame(nowT: number) {
        if (stopped) return;

        const ptr = ptrRef.current;
        const gyr = gyrRef.current;

        // Wall-clock pacing: same water speed at 60 Hz and 120 Hz.
        const elapsed = Math.min((nowT - lastT) / 1000, 0.1);
        lastT = nowT;
        acc += elapsed * SIM_RATE;
        let steps = Math.floor(acc / DT);
        if (steps > MAX_STEP) { steps = MAX_STEP; acc = 0; } else { acc -= steps * DT; }

        smoothAx += 0.25 * (gyr.ax - smoothAx);
        smoothAy += 0.25 * (gyr.ay - smoothAy);

        // Upside-down mirroring with hysteresis around ay = 0.
        if (!flipped && smoothAy < -FLIP_HYST) { flipped = true;  mirrorState(); }
        if ( flipped && smoothAy >  FLIP_HYST) { flipped = false; mirrorState(); }

        const axSim = flipped ? -smoothAx : smoothAx;
        const gx    = Math.abs(axSim) < GYRO_THRESHOLD ? 0 : axSim * tiltForce;
        const gEff  = G * Math.max(Math.abs(smoothAy), G_MIN_AY);

        if (ptr.down && steps > 0) {
          const xSim = flipped ? 1 - ptr.x : ptr.x;
          applyForce(xSim, SPLASH_AMP * (steps * DT) / 0.6);
        }

        for (let s = 0; s < steps; s++) stepSWE(gx, gEff);

        fillSurfBuf(axSim);
        const midY = surfBuf[Math.floor(N / 2)];
        onWlRef.current?.(flipped ? 1 - midY : midY, flipped);

        q.writeBuffer(surfGPU, 0, surfBuf);
        dimArr[0] = canvas.width;
        dimArr[1] = canvas.height;
        dimArr[2] = flipped ? 1 : 0;
        q.writeBuffer(dimBuf, 0, dimArr);

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

      raf = requestAnimationFrame(frame);
    }

    run().catch(console.error);

    // ── Input ─────────────────────────────────────────────────────────────────
    const c  = canvas;
    const rc = () => c.getBoundingClientRect();
    const nx = (e: MouseEvent | Touch) => (e.clientX - rc().left) / rc().width;

    // iOS 13+ only delivers orientation events after an explicit permission
    // request, and only from a user gesture in a secure (HTTPS) context.
    const DOE = (window as D).DeviceOrientationEvent;
    const needsPerm = DOE && typeof DOE.requestPermission === "function";
    if (needsPerm) setNeedsGyroPerm(true);
    const askGyroPermission = () => {
      if (!needsPerm) return;
      DOE.requestPermission()
        .then((res: string) => { if (res === "granted") setNeedsGyroPerm(false); })
        .catch(() => {});
    };
    askGyroRef.current = askGyroPermission;

    const onDown   = (e: MouseEvent)  => { askGyroPermission(); ptrRef.current = { x: nx(e), down: true }; };
    const onMove   = (e: MouseEvent)  => { if (ptrRef.current.down) ptrRef.current.x = nx(e); };
    const onUp     = ()               => { ptrRef.current.down = false; };
    const onTDown  = (e: TouchEvent)  => { e.preventDefault(); askGyroPermission(); ptrRef.current = { x: nx(e.touches[0]), down: true }; };
    const onTMove  = (e: TouchEvent)  => { e.preventDefault(); if (ptrRef.current.down) ptrRef.current.x = nx(e.touches[0]); };
    const onTUp    = ()               => { ptrRef.current.down = false; };

    // Single source of truth for tilt. Raw gamma is NOT gravity-along-screen-x:
    // reconstruct the gravity direction in device coordinates from the spec's
    // Z-X'-Y'' Euler angles, then rotate into the screen frame so landscape
    // and upside-down orientations behave correctly.
    //   g_device = (sin γ · cos β,  −sin β,  −cos γ · cos β)   (unit, points down)
    const onOrient = (e: DeviceOrientationEvent) => {
      if (e.beta == null || e.gamma == null) return;
      setNeedsGyroPerm(false); // events flowing → permission already granted
      const b  = (e.beta  * Math.PI) / 180;
      const g  = (e.gamma * Math.PI) / 180;
      const gxDev = Math.sin(g) * Math.cos(b);
      const gyDev = -Math.sin(b);
      const scr = window.screen as D;
      const angleDeg: number =
        (scr.orientation && typeof scr.orientation.angle === "number"
          ? scr.orientation.angle
          : (window as D).orientation) ?? 0;
      const a = (angleDeg * Math.PI) / 180;
      // screen-x (right) and screen-y (down) components of gravity
      gyrRef.current.ax = gxDev * Math.cos(a) + gyDev * Math.sin(a);
      gyrRef.current.ay = gxDev * Math.sin(a) - gyDev * Math.cos(a);
    };

    const onResize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      computeTiltForce();
    };

    c.addEventListener("mousedown",       onDown);
    window.addEventListener("mousemove",  onMove);
    window.addEventListener("mouseup",    onUp);
    c.addEventListener("touchstart",      onTDown,  { passive: false });
    c.addEventListener("touchmove",       onTMove,  { passive: false });
    c.addEventListener("touchend",        onTUp);
    window.addEventListener("deviceorientation", onOrient);
    window.addEventListener("resize",           onResize);
    window.addEventListener("orientationchange", onResize);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      c.removeEventListener("mousedown",      onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
      c.removeEventListener("touchstart",     onTDown);
      c.removeEventListener("touchmove",      onTMove);
      c.removeEventListener("touchend",       onTUp);
      window.removeEventListener("deviceorientation", onOrient);
      window.removeEventListener("resize",           onResize);
      window.removeEventListener("orientationchange", onResize);
    };
  }, []);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        style={{ cursor: "crosshair" }}
      />
      {needsGyroPerm && (
        <button
          onClick={() => askGyroRef.current()}
          className="absolute bottom-4 right-4 z-20 rounded-full px-4 py-2 text-sm
                     bg-sky-500/80 text-white shadow-lg backdrop-blur-sm"
        >
          💧 Enable tilt
        </button>
      )}
    </>
  );
}
