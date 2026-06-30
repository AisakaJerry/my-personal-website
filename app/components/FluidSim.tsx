"use client";

import { useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Fractional-derivative wave simulation — ported from wavesimulation.py
//
// Physics: Grünwald-Letnikov (GL) fractional wave equation
//   u[n+1] = ( (2 - w[1]*F) * u[n]  -  u[n-1]  -  F * Σ_{k=2}^{MEM} w[k]*u[n+1-k]
//             + r² * ∇²u[n] )  /  (1 + w[0]*F)
//
// where  F = η * dt^(2-α),   r² = (c*dt/dy)²
// GL weights: w[0]=1,  w[k] = w[k-1] * (1 - (α+1)/k)
//
// Two display lines:
//   near  = u_hist[n]          (current frame, full amplitude)
//   far   = u_hist[n - LAG]    (lagged, scaled + offset → pseudo-3D depth)
// ─────────────────────────────────────────────────────────────────────────────

const N_COLS  = 512;   // spatial grid points (columns)
const DY      = 0.1;
const DT      = 0.05;
const C       = 2.0;
const ETA     = 0.05;
const ALPHA   = 1.3;
const MEM     = 300;   // GL memory truncation window
const NT_MAX  = 20000; // hard cap on total steps

// Pseudo-3D far-wave parameters (matches Python)
const DEPTH_Y     = 0.08;  // vertical offset for far wave (fraction of viewport)
const DEPTH_SCALE = 0.55;  // amplitude scale for far wave
const LAG         = 15;    // frame lag between near and far line

// Force impulse params
const FORCE_SIGMA   = 0.06;  // fraction of grid width
const FORCE_AMP     = 0.008; // positive → surface dips down (stone-drop depression)
const STEPS_PER_FRAME = 3;   // physics steps per render frame → faster wave motion

// ── Fullscreen quad ───────────────────────────────────────────────────────────
const VERT = `
@vertex
fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[i], 0., 1.);
}`;

// ── Side-view renderer ────────────────────────────────────────────────────────
// Receives two surface arrays (near + far) and canvas dimensions.
// near[x] and far[x] are normalised Y values (0=top, 1=bottom), rest ≈ 0.5.
const FRAG = `
@group(0) @binding(0) var<storage,read> near: array<f32>;
@group(0) @binding(1) var<storage,read> far:  array<f32>;
@group(0) @binding(2) var<uniform>      dim:  vec4<f32>;

fn sampleLine(buf: ptr<storage,array<f32>,read>, uv_x: f32) -> f32 {
  let fx = uv_x * f32(${N_COLS} - 1);
  let xi = i32(floor(fx));
  let xf = fract(fx);
  let a  = buf[clamp(xi,   0, ${N_COLS}-1)];
  let b  = buf[clamp(xi+1, 0, ${N_COLS}-1)];
  return mix(a, b, xf);
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv  = pos.xy / dim.xy;   // (0,0)=top-left (1,1)=bottom-right
  let sy_near = sampleLine(&near, uv.x);
  let sy_far  = sampleLine(&far,  uv.x);

  // ── Surface slope → normal for specular (near wave) ─────────────────────
  let e    = 1.5 / f32(${N_COLS});
  let hn_l = sampleLine(&near, clamp(uv.x - e, 0.0, 1.0));
  let hn_r = sampleLine(&near, clamp(uv.x + e, 0.0, 1.0));
  let slp  = (hn_r - hn_l) / (2.0 * e) * (dim.y / dim.x) * 0.5;
  let nrm  = normalize(vec2(-slp, 1.0));
  let lit  = normalize(vec2(0.3, -1.0));
  let spec = pow(max(dot(nrm, lit), 0.0), 90.0);

  let d_near = uv.y - sy_near;
  let d_far  = uv.y - sy_far;

  // ── Air zone (above near wave) ───────────────────────────────────────────
  if (d_near < -0.003) {
    return vec4(1., 1., 1., 1.);
  }

  // ── Near surface transition band ─────────────────────────────────────────
  if (d_near < 0.003) {
    let t  = (d_near + 0.003) / 0.006;
    let sc = vec3(0.48, 0.76, 1.00) + spec * 0.9;
    return vec4(mix(vec3(1.), clamp(vec3(sc), vec3(0.), vec3(1.)), t), 1.);
  }

  // ── Water body ────────────────────────────────────────────────────────────
  let shallow = vec3(0.18, 0.55, 0.97);
  let deep    = vec3(0.02, 0.10, 0.44);
  var wc      = mix(shallow, deep, clamp(d_near * 3.0, 0., 1.));

  // Subsurface scatter
  wc += exp(-d_near * 20.) * vec3(0.06, 0.10, 0.16);

  // Caustic shimmer
  let caus = sin(pos.x * 0.06 + sy_near * 28.) * sin(pos.y * 0.09 - sy_near * 20.);
  wc += clamp(caus, 0., 1.) * exp(-d_near * 14.) * 0.06;

  // ── Far wave line overlay (pseudo-3D depth) ───────────────────────────────
  // Draw it as a thin bright line inside the water body.
  let dist_far = abs(uv.y - sy_far);
  let far_glow = exp(-dist_far * dist_far * 120000.0) * 0.55;
  wc = mix(wc, vec3(0.62, 0.88, 1.0), far_glow * clamp(d_far + 0.02, 0., 1.));

  return vec4(clamp(wc, vec3(0.), vec3(1.)), 1.);
}`;

type D = ReturnType<typeof Object.create>;

// ── Precompute GL weights (w[0..MEM+1]) ──────────────────────────────────────
function buildGLWeights(): Float64Array {
  const w = new Float64Array(MEM + 2);
  w[0] = 1.0;
  for (let k = 1; k < MEM + 2; k++) {
    w[k] = (1.0 - (ALPHA + 1.0) / k) * w[k - 1];
  }
  return w;
}

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

    // ── Simulation state ──────────────────────────────────────────────────────
    const glWeights = buildGLWeights();

    // u_hist[n][x] stores displacement (centred at 0, rest = 0).
    // We allocate a rolling circular buffer of depth MEM+2 to avoid NT_MAX×N_COLS RAM.
    const HIST_DEPTH = MEM + 4;
    const uHist   = new Array<Float64Array>(HIST_DEPTH);
    for (let i = 0; i < HIST_DEPTH; i++) uHist[i] = new Float64Array(N_COLS);

    const rSq             = (C * DT / DY) ** 2;
    const fractionalFactor = ETA * DT ** (2 - ALPHA);

    // Initialise step 0 and 1 with a Gaussian in the centre (matches Python).
    const cx = Math.floor(N_COLS / 2);
    const sigCols = Math.round(1.5 / DY);
    for (let x = 0; x < N_COLS; x++) {
      const v = Math.exp(-((x - cx) ** 2) / (2 * sigCols * sigCols));
      uHist[0][x] = v;
      uHist[1][x] = v;
    }

    let stepN = 1; // current step index

    // nearBuf / farBuf fed to GPU — normalised to [0,1] screen Y (rest = 0.5)
    const nearBuf = new Float32Array(N_COLS);
    const farBuf  = new Float32Array(N_COLS);

    function toScreenY(disp: number): number {
      return Math.max(0.05, Math.min(0.95, 0.5 + disp * 0.06));
    }

    function applyForce(xFrac: number) {
      const hi = stepN % HIST_DEPTH;
      const xCenter = xFrac * N_COLS;
      const sigPx = FORCE_SIGMA * N_COLS;
      for (let x = 0; x < N_COLS; x++) {
        const dx = (x - xCenter) / sigPx;
        const impulse = FORCE_AMP * Math.exp(-(dx * dx) * 0.5);
        uHist[hi][x] += impulse;
        const hiPrev = ((stepN - 1) % HIST_DEPTH + HIST_DEPTH) % HIST_DEPTH;
        uHist[hiPrev][x] += impulse * 0.5;
      }
    }

    function stepWave() {
      if (stepN >= NT_MAX) return;
      const n    = stepN;
      const hi   = n         % HIST_DEPTH;           // current
      const hiP  = (n - 1 + HIST_DEPTH) % HIST_DEPTH; // prev
      const hiN  = (n + 1)   % HIST_DEPTH;           // next (to write)

      const memLen = Math.min(n, MEM);
      const cur    = uHist[hi];
      const prev   = uHist[hiP];
      const next   = uHist[hiN];

      for (let x = 1; x < N_COLS - 1; x++) {
        const lap = cur[x + 1] - 2 * cur[x] + cur[x - 1];

        // GL memory sum: Σ_{k=2}^{memLen+1} w[k] * u[n+1-k, x]
        let memSum = 0;
        for (let k = 2; k <= memLen + 1; k++) {
          const hi_k = ((n + 1 - k) % HIST_DEPTH + HIST_DEPTH) % HIST_DEPTH;
          memSum += glWeights[k] * uHist[hi_k][x];
        }

        next[x] = (
          (2 - glWeights[1] * fractionalFactor) * cur[x]
          - prev[x]
          - fractionalFactor * memSum
          + rSq * lap
        ) / (1.0 + glWeights[0] * fractionalFactor);
      }
      // Boundary: fixed ends
      next[0] = 0;
      next[N_COLS - 1] = 0;

      stepN++;
    }

    function fillDisplayBuffers() {
      const n   = stepN;
      const hi  = n % HIST_DEPTH;
      // Far wave: LAG frames behind
      const nFar   = Math.max(1, n - LAG);
      const hiFar  = nFar % HIST_DEPTH;
      const curArr = uHist[hi];
      const farArr = uHist[hiFar];

      for (let x = 0; x < N_COLS; x++) {
        nearBuf[x] = toScreenY(curArr[x]);
        farBuf[x]  = toScreenY(farArr[x] * DEPTH_SCALE + DEPTH_Y);
      }
    }

    // ── WebGPU setup ──────────────────────────────────────────────────────────
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

      const nearGPU = dev.createBuffer({ size: N_COLS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const farGPU  = dev.createBuffer({ size: N_COLS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const dimBuf  = dev.createBuffer({ size: 16,         usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
      q.writeBuffer(dimBuf, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));

      const renPL = dev.createRenderPipeline({
        layout:    "auto",
        vertex:    { module: dev.createShaderModule({ code: VERT }), entryPoint: "main" },
        fragment:  { module: dev.createShaderModule({ code: FRAG }), entryPoint: "main", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-strip" },
      });

      function frame() {
        if (stopped) return;

        // Run multiple physics steps per frame for faster wave propagation
        const ptr = ptrRef.current;
        const gyr = gyrRef.current;
        for (let s = 0; s < STEPS_PER_FRAME; s++) {
          if (ptr.down) applyForce(ptr.x);
          if (Math.abs(gyr.ax) > 0.8) applyForce(0.5 + Math.sign(gyr.ax) * 0.35);
          stepWave();
        }
        fillDisplayBuffers();

        // Report centre waterline to homepage for text colour adaptation
        onWlRef.current?.(nearBuf[Math.floor(N_COLS / 2)]);

        q.writeBuffer(nearGPU, 0, nearBuf);
        q.writeBuffer(farGPU,  0, farBuf);

        const enc = dev.createCommandEncoder();
        const bg  = dev.createBindGroup({
          layout: renPL.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: nearGPU } },
            { binding: 1, resource: { buffer: farGPU  } },
            { binding: 2, resource: { buffer: dimBuf  } },
          ],
        });
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:1,g:1,b:1,a:1 }, loadOp:"clear", storeOp:"store" }] });
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

    const onDown  = (e: MouseEvent)  => { ptrRef.current = { x: nx(e), down: true }; };
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
