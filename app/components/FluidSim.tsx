"use client";

import { useEffect, useRef } from "react";

// Simulation grid resolution — compute runs at this size, render upscales to viewport.
const SW = 256, SH = 256;

// ---------------------------------------------------------------------------
// Wave equation compute shader.
// Uses three ping-pong storage buffers (prev, cur, next).
// h_next = D * (2*h_cur - h_prev + C² * ∇²h_cur)
// Stability: C² ≤ 0.5 in 2D. We use 0.40 for faster-looking ripples.
// ---------------------------------------------------------------------------
const WAVE = `
@group(0) @binding(0) var<storage,read>       prev: array<f32>;
@group(0) @binding(1) var<storage,read>       cur:  array<f32>;
@group(0) @binding(2) var<storage,read_write> nxt:  array<f32>;

fn s(x: i32, y: i32) -> f32 {
  return cur[u32(clamp(y,0,${SH}-1)) * ${SW}u + u32(clamp(x,0,${SW}-1))];
}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${SW}u || id.y >= ${SH}u) { return; }
  let x = i32(id.x); let y = i32(id.y);
  let i = id.y * ${SW}u + id.x;
  let h   = s(x, y);
  let lap = s(x-1,y) + s(x+1,y) + s(x,y-1) + s(x,y+1) - 4.0*h;
  nxt[i] = 0.994 * (2.0*h - prev[i] + 0.40*lap);
}`;

// ---------------------------------------------------------------------------
// Splat compute shader — Gaussian disturbance injected into a height buffer.
// ---------------------------------------------------------------------------
const SPLAT = `
struct S { x: f32, y: f32, r: f32, str: f32 };
@group(0) @binding(0) var<uniform>            u: S;
@group(0) @binding(1) var<storage,read_write> h: array<f32>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${SW}u || id.y >= ${SH}u) { return; }
  let dx = f32(id.x)/f32(${SW}) - u.x;
  let dy = f32(id.y)/f32(${SH}) - u.y;
  h[id.y*${SW}u+id.x] += u.str * exp(-(dx*dx+dy*dy)/(u.r*u.r));
}`;

// ---------------------------------------------------------------------------
// Fullscreen quad — just covers clip space.
// ---------------------------------------------------------------------------
const VERT = `
@vertex
fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[i], 0., 1.);
}`;

// ---------------------------------------------------------------------------
// Water surface fragment shader.
// Reads height field directly from storage buffer (no texture needed).
// Computes surface normal from height gradient → Phong shading.
// Final color is blended over white page background.
// ---------------------------------------------------------------------------
const FRAG = `
@group(0) @binding(0) var<storage,read> h: array<f32>;
@group(0) @binding(1) var<uniform>      dim: vec4<f32>; // canvas_w, canvas_h, _, _

fn at(x: i32, y: i32) -> f32 {
  return h[u32(clamp(y,0,${SH}-1)) * ${SW}u + u32(clamp(x,0,${SW}-1))];
}

// Bilinear sample of height field at a UV coordinate.
fn sh(uv: vec2<f32>) -> f32 {
  let p = uv * vec2(f32(${SW}), f32(${SH})) - 0.5;
  let i = vec2<i32>(floor(p));
  let f = fract(p);
  return mix(
    mix(at(i.x,   i.y),   at(i.x+1, i.y),   f.x),
    mix(at(i.x,   i.y+1), at(i.x+1, i.y+1), f.x),
    f.y
  );
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / dim.xy;

  // Sample height and four neighbors to compute gradient.
  let e  = 1.5 / f32(${SW});
  let hC = sh(uv);
  let nx = (sh(uv + vec2(-e, 0.)) - sh(uv + vec2(e, 0.))) * 6.0;
  let ny = (sh(uv + vec2(0.,-e)) - sh(uv + vec2(0., e))) * 6.0;
  let nrm = normalize(vec3(nx, ny, 1.0));

  // Light from slightly off-center overhead.
  let lit  = normalize(vec3(-0.15, -0.25, 1.0));
  let diff = clamp(dot(nrm, lit), 0.0, 1.0);
  let spec = pow(clamp(dot(reflect(-lit, nrm), vec3(0.,0.,1.)), 0., 1.), 120.0);

  // Deep vs surface water color keyed by height.
  let deep = vec3(0.02, 0.15, 0.48);
  let surf = vec3(0.14, 0.48, 0.88);
  var wc   = mix(deep, surf, clamp(hC * 4.0 + 0.5, 0., 1.));

  // Diffuse shading + specular highlight + Fresnel rim.
  wc = wc * (0.30 + 0.70 * diff)
     + vec3(spec * 0.95)
     + vec3(0.65, 0.82, 1.0) * pow(1.0 - nrm.z, 5.0) * 0.20;

  // White foam on very high crests.
  wc = mix(wc, vec3(0.95, 0.98, 1.0), smoothstep(0.38, 0.58, hC));

  // Blend 88% water over white page background.
  return vec4(mix(vec3(1.0), wc, 0.88), 1.0);
}`;

type D = ReturnType<typeof Object.create>;

export default function FluidSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ptrRef    = useRef({ x: .5, y: .5, dx: 0, dy: 0, down: false });
  const gyrRef    = useRef({ ax: 0, ay: 0 });

  useEffect(() => {
    const canvas = canvasRef.current!;
    let raf: number, stopped = false;

    // Match WebGPU render target to actual display pixels.
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;

    async function init() {
      const nav = navigator as D;
      if (!nav.gpu) return;
      const adapter: D = await nav.gpu.requestAdapter();
      if (!adapter) return;
      const dev: D = await adapter.requestDevice();
      const q: D = dev.queue;

      const ctx: D = canvas.getContext("webgpu");
      const fmt: string = nav.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: dev, format: fmt, alphaMode: "opaque" });

      const N = SW * SH;
      const sb = (n: number) => dev.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const ub = (n: number) => dev.createBuffer({ size: n, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const md = (code: string) => dev.createShaderModule({ code });
      const cp = (code: string) => dev.createComputePipeline({ layout:"auto", compute:{ module:md(code), entryPoint:"main" }});

      // Three height buffers for ping-pong wave equation (needs prev + cur → next).
      const hBuf   = [sb(N*4), sb(N*4), sb(N*4)];
      const splatU = ub(16);
      const dimU   = ub(16);

      q.writeBuffer(dimU, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));

      const wavePL  = cp(WAVE);
      const splatPL = cp(SPLAT);
      const renPL   = dev.createRenderPipeline({
        layout:   "auto",
        vertex:   { module: md(VERT), entryPoint: "main" },
        fragment: { module: md(FRAG), entryPoint: "main", targets: [{ format: fmt }] },
        primitive:{ topology: "triangle-strip" },
      });

      const bg = (pl: D, entries: D[]) =>
        dev.createBindGroup({ layout: pl.getBindGroupLayout(0), entries });

      function splat(enc: D, bi: number, x: number, y: number, r: number, str: number) {
        q.writeBuffer(splatU, 0, new Float32Array([x, y, r, str]));
        const p = enc.beginComputePass();
        p.setPipeline(splatPL);
        p.setBindGroup(0, bg(splatPL, [
          { binding:0, resource:{ buffer: splatU } },
          { binding:1, resource:{ buffer: hBuf[bi] } },
        ]));
        p.dispatchWorkgroups(Math.ceil(SW/16), Math.ceil(SH/16));
        p.end();
      }

      // Seed a few initial ripples so the water isn't completely flat on load.
      const seed = dev.createCommandEncoder();
      splat(seed, 1, 0.28, 0.38, 0.04, -2.0);
      splat(seed, 1, 0.72, 0.62, 0.03, -1.8);
      splat(seed, 1, 0.50, 0.28, 0.05, -1.5);
      splat(seed, 1, 0.35, 0.72, 0.03, -1.6);
      q.submit([seed.finish()]);

      let frame = 0;

      function step() {
        if (stopped) return;
        const enc = dev.createCommandEncoder();
        const pi  = frame % 3;
        const ci  = (frame + 1) % 3;
        const ni  = (frame + 2) % 3;

        // Inject pointer disturbance into current buffer before wave step.
        const ptr = ptrRef.current;
        const gyr = gyrRef.current;
        if (ptr.down && Math.abs(ptr.dx) + Math.abs(ptr.dy) > 0.0002) {
          splat(enc, ci, ptr.x, ptr.y, 0.022, -2.0);
          ptr.dx = 0; ptr.dy = 0;
        }
        // Gyro: tilt shifts the splat point, simulating gravity pulling the water.
        if (Math.abs(gyr.ax) + Math.abs(gyr.ay) > 0.6) {
          splat(enc, ci, 0.5 + gyr.ay * 0.06, 0.5 - gyr.ax * 0.06, 0.035, -0.9);
        }

        // Wave propagation step: prev, cur → next.
        {
          const p = enc.beginComputePass();
          p.setPipeline(wavePL);
          p.setBindGroup(0, bg(wavePL, [
            { binding:0, resource:{ buffer: hBuf[pi] } },
            { binding:1, resource:{ buffer: hBuf[ci] } },
            { binding:2, resource:{ buffer: hBuf[ni] } },
          ]));
          p.dispatchWorkgroups(Math.ceil(SW/16), Math.ceil(SH/16));
          p.end();
        }

        // Render the newly computed frame (ni) to the canvas.
        const rp = enc.beginRenderPass({ colorAttachments: [{
          view:       ctx.getCurrentTexture().createView(),
          clearValue: { r:1, g:1, b:1, a:1 },
          loadOp:     "clear",
          storeOp:    "store",
        }]});
        rp.setPipeline(renPL);
        rp.setBindGroup(0, bg(renPL, [
          { binding:0, resource:{ buffer: hBuf[ni] } },
          { binding:1, resource:{ buffer: dimU } },
        ]));
        rp.draw(4);
        rp.end();

        q.submit([enc.finish()]);
        frame++;
        raf = requestAnimationFrame(step);
      }

      step();
    }

    init().catch(console.error);

    // ---- Input ----
    const c = canvas;
    let last = { x: .5, y: .5 };
    const rc  = () => c.getBoundingClientRect();
    const nv  = (e: MouseEvent | Touch, r: DOMRect) => ({
      x: (e.clientX - r.left) / r.width,
      y: (e.clientY - r.top)  / r.height,
    });

    const onDown  = (e: MouseEvent) => { last = nv(e, rc()); ptrRef.current.down = true; };
    const onMove  = (e: MouseEvent) => {
      if (!ptrRef.current.down) return;
      const pos = nv(e, rc());
      ptrRef.current = { ...ptrRef.current, x:pos.x, y:pos.y, dx:pos.x-last.x, dy:pos.y-last.y };
      last = pos;
    };
    const onUp    = () => { ptrRef.current.down = false; };
    const onTDown = (e: TouchEvent) => { e.preventDefault(); last = nv(e.touches[0], rc()); ptrRef.current.down = true; };
    const onTMove = (e: TouchEvent) => {
      e.preventDefault();
      const pos = nv(e.touches[0], rc());
      ptrRef.current = { ...ptrRef.current, x:pos.x, y:pos.y, dx:pos.x-last.x, dy:pos.y-last.y };
      last = pos;
    };
    const onTUp   = () => { ptrRef.current.down = false; };
    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      if (a) gyrRef.current = { ax: a.x ?? 0, ay: a.y ?? 0 };
    };

    c.addEventListener("mousedown",  onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    c.addEventListener("touchstart", onTDown, { passive: false });
    c.addEventListener("touchmove",  onTMove, { passive: false });
    c.addEventListener("touchend",   onTUp);
    window.addEventListener("devicemotion", onMotion);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      c.removeEventListener("mousedown",  onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
      c.removeEventListener("touchstart", onTDown);
      c.removeEventListener("touchmove",  onTMove);
      c.removeEventListener("touchend",   onTUp);
      window.removeEventListener("devicemotion", onMotion);
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ cursor: "crosshair" }}
      />
      <p className="absolute bottom-2 right-3 text-white/40 text-xs select-none pointer-events-none">
        click to ripple · tilt on mobile
      </p>
    </div>
  );
}
