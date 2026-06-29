"use client";

import { useEffect, useRef } from "react";

const SIM_W = 256;
const SIM_H = 256;

// Bilinear helper — used in shaders that need sub-pixel sampling.
// textureLoad (not textureSampleLevel) avoids the unfilterable-float error on rgba32float.
const BILERP_FN = `
fn bilerp(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
  let d  = vec2<f32>(textureDimensions(tex));
  let p  = uv * d - 0.5;
  let i  = vec2<i32>(floor(p));
  let f  = fract(p);
  let mx = vec2<i32>(i32(d.x)-1, i32(d.y)-1);
  let c00 = textureLoad(tex, clamp(i,             vec2(0), mx), 0);
  let c10 = textureLoad(tex, clamp(i+vec2(1,0),   vec2(0), mx), 0);
  let c01 = textureLoad(tex, clamp(i+vec2(0,1),   vec2(0), mx), 0);
  let c11 = textureLoad(tex, clamp(i+vec2(1,1),   vec2(0), mx), 0);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}`;

// Advect: back-trace along velocity and sample source field.
const ADVECT_WGSL = `
struct Uniforms { dt: f32, rdx: f32, dissipation: f32, _pad: f32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var source:   texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
${BILERP_FN}
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(velocity).x;
  let H = textureDimensions(velocity).y;
  if (id.x >= W || id.y >= H) { return; }
  let uv  = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(W), f32(H));
  let vel = bilerp(velocity, uv).xy;
  let prev = clamp(uv - vel * u.dt * u.rdx / vec2<f32>(f32(W), f32(H)), vec2(0.0), vec2(1.0));
  out[id.y * W + id.x] = bilerp(source, prev) * u.dissipation;
}`;

// Divergence: central differences on velocity neighbors.
const DIVERGENCE_WGSL = `
@group(0) @binding(0) var velocity: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> div: array<f32>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(velocity).x;
  let H = textureDimensions(velocity).y;
  if (id.x >= W || id.y >= H) { return; }
  let xi = i32(id.x); let yi = i32(id.y);
  let W1 = i32(W)-1;  let H1 = i32(H)-1;
  let L = textureLoad(velocity, vec2(max(xi-1,0),  yi),         0).x;
  let R = textureLoad(velocity, vec2(min(xi+1,W1), yi),         0).x;
  let B = textureLoad(velocity, vec2(xi, max(yi-1,0)),          0).y;
  let T = textureLoad(velocity, vec2(xi, min(yi+1,H1)),         0).y;
  div[id.y * W + id.x] = 0.5 * (R - L + T - B);
}`;

// Jacobi pressure solve iteration.
const PRESSURE_WGSL = `
@group(0) @binding(0) var pressure: texture_2d<f32>;
@group(0) @binding(1) var<storage, read>       div:  array<f32>;
@group(0) @binding(2) var<storage, read_write> pOut: array<f32>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(pressure).x;
  let H = textureDimensions(pressure).y;
  if (id.x >= W || id.y >= H) { return; }
  let xi = i32(id.x); let yi = i32(id.y);
  let W1 = i32(W)-1;  let H1 = i32(H)-1;
  let L = textureLoad(pressure, vec2(max(xi-1,0),  yi),         0).x;
  let R = textureLoad(pressure, vec2(min(xi+1,W1), yi),         0).x;
  let B = textureLoad(pressure, vec2(xi, max(yi-1,0)),          0).x;
  let T = textureLoad(pressure, vec2(xi, min(yi+1,H1)),         0).x;
  pOut[id.y * W + id.x] = (L + R + B + T - div[id.y * W + id.x]) * 0.25;
}`;

// Subtract pressure gradient from velocity to enforce incompressibility.
const GRADIENT_WGSL = `
@group(0) @binding(0) var pressure: texture_2d<f32>;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> vOut: array<vec4<f32>>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(pressure).x;
  let H = textureDimensions(pressure).y;
  if (id.x >= W || id.y >= H) { return; }
  let xi = i32(id.x); let yi = i32(id.y);
  let W1 = i32(W)-1;  let H1 = i32(H)-1;
  let L = textureLoad(pressure, vec2(max(xi-1,0),  yi),         0).x;
  let R = textureLoad(pressure, vec2(min(xi+1,W1), yi),         0).x;
  let B = textureLoad(pressure, vec2(xi, max(yi-1,0)),          0).x;
  let T = textureLoad(pressure, vec2(xi, min(yi+1,H1)),         0).x;
  let vel = textureLoad(velocity, vec2(xi, yi), 0).xy;
  vOut[id.y * W + id.x] = vec4(vel - 0.5 * vec2(R-L, T-B), 0.0, 1.0);
}`;

// Gaussian splat: injects velocity and dye color at a point.
const SPLAT_WGSL = `
struct S { x: f32, y: f32, vx: f32, vy: f32, radius: f32, r: f32, g: f32, b: f32 };
@group(0) @binding(0) var<uniform> s: S;
@group(0) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dye:      array<vec4<f32>>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = u32(${SIM_W}); let H = u32(${SIM_H});
  if (id.x >= W || id.y >= H) { return; }
  let dx = f32(id.x)/f32(W) - s.x;
  let dy = f32(id.y)/f32(H) - s.y;
  let w  = exp(-(dx*dx + dy*dy) / (s.radius * s.radius));
  let idx = id.y * W + id.x;
  velocity[idx] += vec4(s.vx, s.vy, 0.0, 0.0) * w;
  dye[idx]      += vec4(s.r,  s.g,  s.b, 1.0) * w;
}`;

// Full-screen quad vertex shader.
const RENDER_VERT = `
@vertex
fn main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[vi], 0., 1.);
}`;

// Fragment shader: bilinear-upscale the 256^2 dye texture to the canvas.
const RENDER_FRAG = `
@group(0) @binding(0) var dye: texture_2d<f32>;
${BILERP_FN}
@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let d  = vec2<f32>(textureDimensions(dye));
  let uv = pos.xy / d;
  return vec4(bilerp(dye, uv).rgb, 1.0);
}`;

export default function FluidSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ptr = useRef({ x: .5, y: .5, dx: 0, dy: 0, down: false });
  const gyr = useRef({ ax: 0, ay: 0 });

  useEffect(() => {
    const canvas = canvasRef.current!;
    let raf: number, stopped = false;

    async function init() {
      const nav = navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown>, getPreferredCanvasFormat(): string } };
      if (!nav.gpu) return;
      const adapter = await nav.gpu.requestAdapter() as { requestDevice(d?: unknown): Promise<unknown>, features: Set<string> } | null;
      if (!adapter) return;
      const device = await adapter.requestDevice() as unknown as Record<string, unknown>;
      const queue = device.queue as { writeBuffer(b: unknown, o: number, d: ArrayBufferView): void, submit(c: unknown[]): void };

      const ctx = canvas.getContext("webgpu") as unknown as { configure(d: unknown): void, getCurrentTexture(): { createView(): unknown } };
      const fmt = nav.gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format: fmt, alphaMode: "opaque" });

      const createBuffer = device.createBuffer as (d: unknown) => unknown;
      const createTexture = device.createTexture as (d: unknown) => { createView(): unknown };
      const createSampler = device.createSampler as (d: unknown) => unknown;
      const createShaderModule = device.createShaderModule as (d: unknown) => unknown;
      const createComputePipeline = device.createComputePipeline as (d: unknown) => { getBindGroupLayout(i: number): unknown };
      const createRenderPipeline = device.createRenderPipeline as (d: unknown) => { getBindGroupLayout(i: number): unknown };
      const createBindGroup = device.createBindGroup as (d: unknown) => unknown;
      const createCommandEncoder = device.createCommandEncoder as () => unknown;

      const W = SIM_W, H = SIM_H, N = W * H;

      function storageBuf(size: number) {
        return createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      }
      function makeTex(fmt: string) {
        return createTexture({ size: [W, H], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      }

      const velBufs  = [storageBuf(N*16), storageBuf(N*16)];
      const dyeBufs  = [storageBuf(N*16), storageBuf(N*16)];
      const divBuf   = storageBuf(N*4);
      const presBufs = [storageBuf(N*4), storageBuf(N*4)];

      const velTexs  = [makeTex("rgba32float"), makeTex("rgba32float")];
      const dyeTexs  = [makeTex("rgba32float"), makeTex("rgba32float")];
      const presTexs = [makeTex("r32float"),    makeTex("r32float")];

      // Pipelines
      function computePL(code: string) {
        return createComputePipeline({ layout: "auto", compute: { module: createShaderModule({ code }), entryPoint: "main" } });
      }
      const advectPL  = computePL(ADVECT_WGSL);
      const divPL     = computePL(DIVERGENCE_WGSL);
      const presPL    = computePL(PRESSURE_WGSL);
      const gradPL    = computePL(GRADIENT_WGSL);
      const splatPL   = computePL(SPLAT_WGSL);
      const renderPL  = createRenderPipeline({
        layout: "auto",
        vertex:   { module: createShaderModule({ code: RENDER_VERT }), entryPoint: "main" },
        fragment: { module: createShaderModule({ code: RENDER_FRAG }), entryPoint: "main", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-strip" },
      });

      const advectUni = createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const splatUni  = createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

      function bg(pl: { getBindGroupLayout(i: number): unknown }, entries: unknown[]) {
        return createBindGroup({ layout: pl.getBindGroupLayout(0), entries });
      }
      function copyBufToTex(enc: unknown, buf: unknown, tex: { createView(): unknown }, ch: number) {
        (enc as { copyBufferToTexture(s: unknown, d: unknown, sz: unknown): void })
          .copyBufferToTexture({ buffer: buf, bytesPerRow: W * ch * 4 }, { texture: tex }, [W, H]);
      }
      function dispatch(enc: unknown, pl: { getBindGroupLayout(i: number): unknown }, entries: unknown[]) {
        const pass = (enc as { beginComputePass(): { setPipeline(p: unknown): void, setBindGroup(i: number, b: unknown): void, dispatchWorkgroups(x: number, y: number): void, end(): void } }).beginComputePass();
        pass.setPipeline(pl);
        pass.setBindGroup(0, bg(pl, entries));
        pass.dispatchWorkgroups(Math.ceil(W/16), Math.ceil(H/16));
        pass.end();
      }

      let hue = 0;

      function splat(enc: unknown, x: number, y: number, vx: number, vy: number, vi: number) {
        hue = (hue + 0.04) % 1;
        const h6 = hue*6, i = Math.floor(h6), f = h6-i, q = 1-f;
        let r=1,g=0,b=0;
        if(i===0){g=f}else if(i===1){r=q;g=1}else if(i===2){g=1;b=f}
        else if(i===3){g=q;b=1}else if(i===4){r=f;b=1}else{r=1;b=q}
        queue.writeBuffer(splatUni, 0, new Float32Array([x, y, vx*3, vy*3, 0.012, r*.9, g*.9, b*.9]));
        dispatch(enc, splatPL, [
          { binding:0, resource:{buffer:splatUni} },
          { binding:1, resource:{buffer:velBufs[vi]} },
          { binding:2, resource:{buffer:dyeBufs[vi]} },
        ]);
      }

      let frame = 0;

      function step() {
        if (stopped) return;
        const enc = createCommandEncoder() as unknown;
        const vi = frame%2, vo = 1-vi;

        const p = ptr.current, g = gyr.current;
        if (p.down && Math.abs(p.dx)+Math.abs(p.dy) > 0.0001) {
          splat(enc, p.x, p.y, p.dx*80, p.dy*80, vi);
          p.dx = 0; p.dy = 0;
        }
        if (Math.abs(g.ax)+Math.abs(g.ay) > 0.8) {
          splat(enc, .5+g.ay*.008, .5-g.ax*.008, g.ay*.5, -g.ax*.5, vi);
        }

        copyBufToTex(enc, velBufs[vi], velTexs[vi], 4);
        copyBufToTex(enc, dyeBufs[vi], dyeTexs[vi], 4);

        // Advect velocity
        queue.writeBuffer(advectUni, 0, new Float32Array([0.016, 1, 0.999, 0]));
        dispatch(enc, advectPL, [
          {binding:0,resource:{buffer:advectUni}},
          {binding:1,resource:velTexs[vi].createView()},
          {binding:2,resource:velTexs[vi].createView()},
          {binding:3,resource:{buffer:velBufs[vo]}},
        ]);

        // Advect dye
        queue.writeBuffer(advectUni, 0, new Float32Array([0.016, 1, 0.995, 0]));
        copyBufToTex(enc, velBufs[vo], velTexs[vo], 4);
        dispatch(enc, advectPL, [
          {binding:0,resource:{buffer:advectUni}},
          {binding:1,resource:velTexs[vo].createView()},
          {binding:2,resource:dyeTexs[vi].createView()},
          {binding:3,resource:{buffer:dyeBufs[vo]}},
        ]);

        // Divergence
        copyBufToTex(enc, velBufs[vo], velTexs[vo], 4);
        dispatch(enc, divPL, [
          {binding:0,resource:velTexs[vo].createView()},
          {binding:1,resource:{buffer:divBuf}},
        ]);

        // Pressure solve (20 Jacobi iterations)
        (enc as { copyBufferToTexture(s: unknown, d: unknown, sz: unknown): void })
          .copyBufferToTexture({buffer:presBufs[0],bytesPerRow:W*4},{texture:presTexs[0]},[W,H]);
        for (let it = 0; it < 20; it++) {
          const pi = it%2, po = 1-pi;
          (enc as { copyBufferToTexture(s: unknown, d: unknown, sz: unknown): void })
            .copyBufferToTexture({buffer:presBufs[pi],bytesPerRow:W*4},{texture:presTexs[pi]},[W,H]);
          dispatch(enc, presPL, [
            {binding:0,resource:presTexs[pi].createView()},
            {binding:1,resource:{buffer:divBuf}},
            {binding:2,resource:{buffer:presBufs[po]}},
          ]);
        }

        // Gradient subtract
        (enc as { copyBufferToTexture(s: unknown, d: unknown, sz: unknown): void })
          .copyBufferToTexture({buffer:presBufs[0],bytesPerRow:W*4},{texture:presTexs[0]},[W,H]);
        copyBufToTex(enc, velBufs[vo], velTexs[vo], 4);
        dispatch(enc, gradPL, [
          {binding:0,resource:presTexs[0].createView()},
          {binding:1,resource:velTexs[vo].createView()},
          {binding:2,resource:{buffer:velBufs[vi]}},
        ]);

        // Render dye to canvas
        copyBufToTex(enc, dyeBufs[vo], dyeTexs[vo], 4);
        const rp = (enc as { beginRenderPass(d: unknown): { setPipeline(p: unknown): void, setBindGroup(i: number, b: unknown): void, draw(n: number): void, end(): void } })
          .beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue:{r:0,g:0,b:0,a:1}, loadOp:"clear", storeOp:"store" }] });
        rp.setPipeline(renderPL);
        rp.setBindGroup(0, bg(renderPL, [{binding:0,resource:dyeTexs[vo].createView()}]));
        rp.draw(4);
        rp.end();

        queue.submit([(enc as { finish(): unknown }).finish()]);
        frame++;
        raf = requestAnimationFrame(step);
      }

      // Seed initial splats so there's something to see immediately
      const seed = createCommandEncoder() as unknown;
      [[.3,.4,.5,-.3],[.7,.6,-.4,.2],[.5,.3,.2,.5]].forEach(([x,y,vx,vy]) => splat(seed, x, y, vx, vy, 0));
      queue.submit([(seed as { finish(): unknown }).finish()]);

      raf = requestAnimationFrame(step);
    }

    init().catch(console.error);

    // ---- Pointer / touch / gyro input ----
    const c = canvas;
    let last = { x: .5, y: .5 };
    const rc = () => c.getBoundingClientRect();
    const norm = (e: MouseEvent | Touch, r: DOMRect) => ({
      x: (e.clientX - r.left) / r.width,
      y: (e.clientY - r.top)  / r.height,
    });

    const onDown  = (e: MouseEvent)  => { last = norm(e, rc()); ptr.current.down = true; };
    const onMove  = (e: MouseEvent)  => {
      if (!ptr.current.down) return;
      const pos = norm(e, rc());
      ptr.current = { ...ptr.current, x:pos.x, y:pos.y, dx:pos.x-last.x, dy:pos.y-last.y };
      last = pos;
    };
    const onUp    = ()               => { ptr.current.down = false; };
    const onTDown = (e: TouchEvent)  => { e.preventDefault(); last = norm(e.touches[0], rc()); ptr.current.down = true; };
    const onTMove = (e: TouchEvent)  => {
      e.preventDefault();
      const pos = norm(e.touches[0], rc());
      ptr.current = { ...ptr.current, x:pos.x, y:pos.y, dx:pos.x-last.x, dy:pos.y-last.y };
      last = pos;
    };
    const onTUp   = ()               => { ptr.current.down = false; };
    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      if (a) gyr.current = { ax: a.x??0, ay: a.y??0 };
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
        width={SIM_W}
        height={SIM_H}
        className="w-full h-full"
        style={{ imageRendering: "auto", cursor: "crosshair" }}
      />
      <p className="absolute bottom-2 right-3 text-white/30 text-xs select-none pointer-events-none">
        drag to paint · tilt on mobile
      </p>
    </div>
  );
}
