"use client";


import { useEffect, useRef } from "react";

const SIM_W = 256;
const SIM_H = 256;

const ADVECT_WGSL = `
struct Uniforms { dt: f32, rdx: f32, dissipation: f32, _pad: f32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var velocity: texture_2d<f32>;
@group(0) @binding(3) var source: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(velocity).x;
  let H = textureDimensions(velocity).y;
  if (id.x >= W || id.y >= H) { return; }
  let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(W), f32(H));
  let vel = textureSampleLevel(velocity, samp, uv, 0.0).xy;
  let prevUV = uv - vel * u.dt * u.rdx / vec2<f32>(f32(W), f32(H));
  let val = textureSampleLevel(source, samp, prevUV, 0.0) * u.dissipation;
  out[id.y * W + id.x] = val;
}`;

const DIVERGENCE_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> div: array<f32>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(velocity).x;
  let H = textureDimensions(velocity).y;
  if (id.x >= W || id.y >= H) { return; }
  let texel = vec2<f32>(1.0/f32(W), 1.0/f32(H));
  let uv = (vec2<f32>(id.xy) + 0.5) * texel;
  let L = textureSampleLevel(velocity, samp, uv - vec2(texel.x,0.0), 0.0).x;
  let R = textureSampleLevel(velocity, samp, uv + vec2(texel.x,0.0), 0.0).x;
  let B = textureSampleLevel(velocity, samp, uv - vec2(0.0,texel.y), 0.0).y;
  let T = textureSampleLevel(velocity, samp, uv + vec2(0.0,texel.y), 0.0).y;
  div[id.y * W + id.x] = 0.5 * (R - L + T - B);
}`;

const PRESSURE_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var pressure: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> pOut: array<f32>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(pressure).x;
  let H = textureDimensions(pressure).y;
  if (id.x >= W || id.y >= H) { return; }
  let texel = vec2<f32>(1.0/f32(W), 1.0/f32(H));
  let uv = (vec2<f32>(id.xy) + 0.5) * texel;
  let L = textureSampleLevel(pressure, samp, uv - vec2(texel.x,0.0), 0.0).x;
  let R = textureSampleLevel(pressure, samp, uv + vec2(texel.x,0.0), 0.0).x;
  let B = textureSampleLevel(pressure, samp, uv - vec2(0.0,texel.y), 0.0).x;
  let T = textureSampleLevel(pressure, samp, uv + vec2(0.0,texel.y), 0.0).x;
  let d = div[id.y * W + id.x];
  pOut[id.y * W + id.x] = (L + R + B + T - d) * 0.25;
}`;

const GRADIENT_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var pressure: texture_2d<f32>;
@group(0) @binding(2) var velocity: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> vOut: array<vec4<f32>>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = textureDimensions(pressure).x;
  let H = textureDimensions(pressure).y;
  if (id.x >= W || id.y >= H) { return; }
  let texel = vec2<f32>(1.0/f32(W), 1.0/f32(H));
  let uv = (vec2<f32>(id.xy) + 0.5) * texel;
  let L = textureSampleLevel(pressure, samp, uv - vec2(texel.x,0.0), 0.0).x;
  let R = textureSampleLevel(pressure, samp, uv + vec2(texel.x,0.0), 0.0).x;
  let B = textureSampleLevel(pressure, samp, uv - vec2(0.0,texel.y), 0.0).x;
  let T = textureSampleLevel(pressure, samp, uv + vec2(0.0,texel.y), 0.0).x;
  let vel = textureSampleLevel(velocity, samp, uv, 0.0).xy;
  vOut[id.y * W + id.x] = vec4(vel - 0.5 * vec2(R-L, T-B), 0.0, 1.0);
}`;

const SPLAT_WGSL = `
struct SplatUniforms { x: f32, y: f32, vx: f32, vy: f32, radius: f32, r: f32, g: f32, b: f32 };
@group(0) @binding(0) var<uniform> s: SplatUniforms;
@group(0) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dye: array<vec4<f32>>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = u32(${SIM_W});
  let H = u32(${SIM_H});
  if (id.x >= W || id.y >= H) { return; }
  let px = f32(id.x) / f32(W);
  let py = f32(id.y) / f32(H);
  let dx = px - s.x;
  let dy = py - s.y;
  let dist2 = dx*dx + dy*dy;
  let falloff = exp(-dist2 / (s.radius * s.radius));
  let idx = id.y * W + id.x;
  velocity[idx] += vec4(s.vx, s.vy, 0.0, 0.0) * falloff;
  dye[idx] += vec4(s.r, s.g, s.b, 1.0) * falloff;
}`;

const RENDER_VERT = `
@vertex
fn main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>,4>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(-1.0,1.0),vec2(1.0,1.0));
  return vec4(pos[vi], 0.0, 1.0);
}`;

const RENDER_FRAG = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var dye: texture_2d<f32>;

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(dye));
  let uv = pos.xy / dims;
  let c = textureSampleLevel(dye, samp, uv, 0.0);
  return vec4(c.rgb, 1.0);
}`;

export default function FluidSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointerRef = useRef({ x: 0.5, y: 0.5, dx: 0, dy: 0, down: false });
  const gyroRef = useRef({ ax: 0, ay: 0 });

  useEffect(() => {
    const canvas = canvasRef.current!;
    let raf: number;
    let stopped = false;

    async function init() {
      const nav = navigator as any;
      if (!nav.gpu) return;
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) return;
      const device: any = await adapter.requestDevice();

      const ctx: any = canvas.getContext("webgpu");
      const fmt: string = nav.gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format: fmt, alphaMode: "opaque" });

      const W = SIM_W, H = SIM_H;
      const N = W * H;
      const floatBytes = N * 4 * 4;

      function makeStorageBuf(size: number) {
        return device.createBuffer({
          size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
      }

      const velBufs = [makeStorageBuf(floatBytes), makeStorageBuf(floatBytes)];
      const dyeBufs = [makeStorageBuf(floatBytes), makeStorageBuf(floatBytes)];
      const divBuf = makeStorageBuf(N * 4);
      const presBufs = [makeStorageBuf(N * 4), makeStorageBuf(N * 4)];

      function makeTex(w: number, h: number, format: string, extraUsage = 0) {
        return device.createTexture({
          size: [w, h],
          format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | extraUsage,
        });
      }

      const velTexs = [makeTex(W, H, "rgba32float"), makeTex(W, H, "rgba32float")];
      const dyeTexs = [makeTex(W, H, "rgba32float"), makeTex(W, H, "rgba32float")];
      const presTexs = [makeTex(W, H, "r32float"), makeTex(W, H, "r32float")];

      const sampler = device.createSampler({
        addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
        magFilter: "linear", minFilter: "linear",
      });

      function copyBufToTex(enc: any, buf: any, tex: any, w: number, h: number, ch = 4) {
        enc.copyBufferToTexture({ buffer: buf, bytesPerRow: w * ch * 4 }, { texture: tex }, [w, h]);
      }

      const advectMod = device.createShaderModule({ code: ADVECT_WGSL });
      const divMod = device.createShaderModule({ code: DIVERGENCE_WGSL });
      const presMod = device.createShaderModule({ code: PRESSURE_WGSL });
      const gradMod = device.createShaderModule({ code: GRADIENT_WGSL });
      const splatMod = device.createShaderModule({ code: SPLAT_WGSL });
      const vertMod = device.createShaderModule({ code: RENDER_VERT });
      const fragMod = device.createShaderModule({ code: RENDER_FRAG });

      const advectUni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const splatUni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

      const renderPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: vertMod, entryPoint: "main" },
        fragment: { module: fragMod, entryPoint: "main", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-strip" },
      });

      const splatPipeline = device.createComputePipeline({ layout: "auto", compute: { module: splatMod, entryPoint: "main" } });
      const advectPipeline = device.createComputePipeline({ layout: "auto", compute: { module: advectMod, entryPoint: "main" } });
      const divPipeline = device.createComputePipeline({ layout: "auto", compute: { module: divMod, entryPoint: "main" } });
      const presPipeline = device.createComputePipeline({ layout: "auto", compute: { module: presMod, entryPoint: "main" } });
      const gradPipeline = device.createComputePipeline({ layout: "auto", compute: { module: gradMod, entryPoint: "main" } });

      function bg(pipeline: any, entries: any[]) {
        return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      }

      let hue = 0;

      function splat(enc: any, x: number, y: number, vx: number, vy: number, vi: number) {
        hue = (hue + 0.04) % 1;
        const h6 = hue * 6, i = Math.floor(h6), f = h6 - i, q = 1 - f;
        let r = 1, g = 0, b = 0;
        if (i === 0) { g = f; } else if (i === 1) { r = q; g = 1; } else if (i === 2) { g = 1; b = f; }
        else if (i === 3) { g = q; b = 1; } else if (i === 4) { r = f; b = 1; } else { r = 1; b = q; }
        device.queue.writeBuffer(splatUni, 0, new Float32Array([x, y, vx * 3, vy * 3, 0.012, r * 0.9, g * 0.9, b * 0.9]));
        const pass = enc.beginComputePass();
        pass.setPipeline(splatPipeline);
        pass.setBindGroup(0, bg(splatPipeline, [
          { binding: 0, resource: { buffer: splatUni } },
          { binding: 1, resource: { buffer: velBufs[vi] } },
          { binding: 2, resource: { buffer: dyeBufs[vi] } },
        ]));
        pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16));
        pass.end();
      }

      function computePass(enc: any, pipeline: any, entries: any[]) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg(pipeline, entries));
        pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16));
        pass.end();
      }

      let frame = 0;

      function step() {
        if (stopped) return;
        const enc = device.createCommandEncoder();
        const vi = frame % 2, vo = 1 - vi;

        const p = pointerRef.current;
        const gy = gyroRef.current;
        if (p.down && (Math.abs(p.dx) + Math.abs(p.dy) > 0.0001)) {
          splat(enc, p.x, p.y, p.dx * 80, p.dy * 80, vi);
          p.dx = 0; p.dy = 0;
        }
        if (Math.abs(gy.ax) + Math.abs(gy.ay) > 0.8) {
          splat(enc, 0.5 + gy.ay * 0.008, 0.5 - gy.ax * 0.008, gy.ay * 0.5, -gy.ax * 0.5, vi);
        }

        // Copy bufs → textures
        copyBufToTex(enc, velBufs[vi], velTexs[vi], W, H);
        copyBufToTex(enc, dyeBufs[vi], dyeTexs[vi], W, H);

        // Advect velocity
        device.queue.writeBuffer(advectUni, 0, new Float32Array([0.016, 1.0, 0.999, 0]));
        computePass(enc, advectPipeline, [
          { binding: 0, resource: { buffer: advectUni } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: velTexs[vi].createView() },
          { binding: 3, resource: velTexs[vi].createView() },
          { binding: 4, resource: { buffer: velBufs[vo] } },
        ]);

        // Advect dye
        device.queue.writeBuffer(advectUni, 0, new Float32Array([0.016, 1.0, 0.995, 0]));
        copyBufToTex(enc, velBufs[vo], velTexs[vo], W, H);
        computePass(enc, advectPipeline, [
          { binding: 0, resource: { buffer: advectUni } },
          { binding: 1, resource: sampler },
          { binding: 2, resource: velTexs[vo].createView() },
          { binding: 3, resource: dyeTexs[vi].createView() },
          { binding: 4, resource: { buffer: dyeBufs[vo] } },
        ]);

        // Divergence
        copyBufToTex(enc, velBufs[vo], velTexs[vo], W, H);
        computePass(enc, divPipeline, [
          { binding: 0, resource: sampler },
          { binding: 1, resource: velTexs[vo].createView() },
          { binding: 2, resource: { buffer: divBuf } },
        ]);

        // Pressure iterations
        enc.copyBufferToTexture({ buffer: presBufs[0], bytesPerRow: W * 4 }, { texture: presTexs[0] }, [W, H]);
        for (let it = 0; it < 20; it++) {
          const pi = it % 2, po = 1 - pi;
          enc.copyBufferToTexture({ buffer: presBufs[pi], bytesPerRow: W * 4 }, { texture: presTexs[pi] }, [W, H]);
          computePass(enc, presPipeline, [
            { binding: 0, resource: sampler },
            { binding: 1, resource: presTexs[pi].createView() },
            { binding: 2, resource: { buffer: divBuf } },
            { binding: 3, resource: { buffer: presBufs[po] } },
          ]);
        }

        // Gradient subtract
        enc.copyBufferToTexture({ buffer: presBufs[0], bytesPerRow: W * 4 }, { texture: presTexs[0] }, [W, H]);
        copyBufToTex(enc, velBufs[vo], velTexs[vo], W, H);
        computePass(enc, gradPipeline, [
          { binding: 0, resource: sampler },
          { binding: 1, resource: presTexs[0].createView() },
          { binding: 2, resource: velTexs[vo].createView() },
          { binding: 3, resource: { buffer: velBufs[vi] } },
        ]);

        // Render dye → canvas
        copyBufToTex(enc, dyeBufs[vo], dyeTexs[vo], W, H);
        const renderPass = enc.beginRenderPass({
          colorAttachments: [{
            view: ctx.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear", storeOp: "store",
          }],
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, device.createBindGroup({
          layout: renderPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: dyeTexs[vo].createView() },
          ],
        }));
        renderPass.draw(4);
        renderPass.end();

        device.queue.submit([enc.finish()]);
        frame++;
        raf = requestAnimationFrame(step);
      }

      // Seed initial splats
      const initEnc = device.createCommandEncoder();
      [[0.3, 0.4, 0.5, -0.3], [0.7, 0.6, -0.4, 0.2], [0.5, 0.3, 0.2, 0.5]].forEach(([x, y, vx, vy]) => {
        splat(initEnc, x, y, vx, vy, 0);
      });
      device.queue.submit([initEnc.finish()]);

      raf = requestAnimationFrame(step);
    }

    init().catch(console.error);

    // --- Input ---
    const c = canvas;
    let lastPos = { x: 0.5, y: 0.5 };

    function xy(e: MouseEvent | Touch, rect: DOMRect) {
      return { x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height };
    }

    function onMouseDown(e: MouseEvent) {
      lastPos = xy(e, c.getBoundingClientRect());
      pointerRef.current.down = true;
    }
    function onMouseMove(e: MouseEvent) {
      if (!pointerRef.current.down) return;
      const pos = xy(e, c.getBoundingClientRect());
      pointerRef.current = { ...pointerRef.current, x: pos.x, y: pos.y, dx: pos.x - lastPos.x, dy: pos.y - lastPos.y };
      lastPos = pos;
    }
    function onMouseUp() { pointerRef.current.down = false; }

    function onTouchStart(e: TouchEvent) {
      e.preventDefault();
      lastPos = xy(e.touches[0], c.getBoundingClientRect());
      pointerRef.current.down = true;
    }
    function onTouchMove(e: TouchEvent) {
      e.preventDefault();
      const pos = xy(e.touches[0], c.getBoundingClientRect());
      pointerRef.current = { ...pointerRef.current, x: pos.x, y: pos.y, dx: pos.x - lastPos.x, dy: pos.y - lastPos.y };
      lastPos = pos;
    }
    function onTouchEnd() { pointerRef.current.down = false; }

    function onDeviceMotion(e: DeviceMotionEvent) {
      const a = e.accelerationIncludingGravity;
      if (!a) return;
      gyroRef.current = { ax: a.x ?? 0, ay: a.y ?? 0 };
    }

    c.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    c.addEventListener("touchstart", onTouchStart, { passive: false });
    c.addEventListener("touchmove", onTouchMove, { passive: false });
    c.addEventListener("touchend", onTouchEnd);
    window.addEventListener("devicemotion", onDeviceMotion);

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      c.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      c.removeEventListener("touchstart", onTouchStart);
      c.removeEventListener("touchmove", onTouchMove);
      c.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("devicemotion", onDeviceMotion);
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
