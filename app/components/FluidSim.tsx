"use client";

import { useEffect, useRef } from "react";

const W = 256, H = 256;

// Manual bilinear sample using textureLoad (rgba32float / r32float are
// "unfilterable-float" in WebGPU — they cannot go through a sampler).
const BILERP = `
fn bilerp4(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
  let d = vec2<f32>(textureDimensions(tex));
  let p = uv * d - 0.5;
  let i = vec2<i32>(floor(p));
  let f = fract(p);
  let mx = vec2<i32>(i32(d.x)-1, i32(d.y)-1);
  let c00 = textureLoad(tex, clamp(i,           vec2(0),mx), 0);
  let c10 = textureLoad(tex, clamp(i+vec2(1,0), vec2(0),mx), 0);
  let c01 = textureLoad(tex, clamp(i+vec2(0,1), vec2(0),mx), 0);
  let c11 = textureLoad(tex, clamp(i+vec2(1,1), vec2(0),mx), 0);
  return mix(mix(c00,c10,f.x), mix(c01,c11,f.x), f.y);
}
fn bilerp1(tex: texture_2d<f32>, uv: vec2<f32>) -> f32 {
  return bilerp4(tex, uv).x;
}`;

const ADVECT = `
struct U { dt:f32, rdx:f32, diss:f32, _p:f32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var<storage,read_write> out: array<vec4<f32>>;
${BILERP}
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${W}u || id.y >= ${H}u) { return; }
  let uv   = (vec2<f32>(id.xy)+0.5) / vec2<f32>(${W}.0,${H}.0);
  let v    = bilerp4(vel, uv).xy;
  let prev = clamp(uv - v*u.dt*u.rdx/vec2<f32>(${W}.0,${H}.0), vec2(0.0), vec2(1.0));
  out[id.y*${W}u+id.x] = bilerp4(src, prev) * u.diss;
}`;

const DIVERGENCE = `
@group(0) @binding(0) var vel: texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> div: array<f32>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${W}u || id.y >= ${H}u) { return; }
  let x=i32(id.x); let y=i32(id.y);
  let L=textureLoad(vel,vec2(max(x-1,0),   y),0).x;
  let R=textureLoad(vel,vec2(min(x+1,${W}-1),y),0).x;
  let B=textureLoad(vel,vec2(x,max(y-1,0)),   0).y;
  let T=textureLoad(vel,vec2(x,min(y+1,${H}-1)),0).y;
  div[id.y*${W}u+id.x] = 0.5*(R-L+T-B);
}`;

const PRESSURE = `
@group(0) @binding(0) var pres: texture_2d<f32>;
@group(0) @binding(1) var<storage,read>       div:  array<f32>;
@group(0) @binding(2) var<storage,read_write> pOut: array<f32>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${W}u || id.y >= ${H}u) { return; }
  let x=i32(id.x); let y=i32(id.y);
  let L=textureLoad(pres,vec2(max(x-1,0),    y),0).x;
  let R=textureLoad(pres,vec2(min(x+1,${W}-1),y),0).x;
  let B=textureLoad(pres,vec2(x,max(y-1,0)),    0).x;
  let T=textureLoad(pres,vec2(x,min(y+1,${H}-1)),0).x;
  pOut[id.y*${W}u+id.x] = (L+R+B+T - div[id.y*${W}u+id.x])*0.25;
}`;

const GRADIENT = `
@group(0) @binding(0) var pres: texture_2d<f32>;
@group(0) @binding(1) var vel:  texture_2d<f32>;
@group(0) @binding(2) var<storage,read_write> vOut: array<vec4<f32>>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${W}u || id.y >= ${H}u) { return; }
  let x=i32(id.x); let y=i32(id.y);
  let L=textureLoad(pres,vec2(max(x-1,0),    y),0).x;
  let R=textureLoad(pres,vec2(min(x+1,${W}-1),y),0).x;
  let B=textureLoad(pres,vec2(x,max(y-1,0)),    0).x;
  let T=textureLoad(pres,vec2(x,min(y+1,${H}-1)),0).x;
  let v=textureLoad(vel,vec2(x,y),0).xy;
  vOut[id.y*${W}u+id.x]=vec4(v-0.5*vec2(R-L,T-B),0.0,1.0);
}`;

const SPLAT = `
struct S{x:f32,y:f32,vx:f32,vy:f32,rad:f32,r:f32,g:f32,b:f32};
@group(0) @binding(0) var<uniform> s: S;
@group(0) @binding(1) var<storage,read_write> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> dye: array<vec4<f32>>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${W}u || id.y >= ${H}u) { return; }
  let dx=f32(id.x)/${W}.0-s.x; let dy=f32(id.y)/${H}.0-s.y;
  let w=exp(-(dx*dx+dy*dy)/(s.rad*s.rad));
  let i=id.y*${W}u+id.x;
  vel[i]+=vec4(s.vx,s.vy,0.0,0.0)*w;
  dye[i]+=vec4(s.r,s.g,s.b,1.0)*w;
}`;

const VERT = `
@vertex fn main(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{
  var p=array<vec2<f32>,4>(vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(1.,1.));
  return vec4(p[i],0.,1.);
}`;

const FRAG = `
@group(0) @binding(0) var dye: texture_2d<f32>;
${BILERP}
@fragment fn main(@builtin(position) pos:vec4<f32>)->@location(0) vec4<f32>{
  return vec4(bilerp4(dye, pos.xy/vec2<f32>(textureDimensions(dye))).rgb, 1.0);
}`;

type D = ReturnType<typeof Object.create>;

export default function FluidSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ptr = useRef({ x: .5, y: .5, dx: 0, dy: 0, down: false });
  const gyr = useRef({ ax: 0, ay: 0 });

  useEffect(() => {
    const canvas = canvasRef.current!;
    let raf: number, stopped = false;

    async function init() {
      const nav = navigator as D;
      if (!nav.gpu) return;
      const adapter: D = await nav.gpu.requestAdapter();
      if (!adapter) return;
      const dev: D = await adapter.requestDevice();
      const queue: D = dev.queue;

      const ctx: D = canvas.getContext("webgpu");
      const fmt: string = nav.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: dev, format: fmt, alphaMode: "opaque" });

      const N = W * H;
      const sb  = (n: number) => dev.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });
      const tex = (f: string) => dev.createTexture({ size:[W,H], format:f, usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST });
      const mod = (code: string) => dev.createShaderModule({ code });
      const cpl = (code: string) => dev.createComputePipeline({ layout:"auto", compute:{ module:mod(code), entryPoint:"main" }});

      const velB  = [sb(N*16), sb(N*16)];
      const dyeB  = [sb(N*16), sb(N*16)];
      const divB  = sb(N*4);
      const presB = [sb(N*4),  sb(N*4)];
      const velT  = [tex("rgba32float"), tex("rgba32float")];
      const dyeT  = [tex("rgba32float"), tex("rgba32float")];
      const presT = [tex("r32float"),    tex("r32float")];

      const advPL  = cpl(ADVECT);
      const divPL  = cpl(DIVERGENCE);
      const presPL = cpl(PRESSURE);
      const gradPL = cpl(GRADIENT);
      const splPL  = cpl(SPLAT);
      const renPL  = dev.createRenderPipeline({
        layout:"auto",
        vertex:   { module:mod(VERT), entryPoint:"main" },
        fragment: { module:mod(FRAG), entryPoint:"main", targets:[{format:fmt}] },
        primitive:{ topology:"triangle-strip" },
      });

      const advU = dev.createBuffer({ size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
      const splU = dev.createBuffer({ size:32, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });

      const bg = (pl: D, entries: D[]) => dev.createBindGroup({ layout:pl.getBindGroupLayout(0), entries });

      function b2t(enc: D, buf: D, t: D, ch: number) {
        enc.copyBufferToTexture({ buffer:buf, bytesPerRow:W*ch*4 }, { texture:t }, [W,H]);
      }
      function run(enc: D, pl: D, entries: D[]) {
        const p = enc.beginComputePass();
        p.setPipeline(pl);
        p.setBindGroup(0, bg(pl, entries));
        p.dispatchWorkgroups(Math.ceil(W/16), Math.ceil(H/16));
        p.end();
      }

      let hue = 0;
      function splat(enc: D, x: number, y: number, vx: number, vy: number, vi: number) {
        hue = (hue+0.04)%1;
        const h6=hue*6, i=Math.floor(h6), f=h6-i, q=1-f;
        let r=1,g=0,b=0;
        if(i===0){g=f}else if(i===1){r=q;g=1}else if(i===2){g=1;b=f}
        else if(i===3){g=q;b=1}else if(i===4){r=f;b=1}else{r=1;b=q}
        queue.writeBuffer(splU, 0, new Float32Array([x,y,vx*3,vy*3,0.012,r*.9,g*.9,b*.9]));
        run(enc, splPL, [
          {binding:0,resource:{buffer:splU}},
          {binding:1,resource:{buffer:velB[vi]}},
          {binding:2,resource:{buffer:dyeB[vi]}},
        ]);
      }

      let frame = 0;
      function step() {
        if (stopped) return;
        const enc = dev.createCommandEncoder();
        const vi = frame%2, vo = 1-vi;

        const p = ptr.current, gy = gyr.current;
        if (p.down && Math.abs(p.dx)+Math.abs(p.dy) > 0.0001) {
          splat(enc, p.x, p.y, p.dx*80, p.dy*80, vi);
          p.dx = 0; p.dy = 0;
        }
        if (Math.abs(gy.ax)+Math.abs(gy.ay) > 0.8) {
          splat(enc, .5+gy.ay*.008, .5-gy.ax*.008, gy.ay*.5, -gy.ax*.5, vi);
        }

        b2t(enc, velB[vi], velT[vi], 4);
        b2t(enc, dyeB[vi], dyeT[vi], 4);

        queue.writeBuffer(advU, 0, new Float32Array([0.016,1,0.999,0]));
        run(enc, advPL, [
          {binding:0,resource:{buffer:advU}},
          {binding:1,resource:velT[vi].createView()},
          {binding:2,resource:velT[vi].createView()},
          {binding:3,resource:{buffer:velB[vo]}},
        ]);

        queue.writeBuffer(advU, 0, new Float32Array([0.016,1,0.995,0]));
        b2t(enc, velB[vo], velT[vo], 4);
        run(enc, advPL, [
          {binding:0,resource:{buffer:advU}},
          {binding:1,resource:velT[vo].createView()},
          {binding:2,resource:dyeT[vi].createView()},
          {binding:3,resource:{buffer:dyeB[vo]}},
        ]);

        b2t(enc, velB[vo], velT[vo], 4);
        run(enc, divPL, [
          {binding:0,resource:velT[vo].createView()},
          {binding:1,resource:{buffer:divB}},
        ]);

        enc.copyBufferToTexture({buffer:presB[0],bytesPerRow:W*4},{texture:presT[0]},[W,H]);
        for (let it=0; it<20; it++) {
          const pi=it%2, po=1-pi;
          enc.copyBufferToTexture({buffer:presB[pi],bytesPerRow:W*4},{texture:presT[pi]},[W,H]);
          run(enc, presPL, [
            {binding:0,resource:presT[pi].createView()},
            {binding:1,resource:{buffer:divB}},
            {binding:2,resource:{buffer:presB[po]}},
          ]);
        }

        enc.copyBufferToTexture({buffer:presB[0],bytesPerRow:W*4},{texture:presT[0]},[W,H]);
        b2t(enc, velB[vo], velT[vo], 4);
        run(enc, gradPL, [
          {binding:0,resource:presT[0].createView()},
          {binding:1,resource:velT[vo].createView()},
          {binding:2,resource:{buffer:velB[vi]}},
        ]);

        b2t(enc, dyeB[vo], dyeT[vo], 4);
        const rp = enc.beginRenderPass({
          colorAttachments:[{
            view: ctx.getCurrentTexture().createView(),
            clearValue:{r:0,g:0,b:0,a:1},
            loadOp:"clear", storeOp:"store",
          }],
        });
        rp.setPipeline(renPL);
        rp.setBindGroup(0, bg(renPL, [{binding:0,resource:dyeT[vo].createView()}]));
        rp.draw(4);
        rp.end();

        queue.submit([enc.finish()]);
        frame++;
        raf = requestAnimationFrame(step);
      }

      // Seed a few splats on load
      const seed = dev.createCommandEncoder();
      [[.3,.4,.5,-.3],[.7,.6,-.4,.2],[.5,.3,.2,.5]].forEach(([x,y,vx,vy]) => splat(seed,x,y,vx,vy,0));
      queue.submit([seed.finish()]);

      raf = requestAnimationFrame(step);
    }

    init().catch(console.error);

    // ---- Input ----
    const c = canvas;
    let last = { x:.5, y:.5 };
    const rc  = () => c.getBoundingClientRect();
    const norm = (e: MouseEvent|Touch, r: DOMRect) => ({ x:(e.clientX-r.left)/r.width, y:(e.clientY-r.top)/r.height });

    const onDown  = (e: MouseEvent)  => { last=norm(e,rc()); ptr.current.down=true; };
    const onMove  = (e: MouseEvent)  => {
      if (!ptr.current.down) return;
      const pos=norm(e,rc());
      ptr.current={...ptr.current,x:pos.x,y:pos.y,dx:pos.x-last.x,dy:pos.y-last.y};
      last=pos;
    };
    const onUp    = () => { ptr.current.down=false; };
    const onTDown = (e: TouchEvent) => { e.preventDefault(); last=norm(e.touches[0],rc()); ptr.current.down=true; };
    const onTMove = (e: TouchEvent) => {
      e.preventDefault();
      const pos=norm(e.touches[0],rc());
      ptr.current={...ptr.current,x:pos.x,y:pos.y,dx:pos.x-last.x,dy:pos.y-last.y};
      last=pos;
    };
    const onTUp   = () => { ptr.current.down=false; };
    const onMotion = (e: DeviceMotionEvent) => {
      const a=e.accelerationIncludingGravity;
      if (a) gyr.current={ax:a.x??0,ay:a.y??0};
    };

    c.addEventListener("mousedown",  onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    c.addEventListener("touchstart", onTDown, {passive:false});
    c.addEventListener("touchmove",  onTMove, {passive:false});
    c.addEventListener("touchend",   onTUp);
    window.addEventListener("devicemotion", onMotion);

    return () => {
      stopped=true;
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
        width={W}
        height={H}
        className="w-full h-full"
        style={{ imageRendering:"auto", cursor:"crosshair" }}
      />
      <p className="absolute bottom-2 right-3 text-white/30 text-xs select-none pointer-events-none">
        drag to paint · tilt on mobile
      </p>
    </div>
  );
}
