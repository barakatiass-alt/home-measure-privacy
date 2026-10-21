// Gaussian-splat renderer and trainer on the GPU (WebGPU / WGSL), stage 3.2 step 4.
// Mirrors src/splat/reference.ts so every kernel can be checked against it; the same
// kernels are then ported to Metal for the iPhone.
//
// One Gaussian in the parameter buffer (16 floats):
//   0-2 mean, 3 pad, 4-6 logScale, 7 opacityLogit, 8-11 rotation wxyz, 12-14 colour, 15 pad
// One projected splat (12 floats):
//   u, v, conicA, conicB, conicC, opacity, radius, depth, r, g, b, valid
// Gradients per Gaussian (16 floats, accumulated with float atomics):
//   0-2 colour, 3 opacityLogit, 4-8 screen (u, v, a, b, c), 9-15 unused here
// Shape gradients per Gaussian (10 floats): mean (3), logScale (3), rotation (4)

const COMMON = /* wgsl */ `
struct Camera { view: array<vec4f, 3>, intr: vec4f, size: vec2f, count: u32, pixels: u32 };

struct Projected { ok: bool, u: f32, v: f32, a: f32, b: f32, c: f32, radius: f32, depth: f32 };

fn quatToMat(q: vec4f) -> mat3x3f {
  let n = q / max(length(q), 1e-12);
  let w = n.x; let x = n.y; let y = n.z; let z = n.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

/** Same steps as project() in reference.ts. q: mean.xyz, logScale.xyz, rotation.wxyz. */
fn projectShape(cam: Camera, q: array<f32, 10>) -> Projected {
  var out: Projected;
  out.ok = false;
  let mean = vec3f(q[0], q[1], q[2]);
  let r0 = cam.view[0]; let r1 = cam.view[1]; let r2 = cam.view[2];
  let p = vec3f(dot(r0.xyz, mean) + r0.w, dot(r1.xyz, mean) + r1.w, dot(r2.xyz, mean) + r2.w);
  if (p.z < 0.2) { return out; }
  let s = exp(vec3f(q[3], q[4], q[5]));
  let R = quatToMat(vec4f(q[6], q[7], q[8], q[9]));
  let M = R * mat3x3f(vec3f(s.x, 0.0, 0.0), vec3f(0.0, s.y, 0.0), vec3f(0.0, 0.0, s.z));
  let W = transpose(mat3x3f(r0.xyz, r1.xyz, r2.xyz));
  let C = W * (M * transpose(M)) * transpose(W);
  let fx = cam.intr.x; let fy = cam.intr.y;
  let j0 = vec3f(fx / p.z, 0.0, -fx * p.x / (p.z * p.z));
  let j1 = vec3f(0.0, fy / p.z, -fy * p.y / (p.z * p.z));
  let a = dot(j0, C * j0) + 0.3;
  let b = dot(j0, C * j1);
  let c = dot(j1, C * j1) + 0.3;
  let det = a * c - b * b;
  if (det <= 0.0) { return out; }
  let mid = (a + c) / 2.0;
  out.ok = true;
  out.u = fx * (p.x / p.z) + cam.intr.z;
  out.v = fy * (p.y / p.z) + cam.intr.w;
  out.a = c / det;
  out.b = -b / det;
  out.c = a / det;
  out.radius = ceil(3.0 * sqrt(mid + sqrt(max(0.1, mid * mid - det))));
  out.depth = p.z;
  return out;
}

`;

const PREPROCESS = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read_write> gs: array<f32>;
@group(0) @binding(2) var<storage, read_write> splats: array<f32>;

fn shapeOf(i: u32) -> array<f32, 10> {
  let b = i * 16u;
  return array<f32, 10>(gs[b], gs[b + 1u], gs[b + 2u], gs[b + 4u], gs[b + 5u], gs[b + 6u], gs[b + 8u], gs[b + 9u], gs[b + 10u], gs[b + 11u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cam.count) { return; }
  let o = i * 12u;
  let p = projectShape(cam, shapeOf(i));
  splats[o + 11u] = select(0.0, 1.0, p.ok);
  if (!p.ok) { return; }
  let b = i * 16u;
  splats[o] = p.u; splats[o + 1u] = p.v;
  splats[o + 2u] = p.a; splats[o + 3u] = p.b; splats[o + 4u] = p.c;
  splats[o + 5u] = 1.0 / (1.0 + exp(-gs[b + 7u]));
  splats[o + 6u] = p.radius; splats[o + 7u] = p.depth;
  splats[o + 8u] = gs[b + 12u]; splats[o + 9u] = gs[b + 13u]; splats[o + 10u] = gs[b + 14u];
}`;

// Shared by the forward and backward passes: whether splat o covers the pixel, and its alpha.
const BLEND = /* wgsl */ `
fn alphaAt(o: u32, px: f32, py: f32) -> vec4f {
  // Returns (alpha, dx, dy, ok).
  if (splats[o + 11u] < 0.5) { return vec4f(0.0); }
  let dx = px - splats[o]; let dy = py - splats[o + 1u];
  let r = splats[o + 6u];
  if (abs(dx) > r || abs(dy) > r) { return vec4f(0.0); }
  let power = -0.5 * (splats[o + 2u] * dx * dx + splats[o + 4u] * dy * dy) - splats[o + 3u] * dx * dy;
  if (power > 0.0) { return vec4f(0.0); }
  let alpha = min(0.99, splats[o + 5u] * exp(power));
  if (alpha < 1.0 / 255.0) { return vec4f(0.0); }
  return vec4f(alpha, dx, dy, 1.0);
}`;

const FORWARD = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> image: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read> ranges: array<u32>;
` + BLEND + /* wgsl */ `
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let w = u32(cam.size.x); let h = u32(cam.size.y);
  if (id.x >= w || id.y >= h) { return; }
  let px = f32(id.x) + 0.5; let py = f32(id.y) + 0.5;
  let tile = (id.y / 16u) * ((w + 15u) / 16u) + id.x / 16u;
  let start = ranges[tile * 2u];
  var T = 1.0;
  var rgb = vec3f(0.0);
  var last = start;
  for (var k = start; k < ranges[tile * 2u + 1u]; k++) {
    if (T <= 1e-4) { break; }
    let o = order[k] * 12u;
    let hit = alphaAt(o, px, py);
    if (hit.w < 0.5) { continue; }
    rgb += vec3f(splats[o + 8u], splats[o + 9u], splats[o + 10u]) * hit.x * T;
    T *= 1.0 - hit.x;
    last = k + 1u;
  }
  let pix = id.y * w + id.x;
  image[pix * 3u] = rgb.x; image[pix * 3u + 1u] = rgb.y; image[pix * 3u + 2u] = rgb.z;
  // Kept for the backward pass: final transmittance and how far the blend went.
  state[pix * 2u] = T;
  state[pix * 2u + 1u] = f32(last);
}`;

const BACKWARD = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> image: array<f32>;
@group(0) @binding(4) var<storage, read> state: array<f32>;
@group(0) @binding(5) var<storage, read> targetImg: array<f32>;
@group(0) @binding(6) var<storage, read_write> grads: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> ranges: array<u32>;
` + BLEND + /* wgsl */ `
fn addF(i: u32, v: f32) {
  if (v == 0.0) { return; }
  var old = atomicLoad(&grads[i]);
  loop {
    let next = bitcast<u32>(bitcast<f32>(old) + v);
    let r = atomicCompareExchangeWeak(&grads[i], old, next);
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let w = u32(cam.size.x); let h = u32(cam.size.y);
  if (id.x >= w || id.y >= h) { return; }
  let pix = id.y * w + id.x;
  let px = f32(id.x) + 0.5; let py = f32(id.y) + 0.5;
  let n = f32(cam.pixels * 3u);
  // L1 loss gradient for this pixel.
  let dL = vec3f(sign(image[pix * 3u] - targetImg[pix * 3u]), sign(image[pix * 3u + 1u] - targetImg[pix * 3u + 1u]), sign(image[pix * 3u + 2u] - targetImg[pix * 3u + 2u])) / n;
  let tile = (id.y / 16u) * ((w + 15u) / 16u) + id.x / 16u;
  let start = ranges[tile * 2u];
  var T = state[pix * 2u];
  var behind = vec3f(0.0);
  var k = u32(state[pix * 2u + 1u]);
  loop {
    if (k <= start) { break; }
    k -= 1u;
    let gi = order[k];
    let o = gi * 12u;
    let hit = alphaAt(o, px, py);
    if (hit.w < 0.5) { continue; }
    let alpha = hit.x;
    let Tk = T / (1.0 - alpha);
    let c = vec3f(splats[o + 8u], splats[o + 9u], splats[o + 10u]);
    let g = gi * 16u;
    addF(g, dL.x * alpha * Tk);
    addF(g + 1u, dL.y * alpha * Tk);
    addF(g + 2u, dL.z * alpha * Tk);
    let dAlpha = dot(dL, Tk * c - behind / (1.0 - alpha));
    behind += c * alpha * Tk;
    T = Tk;
    if (alpha < 0.99) {
      let op = splats[o + 5u];
      addF(g + 3u, dAlpha * (alpha / op) * op * (1.0 - op));
      let dPower = dAlpha * alpha;
      let dx = hit.y; let dy = hit.z;
      let A = splats[o + 2u]; let B = splats[o + 3u]; let C = splats[o + 4u];
      addF(g + 4u, dPower * (A * dx + B * dy));
      addF(g + 5u, dPower * (C * dy + B * dx));
      addF(g + 6u, dPower * -0.5 * dx * dx);
      addF(g + 7u, dPower * -dx * dy);
      addF(g + 8u, dPower * -0.5 * dy * dy);
    }
  }
}`;

// Chains each Gaussian's screen-space gradient to mean, log scale and rotation through
// the projection's Jacobian, by central differences as the reference does.
const SHAPE = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read_write> gs: array<f32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> shape: array<f32>;

fn shapeOf(i: u32) -> array<f32, 10> {
  let b = i * 16u;
  return array<f32, 10>(gs[b], gs[b + 1u], gs[b + 2u], gs[b + 4u], gs[b + 5u], gs[b + 6u], gs[b + 8u], gs[b + 9u], gs[b + 10u], gs[b + 11u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cam.count) { return; }
  let g = i * 16u;
  let sg = array<f32, 5>(grads[g + 4u], grads[g + 5u], grads[g + 6u], grads[g + 7u], grads[g + 8u]);
  let base = shapeOf(i);
  for (var k = 0u; k < 10u; k++) {
    // f32 needs a larger step than the reference's f64 one.
    let h = select(1e-3, 2e-3, k >= 6u);
    var up = base; up[k] = base[k] + h;
    var down = base; down[k] = base[k] - h;
    let a = projectShape(cam, up);
    let b = projectShape(cam, down);
    var d = 0.0;
    if (a.ok && b.ok) {
      d = (sg[0] * (a.u - b.u) + sg[1] * (a.v - b.v) + sg[2] * (a.a - b.a) + sg[3] * (a.b - b.b) + sg[4] * (a.c - b.c)) / (2.0 * h);
    }
    shape[i * 10u + k] = d;
  }
}`;

// Adam on all 14 trainable numbers of each Gaussian, with the reference's learning rates.
const ADAM = /* wgsl */ `
struct Step { t: f32, count: u32, lrColor: f32, lrOpacity: f32, lrMean: f32, lrScale: f32, lrRotation: f32, pad: f32 };
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<storage, read_write> gs: array<f32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read> shape: array<f32>;
@group(0) @binding(4) var<storage, read_write> moments: array<f32>;

// Where each of the 14 trainable numbers lives in the 16-float Gaussian.
const SLOT = array<u32, 14>(12u, 13u, 14u, 7u, 0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u, 11u);

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= step.count) { return; }
  for (var k = 0u; k < 14u; k++) {
    var g: f32;
    if (k < 4u) { g = grads[i * 16u + k]; } else { g = shape[i * 10u + k - 4u]; }
    var lr = step.lrRotation;
    if (k < 3u) { lr = step.lrColor; } else if (k == 3u) { lr = step.lrOpacity; } else if (k < 7u) { lr = step.lrMean; } else if (k < 10u) { lr = step.lrScale; }
    let m = 0.9 * moments[i * 28u + k] + 0.1 * g;
    let v = 0.999 * moments[i * 28u + 14u + k] + 0.001 * g * g;
    moments[i * 28u + k] = m;
    moments[i * 28u + 14u + k] = v;
    let update = lr * (m / (1.0 - pow(0.9, step.t))) / (sqrt(v / (1.0 - pow(0.999, step.t))) + 1e-8);
    gs[i * 16u + SLOT[k]] -= update;
  }
  let b = i * 16u;
  for (var c = 12u; c < 15u; c++) { gs[b + c] = clamp(gs[b + c], 0.0, 1.0); }
  let q = vec4f(gs[b + 8u], gs[b + 9u], gs[b + 10u], gs[b + 11u]);
  let n = max(length(q), 1e-12);
  gs[b + 8u] = q.x / n; gs[b + 9u] = q.y / n; gs[b + 10u] = q.z / n; gs[b + 11u] = q.w / n;
}`;

/**
 * Splats per 16 x 16 pixel tile: a flat list of splat indices and, per tile, its
 * [start, end) in that list. Splats are taken in depth order, so each tile's list is
 * already front to back (the same order the reference blends in).
 */
export function tileLists(splats, n, width, height) {
  const tilesX = Math.ceil(width / 16);
  const tilesY = Math.ceil(height / 16);
  const valid = [];
  for (let i = 0; i < n; i++) if (splats[i * 12 + 11] > 0.5) valid.push(i);
  valid.sort((a, b) => splats[a * 12 + 7] - splats[b * 12 + 7]);
  const bounds = (i) => {
    const u = splats[i * 12], v = splats[i * 12 + 1], r = splats[i * 12 + 6];
    return [
      Math.max(0, Math.floor((u - r - 0.5) / 16)), Math.min(tilesX - 1, Math.floor((u + r - 0.5) / 16)),
      Math.max(0, Math.floor((v - r - 0.5) / 16)), Math.min(tilesY - 1, Math.floor((v + r - 0.5) / 16)),
    ];
  };
  const counts = new Uint32Array(tilesX * tilesY);
  for (const i of valid) {
    const [x0, x1, y0, y1] = bounds(i);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) counts[y * tilesX + x]++;
  }
  const ranges = new Uint32Array(tilesX * tilesY * 2);
  let total = 0;
  for (let t = 0; t < counts.length; t++) {
    ranges[t * 2] = total;
    total += counts[t];
    ranges[t * 2 + 1] = ranges[t * 2];
  }
  const list = new Uint32Array(Math.max(1, total));
  for (const i of valid) {
    const [x0, x1, y0, y1] = bounds(i);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) list[ranges[(y * tilesX + x) * 2 + 1]++] = i;
  }
  return { list, ranges };
}

// Densify statistics: how hard each splat's 2D centre was pulled, and in how many views.
const STATS = /* wgsl */ `
struct Step { t: f32, count: u32, lrColor: f32, lrOpacity: f32, lrMean: f32, lrScale: f32, lrRotation: f32, pad: f32 };
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read> splats: array<f32>;
@group(0) @binding(3) var<storage, read_write> stats: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= step.count || splats[i * 12u + 11u] < 0.5) { return; }
  stats[i * 2u] += length(vec2f(grads[i * 16u + 4u], grads[i * 16u + 5u]));
  stats[i * 2u + 1u] += 1.0;
}`;

const SH0 = 0.28209479177387814;
const logit = (p) => Math.log(p / (1 - p));

/**
 * Starting splats from the scan's points (COLMAP points3D): one per point, small,
 * mostly transparent, in the point's colour. Training grows and refines them.
 */
export function initFromPoints({ xyz, rgb, count }, size = 0.06) {
  const out = [];
  for (let i = 0; i < count; i++) {
    out.push({
      mean: [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]],
      logScale: [Math.log(size), Math.log(size), Math.log(size)],
      rotation: [1, 0, 0, 0],
      opacityLogit: logit(0.1),
      color: [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]],
    });
  }
  return out;
}

/** Standard 3DGS .ply (binary): opens in the web tour, Brush, SuperSplat and MetalSplatter. */
export function toPly(gaussians) {
  const props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const header = ['ply', 'format binary_little_endian 1.0', `element vertex ${gaussians.length}`, ...props.map((p) => `property float ${p}`), 'end_header', ''].join('\n');
  const body = new Float32Array(gaussians.length * props.length);
  gaussians.forEach((g, i) => {
    body.set([...g.mean, 0, 0, 0, ...g.color.map((c) => (c - 0.5) / SH0), g.opacityLogit, ...g.logScale, ...g.rotation], i * props.length);
  });
  return new Blob([header, body.buffer], { type: 'application/octet-stream' });
}

/** Same rules as densifyAndPrune() in reference.ts. */
export function densify(gaussians, stats, { gradThreshold = 2e-4, sizeThreshold = 0.1, minOpacity = 0.005, maxCount = 400000 } = {}) {
  const out = [];
  const sig = (x) => 1 / (1 + Math.exp(-x));
  gaussians.forEach((g, i) => {
    if (sig(g.opacityLogit) < minOpacity) return;
    const avg = stats[i * 2 + 1] ? stats[i * 2] / stats[i * 2 + 1] : 0;
    if (avg < gradThreshold || out.length > maxCount) return void out.push(g);
    const size = Math.max(...g.logScale.map(Math.exp));
    if (size <= sizeThreshold) {
      out.push(g, structuredClone(g));
    } else {
      for (let k = 0; k < 2; k++) {
        const c = structuredClone(g);
        c.mean = g.mean.map((m, j) => m + (Math.random() - 0.5) * Math.exp(g.logScale[j]));
        c.logScale = g.logScale.map((l) => l - Math.log(1.6));
        out.push(c);
      }
    }
  });
  return out;
}

/** Reads Gaussians back from the GPU layout. */
export function unpackGaussians(data) {
  const out = [];
  for (let i = 0; i < data.length / 16; i++) {
    const d = data.subarray(i * 16, i * 16 + 16);
    out.push({ mean: [d[0], d[1], d[2]], logScale: [d[4], d[5], d[6]], opacityLogit: d[7], rotation: [d[8], d[9], d[10], d[11]], color: [d[12], d[13], d[14]] });
  }
  return out;
}

/** Packs Gaussians ({mean, logScale, rotation, opacityLogit, color}) into the GPU layout. */
export function packGaussians(gaussians) {
  const data = new Float32Array(gaussians.length * 16);
  gaussians.forEach((g, i) => {
    data.set([...g.mean, 0, ...g.logScale, g.opacityLogit, ...g.rotation, ...g.color, 0], i * 16);
  });
  return data;
}

function cameraUniform(cam, count) {
  const u = new ArrayBuffer(80);
  new Float32Array(u).set([...cam.view.slice(0, 12), cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height], 0);
  new Uint32Array(u).set([count, cam.width * cam.height], 18);
  return u;
}

export async function createRenderer() {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const pipe = (code) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pipes = { pre: pipe(PREPROCESS), fwd: pipe(FORWARD), bwd: pipe(BACKWARD), shape: pipe(SHAPE), adam: pipe(ADAM), stats: pipe(STATS) };
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buffer = (bytes) => device.createBuffer({ size: Math.max(16, bytes), usage: STORAGE });

  async function read(buf, bytes) {
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  function run(p, buffers, x, y = 1) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) }));
    pass.dispatchWorkgroups(x, y);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  /** Projects, sorts (on the CPU for now) and blends: the forward pass. */
  async function forward(gBuf, n, cam) {
    const camBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(camBuf, 0, cameraUniform(cam, n));
    const sBuf = buffer(n * 48);
    run(pipes.pre, [camBuf, gBuf, sBuf], Math.ceil(n / 64));
    const splats = await read(sBuf, n * 48);
    const { list, ranges } = tileLists(splats, n, cam.width, cam.height);
    const oBuf = buffer(list.length * 4);
    device.queue.writeBuffer(oBuf, 0, list);
    const rBuf = buffer(ranges.length * 4);
    device.queue.writeBuffer(rBuf, 0, ranges);
    const pixels = cam.width * cam.height;
    const iBuf = buffer(pixels * 12);
    const stBuf = buffer(pixels * 8);
    run(pipes.fwd, [camBuf, sBuf, oBuf, iBuf, stBuf, rBuf], Math.ceil(cam.width / 8), Math.ceil(cam.height / 8));
    return { camBuf, sBuf, oBuf, iBuf, stBuf, rBuf, pixels, visible: list.length };
  }

  /** Renders the Gaussians from one camera; returns RGB floats (width * height * 3). */
  async function render(gaussians, cam) {
    const gBuf = buffer(gaussians.length * 64);
    device.queue.writeBuffer(gBuf, 0, packGaussians(gaussians));
    const f = await forward(gBuf, gaussians.length, cam);
    return read(f.iBuf, f.pixels * 12);
  }

  /**
   * Forward and backward against a target image. Returns the image, the per-Gaussian
   * gradients (colour, opacity logit, screen) and the shape gradients (10 per Gaussian).
   */
  async function gradients(gaussians, cam, target) {
    const n = gaussians.length;
    const gBuf = buffer(n * 64);
    device.queue.writeBuffer(gBuf, 0, packGaussians(gaussians));
    const f = await forward(gBuf, n, cam);
    const tBuf = buffer(f.pixels * 12);
    device.queue.writeBuffer(tBuf, 0, Float32Array.from(target));
    const gradBuf = buffer(n * 64);
    run(pipes.bwd, [f.camBuf, f.sBuf, f.oBuf, f.iBuf, f.stBuf, tBuf, gradBuf, f.rBuf], Math.ceil(cam.width / 8), Math.ceil(cam.height / 8));
    const shapeBuf = buffer(n * 40);
    run(pipes.shape, [f.camBuf, gBuf, gradBuf, shapeBuf], Math.ceil(n / 64));
    return { image: await read(f.iBuf, f.pixels * 12), grads: await read(gradBuf, n * 64), shape: await read(shapeBuf, n * 40) };
  }

  /**
   * Trains Gaussians on photos: each step takes the next view (camera + target image),
   * renders, back-propagates the L1 loss and applies Adam, all on the GPU.
   * Returns the trained Gaussians and the loss of each step.
   */
  async function train(gaussians, views, steps, lr = { color: 0.05, opacity: 0.05, mean: 0.002, scale: 0.01, rotation: 0.005 }) {
    const n = gaussians.length;
    const gBuf = buffer(n * 64);
    device.queue.writeBuffer(gBuf, 0, packGaussians(gaussians));
    const moments = buffer(n * 28 * 4);
    const stepBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const targets = views.map((v) => {
      const b = buffer(v.target.length * 4);
      device.queue.writeBuffer(b, 0, Float32Array.from(v.target));
      return b;
    });
    const losses = [];
    for (let t = 1; t <= steps; t++) {
      const view = views[(t - 1) % views.length];
      const f = await forward(gBuf, n, view.cam);
      const gradBuf = buffer(n * 64);
      run(pipes.bwd, [f.camBuf, f.sBuf, f.oBuf, f.iBuf, f.stBuf, targets[(t - 1) % views.length], gradBuf, f.rBuf], Math.ceil(view.cam.width / 8), Math.ceil(view.cam.height / 8));
      const shapeBuf = buffer(n * 40);
      run(pipes.shape, [f.camBuf, gBuf, gradBuf, shapeBuf], Math.ceil(n / 64));
      const stepData = new ArrayBuffer(32);
      new Float32Array(stepData).set([t, 0, lr.color, lr.opacity, lr.mean, lr.scale, lr.rotation, 0]);
      new Uint32Array(stepData)[1] = n;
      device.queue.writeBuffer(stepBuf, 0, stepData);
      run(pipes.adam, [stepBuf, gBuf, gradBuf, shapeBuf, moments], Math.ceil(n / 64));
      const image = await read(f.iBuf, f.pixels * 12);
      let loss = 0;
      for (let i = 0; i < image.length; i++) loss += Math.abs(image[i] - view.target[i]);
      losses.push(loss / image.length);
      for (const b of [f.sBuf, f.oBuf, f.iBuf, f.stBuf, f.rBuf, gradBuf, shapeBuf]) b.destroy();
    }
    return { gaussians: unpackGaussians(await read(gBuf, n * 64)), losses };
  }

  /**
   * Full training on many photos: random view per step, densify every `densifyEvery`
   * steps between `densifyFrom` and `densifyUntil`, progress through `onProgress`
   * (step, loss, splat count). `shouldStop` ends early. Returns the trained splats.
   * Views hold a camera and an RGB Uint8 image (as decoded from the photos).
   */
  async function trainScene(start, views, { steps = 3000, densifyEvery = 100, densifyFrom = 300, densifyUntil = 2000, onProgress, shouldStop, lr } = {}) {
    lr = { color: 0.025, opacity: 0.05, mean: 0.002, scale: 0.005, rotation: 0.001, ...lr };
    let gaussians = start;
    let n = gaussians.length;
    let gBuf, moments, statsBuf;
    const upload = () => {
      for (const b of [gBuf, moments, statsBuf]) b?.destroy();
      n = gaussians.length;
      gBuf = buffer(n * 64);
      device.queue.writeBuffer(gBuf, 0, packGaussians(gaussians));
      moments = buffer(n * 28 * 4);
      statsBuf = buffer(n * 8);
    };
    upload();
    const stepBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    let lossSum = 0, lossCount = 0;
    for (let t = 1; t <= steps; t++) {
      if (shouldStop?.()) break;
      const view = views[Math.floor(Math.random() * views.length)];
      const f = await forward(gBuf, n, view.cam);
      const tBuf = buffer(f.pixels * 12);
      device.queue.writeBuffer(tBuf, 0, Float32Array.from(view.image, (v) => v / 255));
      const gradBuf = buffer(n * 64);
      run(pipes.bwd, [f.camBuf, f.sBuf, f.oBuf, f.iBuf, f.stBuf, tBuf, gradBuf, f.rBuf], Math.ceil(view.cam.width / 8), Math.ceil(view.cam.height / 8));
      const shapeBuf = buffer(n * 40);
      run(pipes.shape, [f.camBuf, gBuf, gradBuf, shapeBuf], Math.ceil(n / 64));
      const stepData = new ArrayBuffer(32);
      new Float32Array(stepData).set([t, 0, lr.color, lr.opacity, lr.mean, lr.scale, lr.rotation, 0]);
      new Uint32Array(stepData)[1] = n;
      device.queue.writeBuffer(stepBuf, 0, stepData);
      run(pipes.stats, [stepBuf, gradBuf, f.sBuf, statsBuf], Math.ceil(n / 64));
      run(pipes.adam, [stepBuf, gBuf, gradBuf, shapeBuf, moments], Math.ceil(n / 64));
      if (t % 25 === 0) {
        const image = await read(f.iBuf, f.pixels * 12);
        let loss = 0;
        for (let i = 0; i < image.length; i++) loss += Math.abs(image[i] - view.image[i] / 255);
        lossSum += loss / image.length;
        lossCount++;
      }
      for (const b of [f.camBuf, f.sBuf, f.oBuf, f.iBuf, f.stBuf, f.rBuf, tBuf, gradBuf, shapeBuf]) b.destroy();
      if (t >= densifyFrom && t <= densifyUntil && t % densifyEvery === 0) {
        // 3DGS sets its threshold (0.0002) on gradients in NDC units; ours are per pixel.
        const gradThreshold = 2e-4 / (0.5 * views[0].cam.width);
        gaussians = densify(unpackGaussians(await read(gBuf, n * 64)), await read(statsBuf, n * 8), { gradThreshold });
        upload();
      }
      if (t % 50 === 0) {
        onProgress?.({ step: t, loss: lossCount ? lossSum / lossCount : NaN, count: n });
        lossSum = 0;
        lossCount = 0;
        await new Promise((r) => setTimeout(r)); // let the page redraw
      }
    }
    gaussians = unpackGaussians(await read(gBuf, n * 64));
    for (const b of [gBuf, moments, statsBuf, stepBuf]) b.destroy();
    return gaussians;
  }

  return { device, render, gradients, train, trainScene };
}

/** Reads a binary 3DGS .ply (as written by toPly, Brush or the reference tools). */
export function fromPly(buffer) {
  const bytes = new Uint8Array(buffer);
  const headerEnd = new TextDecoder().decode(bytes.subarray(0, 4096)).indexOf('end_header\n') + 'end_header\n'.length;
  const header = new TextDecoder().decode(bytes.subarray(0, headerEnd));
  const count = Number(/element vertex (\d+)/.exec(header)[1]);
  const props = [...header.matchAll(/property float (\w+)/g)].map((m) => m[1]);
  const data = new Float32Array(buffer.slice(headerEnd, headerEnd + count * props.length * 4));
  const at = (i, name) => data[i * props.length + props.indexOf(name)];
  const SH0 = 0.28209479177387814;
  return Array.from({ length: count }, (_, i) => ({
    mean: [at(i, 'x'), at(i, 'y'), at(i, 'z')],
    logScale: [at(i, 'scale_0'), at(i, 'scale_1'), at(i, 'scale_2')],
    rotation: [at(i, 'rot_0'), at(i, 'rot_1'), at(i, 'rot_2'), at(i, 'rot_3')],
    opacityLogit: at(i, 'opacity'),
    color: [0, 1, 2].map((k) => Math.min(1, Math.max(0, at(i, `f_dc_${k}`) * SH0 + 0.5))),
  }));
}
