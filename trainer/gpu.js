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
@group(0) @binding(4) var<storage, read> state: array<f32>;
@group(0) @binding(5) var<storage, read> lossGrad: array<f32>;
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
  // Loss gradient for this pixel (from the LOSS_GRAD pass).
  let dL = vec3f(lossGrad[pix * 3u], lossGrad[pix * 3u + 1u], lossGrad[pix * 3u + 2u]);
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

// The same backward pass, one workgroup per 16 x 16 tile. All pixels of a tile walk
// the tile's list together (back to front), and each splat's contributions are first
// summed across the subgroup, so a splat gets a handful of atomic adds per tile instead
// of one per pixel. Needs the WebGPU "subgroups" feature; otherwise BACKWARD is used.
const BACKWARD_SG = 'enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n' + COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(4) var<storage, read> state: array<f32>;
@group(0) @binding(5) var<storage, read> lossGrad: array<f32>;
@group(0) @binding(6) var<storage, read_write> grads: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> ranges: array<u32>;
var<workgroup> lastAll: atomic<u32>;
var<workgroup> startShared: u32;
var<workgroup> endShared: u32;
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

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let w = u32(cam.size.x); let h = u32(cam.size.y);
  let tile = wg.y * ((w + 15u) / 16u) + wg.x;
  let x = wg.x * 16u + lid.x; let y = wg.y * 16u + lid.y;
  let inside = x < w && y < h;
  let px = f32(x) + 0.5; let py = f32(y) + 0.5;
  let pix = y * w + x;
  var T = 0.0;
  var myLast = 0u;
  var dL = vec3f(0.0);
  if (inside) {
    T = state[pix * 2u];
    myLast = u32(state[pix * 2u + 1u]);
    dL = vec3f(lossGrad[pix * 3u], lossGrad[pix * 3u + 1u], lossGrad[pix * 3u + 2u]);
  }
  if (li == 0u) { atomicStore(&lastAll, 0u); startShared = ranges[tile * 2u]; }
  workgroupBarrier();
  atomicMax(&lastAll, myLast);
  workgroupBarrier();
  if (li == 0u) { endShared = max(atomicLoad(&lastAll), startShared); }
  let start = workgroupUniformLoad(&startShared);
  let end = workgroupUniformLoad(&endShared);
  var behind = vec3f(0.0);
  for (var k = end; k > start; k--) {
    let gi = order[k - 1u];
    let o = gi * 12u;
    var g0 = 0.0; var g1 = 0.0; var g2 = 0.0; var g3 = 0.0;
    var g4 = 0.0; var g5 = 0.0; var g6 = 0.0; var g7 = 0.0; var g8 = 0.0;
    if (inside && k - 1u < myLast) {
      let hit = alphaAt(o, px, py);
      if (hit.w > 0.5) {
        let alpha = hit.x;
        let Tk = T / (1.0 - alpha);
        let c = vec3f(splats[o + 8u], splats[o + 9u], splats[o + 10u]);
        g0 = dL.x * alpha * Tk; g1 = dL.y * alpha * Tk; g2 = dL.z * alpha * Tk;
        let dAlpha = dot(dL, Tk * c - behind / (1.0 - alpha));
        behind += c * alpha * Tk;
        T = Tk;
        if (alpha < 0.99) {
          let op = splats[o + 5u];
          g3 = dAlpha * (alpha / op) * op * (1.0 - op);
          let dPower = dAlpha * alpha;
          let dx = hit.y; let dy = hit.z;
          let A = splats[o + 2u]; let B = splats[o + 3u]; let C = splats[o + 4u];
          g4 = dPower * (A * dx + B * dy);
          g5 = dPower * (C * dy + B * dx);
          g6 = dPower * -0.5 * dx * dx;
          g7 = dPower * -dx * dy;
          g8 = dPower * -0.5 * dy * dy;
        }
      }
    }
    let s0 = subgroupAdd(g0); let s1 = subgroupAdd(g1); let s2 = subgroupAdd(g2);
    let s3 = subgroupAdd(g3); let s4 = subgroupAdd(g4); let s5 = subgroupAdd(g5);
    let s6 = subgroupAdd(g6); let s7 = subgroupAdd(g7); let s8 = subgroupAdd(g8);
    if (subgroupElect()) {
      let b = gi * 16u;
      addF(b, s0); addF(b + 1u, s1); addF(b + 2u, s2); addF(b + 3u, s3); addF(b + 4u, s4);
      addF(b + 5u, s5); addF(b + 6u, s6); addF(b + 7u, s7); addF(b + 8u, s8);
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
// Loss = (1 - lambda) * L1 + lambda * (1 - SSIM), as in 3D Gaussian Splatting, with
// SSIM over 7x7 windows that lie inside the image. SSIM_COEF writes, per window
// centre and channel, the derivatives of that window's SSIM by its mean, mean of
// squares and mean of products; LOSS_GRAD sums them over the windows covering each
// pixel to get dLoss/dPixel. lambda = 0 gives plain L1.
const SSIM_COEF = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> targetImg: array<f32>;
@group(0) @binding(3) var<storage, read_write> coef: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let w = u32(cam.size.x); let h = u32(cam.size.y);
  if (id.x >= w || id.y >= h) { return; }
  let pix = id.y * w + id.x;
  let inside = id.x >= 3u && id.y >= 3u && id.x + 3u < w && id.y + 3u < h;
  for (var c = 0u; c < 3u; c++) {
    var a = 0.0; var b = 0.0; var d = 0.0;
    if (inside) {
      var mx = 0.0; var my = 0.0; var sxx = 0.0; var syy = 0.0; var sxy = 0.0;
      for (var dy = 0u; dy < 7u; dy++) {
        for (var dx = 0u; dx < 7u; dx++) {
          let q = ((id.y + dy - 3u) * w + id.x + dx - 3u) * 3u + c;
          let x = image[q]; let y = targetImg[q];
          mx += x; my += y; sxx += x * x; syy += y * y; sxy += x * y;
        }
      }
      mx /= 49.0; my /= 49.0; sxx /= 49.0; syy /= 49.0; sxy /= 49.0;
      let c1 = 0.0001; let c2 = 0.0009;
      let a1 = 2.0 * mx * my + c1;
      let a2 = 2.0 * (sxy - mx * my) + c2;
      let b1 = mx * mx + my * my + c1;
      let b2 = (sxx - mx * mx) + (syy - my * my) + c2;
      let ssim = a1 * a2 / (b1 * b2);
      a = ssim * (2.0 * my / a1 - 2.0 * my / a2 - 2.0 * mx / b1 + 2.0 * mx / b2);
      b = -ssim / b2;
      d = 2.0 * ssim / a2;
    }
    coef[pix * 9u + c * 3u] = a; coef[pix * 9u + c * 3u + 1u] = b; coef[pix * 9u + c * 3u + 2u] = d;
  }
}`;

const LOSS_GRAD = COMMON + /* wgsl */ `
struct LossParams { lambda: f32, windows: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> targetImg: array<f32>;
@group(0) @binding(3) var<storage, read> coef: array<f32>;
@group(0) @binding(4) var<storage, read_write> lossGrad: array<f32>;
@group(0) @binding(5) var<uniform> params: LossParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let w = i32(cam.size.x); let h = i32(cam.size.y);
  let x0 = i32(id.x); let y0 = i32(id.y);
  if (x0 >= w || y0 >= h) { return; }
  let pix = u32(y0 * w + x0);
  let n = f32(cam.pixels * 3u);
  for (var c = 0u; c < 3u; c++) {
    let x = image[pix * 3u + c]; let y = targetImg[pix * 3u + c];
    var g = (1.0 - params.lambda) * sign(x - y) / n;
    if (params.lambda > 0.0 && params.windows > 0.0) {
      var sa = 0.0; var sb = 0.0; var sd = 0.0;
      for (var dy = -3; dy <= 3; dy++) {
        for (var dx = -3; dx <= 3; dx++) {
          let cx = x0 + dx; let cy = y0 + dy;
          if (cx < 0 || cy < 0 || cx >= w || cy >= h) { continue; }
          let k = u32(cy * w + cx) * 9u + c * 3u;
          sa += coef[k]; sb += coef[k + 1u]; sd += coef[k + 2u];
        }
      }
      // d(mean SSIM)/dx, and the loss uses 1 - SSIM.
      let dSsim = (sa + 2.0 * x * sb + y * sd) / 49.0 / params.windows;
      g -= params.lambda * dSsim;
    }
    lossGrad[pix * 3u + c] = g;
  }
}`;

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

/**
 * The compact .splat format (32 bytes a splat instead of 68 in .ply): position and
 * scale as floats, colour and opacity as bytes, rotation as 4 bytes. Largest and most
 * opaque first, as viewers expect. Read by the tour page's splat viewer.
 */
export function toSplat(all) {
  // A splat with a non-finite value would break the viewer's sort; leave it out.
  const gaussians = all.filter((g) => [...g.mean, ...g.logScale, g.opacityLogit, ...g.rotation, ...g.color].every(Number.isFinite));
  const weight = (g) => Math.exp(g.logScale[0] + g.logScale[1] + g.logScale[2]) / (1 + Math.exp(-g.opacityLogit));
  const order = gaussians.map((g, i) => [weight(g), i]).sort((a, b) => b[0] - a[0]);
  const out = new ArrayBuffer(gaussians.length * 32);
  const f = new Float32Array(out);
  const u = new Uint8Array(out);
  const byte = (v) => Math.max(0, Math.min(255, Math.round(v)));
  order.forEach(([, i], k) => {
    const g = gaussians[i];
    f.set([...g.mean, ...g.logScale.map(Math.exp)], k * 8);
    const q = g.rotation;
    const len = Math.hypot(...q) || 1;
    u.set([...g.color.map((c) => byte(c * 255)), byte(255 / (1 + Math.exp(-g.opacityLogit))), ...q.map((v) => byte((v / len) * 128 + 128))], k * 32 + 24);
  });
  return new Blob([out], { type: 'application/octet-stream' });
}

/** Same rules as densifyAndPrune() in reference.ts. */
/**
 * How far the cameras spread (3DGS's "scene extent"): 1.1 times the largest distance
 * of a camera from their mean. Views carry world-to-camera rows (R | t), so a camera
 * sits at -R^T t.
 */
export function sceneExtent(views) {
  const centres = views.map(({ cam: { view: m } }) => [0, 1, 2].map((j) => -(m[j] * m[3] + m[4 + j] * m[7] + m[8 + j] * m[11])));
  const mean = [0, 1, 2].map((j) => centres.reduce((a, c) => a + c[j], 0) / centres.length);
  return 1.1 * Math.max(1e-3, ...centres.map((c) => Math.hypot(c[0] - mean[0], c[1] - mean[1], c[2] - mean[2])));
}

/**
 * Adds splats where the image gradient is high (clone small ones, split large ones) and
 * removes transparent ones and, with `maxSize`, ones larger than that in world units:
 * those "floaters" explain the photos but cover every other viewpoint.
 */
export function densify(gaussians, stats, { gradThreshold = 2e-4, sizeThreshold = 0.1, minOpacity = 0.005, maxCount = 400000, maxSize = Infinity } = {}) {
  const out = [];
  const sig = (x) => 1 / (1 + Math.exp(-x));
  // Each clone or split adds one splat; stop adding once the total would pass maxCount.
  let room = maxCount - gaussians.length;
  gaussians.forEach((g, i) => {
    if (sig(g.opacityLogit) < minOpacity) return;
    if (Math.max(...g.logScale.map(Math.exp)) > maxSize) return;
    const avg = stats[i * 2 + 1] ? stats[i * 2] / stats[i * 2 + 1] : 0;
    if (avg < gradThreshold || room <= 0) return void out.push(g);
    room--;
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

// Tile lists built on the GPU (no read-back): each splat appends itself to the
// 16 x 16 tiles it covers (up to TILE_CAP per tile), then each tile sorts its list by
// depth with a bitonic sort in workgroup memory and publishes its [start, end).
const TILE_CAP = 1024;

const BIN = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> lists: array<u32>;
const CAP = ${TILE_CAP}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cam.count) { return; }
  let o = i * 12u;
  if (splats[o + 11u] < 0.5) { return; }
  let u = splats[o]; let v = splats[o + 1u]; let r = splats[o + 6u];
  let tilesX = (u32(cam.size.x) + 15u) / 16u;
  let tilesY = (u32(cam.size.y) + 15u) / 16u;
  let x0 = u32(clamp(floor((u - r - 0.5) / 16.0), 0.0, f32(tilesX - 1u)));
  let x1 = u32(clamp(floor((u + r - 0.5) / 16.0), -1.0, f32(tilesX - 1u)) + 1.0);
  let y0 = u32(clamp(floor((v - r - 0.5) / 16.0), 0.0, f32(tilesY - 1u)));
  let y1 = u32(clamp(floor((v + r - 0.5) / 16.0), -1.0, f32(tilesY - 1u)) + 1.0);
  for (var ty = y0; ty < y1; ty++) {
    for (var tx = x0; tx < x1; tx++) {
      let t = ty * tilesX + tx;
      let k = atomicAdd(&counts[t], 1u);
      if (k < CAP) { lists[t * CAP + k] = i; }
    }
  }
}`;

const SORT = /* wgsl */ `
@group(0) @binding(0) var<storage, read> splats: array<f32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> lists: array<u32>;
@group(0) @binding(3) var<storage, read_write> ranges: array<u32>;
const CAP = ${TILE_CAP}u;
var<workgroup> keys: array<f32, ${TILE_CAP}>;
var<workgroup> vals: array<u32, ${TILE_CAP}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wg.x;
  let n = min(counts[t], CAP);
  for (var k = 0u; k < CAP / 256u; k++) {
    let j = lid.x + k * 256u;
    if (j < n) { let s = lists[t * CAP + j]; keys[j] = splats[s * 12u + 7u]; vals[j] = s; }
    else { keys[j] = 3.0e38; vals[j] = 0u; }
  }
  workgroupBarrier();
  for (var size = 2u; size <= CAP; size = size * 2u) {
    for (var stride = size / 2u; stride > 0u; stride = stride / 2u) {
      for (var k = 0u; k < CAP / 256u; k++) {
        let i = lid.x + k * 256u;
        let j = i ^ stride;
        if (j > i) {
          let up = (i & size) == 0u;
          let a = keys[i]; let b = keys[j];
          // Equal depths keep splat-index order, like the reference's stable sort.
          let swap = select(a < b || (a == b && vals[i] < vals[j]), a > b || (a == b && vals[i] > vals[j]), up);
          if (swap) { keys[i] = b; keys[j] = a; let x = vals[i]; vals[i] = vals[j]; vals[j] = x; }
        }
      }
      workgroupBarrier();
    }
  }
  for (var k = 0u; k < CAP / 256u; k++) {
    let j = lid.x + k * 256u;
    if (j < n) { lists[t * CAP + j] = vals[j]; }
  }
  if (lid.x == 0u) { ranges[t * 2u] = t * CAP; ranges[t * 2u + 1u] = t * CAP + n; }
}`;

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

/** `gpuTiles: false` builds tile lists on the CPU (the original path, kept for checks). */
export async function createRenderer({ gpuTiles = true } = {}) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  // Subgroups make the backward pass much cheaper; older GPUs and browsers fall back.
  const subgroups = adapter.features.has('subgroups');
  const device = await adapter.requestDevice({ requiredFeatures: subgroups ? ['subgroups'] : [] });
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));
  const pipe = (code) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pipes = { pre: pipe(PREPROCESS), fwd: pipe(FORWARD), bwd: pipe(BACKWARD), bwdSg: subgroups ? pipe(BACKWARD_SG) : null, shape: pipe(SHAPE), adam: pipe(ADAM), stats: pipe(STATS), bin: pipe(BIN), sort: pipe(SORT), ssim: pipe(SSIM_COEF), lossGrad: pipe(LOSS_GRAD) };
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
    pass.setBindGroup(0, device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.flatMap((b, binding) => (b ? [{ binding, resource: { buffer: b } }] : [])) }));
    pass.dispatchWorkgroups(x, y);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  /**
   * dLoss/dPixel for a rendered image against its photo: L1, plus SSIM with weight
   * `lambda`. With `pooled`, the buffers are reused (training).
   */
  function lossGradient(f, targetBuf, cam, lambda, pooled = false) {
    const get = (name, bytes, usage) => (pooled ? scratch(name, bytes, usage) : device.createBuffer({ size: Math.max(16, bytes), usage: usage ?? STORAGE }));
    const coefBuf = get('ssimCoef', f.pixels * 36);
    const gradOut = get('lossGrad', f.pixels * 12);
    const paramBuf = get('lossParams', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const windows = Math.max(0, cam.width - 6) * Math.max(0, cam.height - 6) * 3;
    device.queue.writeBuffer(paramBuf, 0, new Float32Array([lambda, windows, 0, 0]));
    const x = Math.ceil(cam.width / 8), y = Math.ceil(cam.height / 8);
    if (lambda > 0) run(pipes.ssim, [f.camBuf, f.iBuf, targetBuf, coefBuf], x, y);
    run(pipes.lossGrad, [f.camBuf, f.iBuf, targetBuf, coefBuf, gradOut, paramBuf], x, y);
    return { gradOut, temporary: pooled ? [] : [coefBuf, gradOut, paramBuf] };
  }

  /** The backward pass: per tile with subgroup sums when available, else per pixel. */
  function backward(buffers, cam) {
    // Slot 3 (the rendered image) is only read by the loss pass now.
    buffers = buffers.map((b, i) => (i === 3 ? null : b));
    if (pipes.bwdSg) run(pipes.bwdSg, buffers, Math.ceil(cam.width / 16), Math.ceil(cam.height / 16));
    else run(pipes.bwd, buffers, Math.ceil(cam.width / 8), Math.ceil(cam.height / 8));
  }

  // Buffers reused between training steps (grown when the splat count grows).
  const pool = new Map();
  function scratch(name, bytes, usage = STORAGE) {
    const have = pool.get(name);
    if (have && have.size >= bytes) return have;
    have?.destroy();
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes * 1.25 / 4) * 4), usage });
    pool.set(name, b);
    return b;
  }
  function clear(buf) {
    const enc = device.createCommandEncoder();
    enc.clearBuffer(buf);
    device.queue.submit([enc.finish()]);
  }

  /**
   * Projects, bins into tiles and blends: the forward pass. With `pooled`, the
   * buffers are reused (training); otherwise fresh ones are made (one-off renders).
   */
  async function forward(gBuf, n, cam, pooled = false) {
    const get = (name, bytes) => (pooled ? scratch(name, bytes) : buffer(bytes));
    const camBuf = pooled ? scratch('cam', 80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST) : device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(camBuf, 0, cameraUniform(cam, n));
    const sBuf = get('splats', n * 48);
    run(pipes.pre, [camBuf, gBuf, sBuf], Math.ceil(n / 64));
    const tiles = Math.ceil(cam.width / 16) * Math.ceil(cam.height / 16);
    let oBuf, rBuf;
    if (gpuTiles) {
      const counts = get('counts', tiles * 4);
      if (pooled) clear(counts);
      oBuf = get('lists', tiles * TILE_CAP * 4);
      rBuf = get('ranges', tiles * 8);
      run(pipes.bin, [camBuf, sBuf, counts, oBuf], Math.ceil(n / 64));
      run(pipes.sort, [sBuf, counts, oBuf, rBuf], tiles);
      if (!pooled) counts.destroy();
    } else {
      const splats = await read(sBuf, n * 48);
      const { list, ranges } = tileLists(splats, n, cam.width, cam.height);
      oBuf = buffer(list.length * 4);
      device.queue.writeBuffer(oBuf, 0, list);
      rBuf = buffer(ranges.length * 4);
      device.queue.writeBuffer(rBuf, 0, ranges);
    }
    const pixels = cam.width * cam.height;
    const iBuf = get('image', pixels * 12);
    const stBuf = get('state', pixels * 8);
    run(pipes.fwd, [camBuf, sBuf, oBuf, iBuf, stBuf, rBuf], Math.ceil(cam.width / 8), Math.ceil(cam.height / 8));
    return { camBuf, sBuf, oBuf, iBuf, stBuf, rBuf, pixels };
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
  async function gradients(gaussians, cam, target, ssim = 0) {
    const n = gaussians.length;
    const gBuf = buffer(n * 64);
    device.queue.writeBuffer(gBuf, 0, packGaussians(gaussians));
    const f = await forward(gBuf, n, cam);
    const tBuf = buffer(f.pixels * 12);
    device.queue.writeBuffer(tBuf, 0, Float32Array.from(target));
    const gradBuf = buffer(n * 64);
    const lossOut = lossGradient(f, tBuf, cam, ssim);
    backward([f.camBuf, f.sBuf, f.oBuf, f.iBuf, f.stBuf, lossOut.gradOut, gradBuf, f.rBuf], cam);
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
      const lossOut = lossGradient(f, targets[(t - 1) % views.length], view.cam, 0);
      run(pipes.bwd, [f.camBuf, f.sBuf, f.oBuf, null, f.stBuf, lossOut.gradOut, gradBuf, f.rBuf], Math.ceil(view.cam.width / 8), Math.ceil(view.cam.height / 8));
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
      for (const b of [f.sBuf, f.oBuf, f.iBuf, f.stBuf, f.rBuf, gradBuf, shapeBuf, ...lossOut.temporary]) b.destroy();
    }
    return { gaussians: unpackGaussians(await read(gBuf, n * 64)), losses };
  }

  /**
   * Full training on many photos: random view per step, densify every `densifyEvery`
   * steps between `densifyFrom` and `densifyUntil`, progress through `onProgress`
   * (step, loss, splat count). `shouldStop` ends early. Returns the trained splats.
   * Views hold a camera and an RGB Uint8 image (as decoded from the photos).
   */
  // Defaults measured with scene-test.html (21 Oct, 800 steps, held-out views): splitting
  // at 1 % of the scene extent raised held-out PSNR from about 8.5 to 10.8 dB. SSIM
  // (+17 % time), pruning large splats and the opacity reset showed no gain there, so
  // they are off; real photos may differ.
  async function trainScene(start, views, { steps = 3000, densifyEvery = 100, densifyFrom = 300, densifyUntil = 2000, maxSplats = 400000, ssim = 0, prune = false, opacityReset = false, onProgress, shouldStop, lr } = {}) {
    lr = { color: 0.025, opacity: 0.05, mean: 0.002, scale: 0.005, rotation: 0.001, ...lr };
    let gaussians = start;
    let n = gaussians.length;
    // 3DGS: split splats above 1 % of the scene extent, drop those above 10 %.
    const extent = sceneExtent(views);
    // 3DGS also lowers every opacity now and then while densifying: splats that matter
    // recover, and haze in front of the cameras fades and is pruned. Here once, halfway.
    const resetAt = opacityReset ? Math.round((densifyFrom + densifyUntil) / 2 / densifyEvery) * densifyEvery : -1;
    const lowOpacity = Math.log(0.01 / 0.99);
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
      const f = await forward(gBuf, n, view.cam, true);
      const tBuf = scratch('target', f.pixels * 12);
      device.queue.writeBuffer(tBuf, 0, Float32Array.from(view.image, (v) => v / 255));
      const gradBuf = scratch('grads', n * 64);
      clear(gradBuf);
      const lossOut = lossGradient(f, tBuf, view.cam, ssim, true);
      backward([f.camBuf, f.sBuf, f.oBuf, f.iBuf, f.stBuf, lossOut.gradOut, gradBuf, f.rBuf], view.cam);
      const shapeBuf = scratch('shape', n * 40);
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
      if (t >= densifyFrom && t <= densifyUntil && t % densifyEvery === 0) {
        // 3DGS sets its threshold (0.0002) on gradients in NDC units; ours are per pixel.
        const gradThreshold = 2e-4 / (0.5 * views[0].cam.width);
        gaussians = densify(unpackGaussians(await read(gBuf, n * 64)), await read(statsBuf, n * 8), { gradThreshold, maxCount: maxSplats, sizeThreshold: 0.01 * extent, maxSize: prune ? 0.1 * extent : Infinity });
        if (t === resetAt) for (const g of gaussians) g.opacityLogit = Math.min(g.opacityLogit, lowOpacity);
        upload();
      }
      if (t % 50 === 0) {
        // Awaited, so the caller can pause training (the phone does when it gets hot).
        await onProgress?.({ step: t, loss: lossCount ? lossSum / lossCount : NaN, count: n });
        lossSum = 0;
        lossCount = 0;
        await new Promise((r) => setTimeout(r)); // let the page redraw
      }
    }
    gaussians = unpackGaussians(await read(gBuf, n * 64));
    for (const b of [gBuf, moments, statsBuf, stepBuf]) b.destroy();
    return gaussians;
  }

  /** dLoss/dPixel for an image and its photo (for checks against finite differences). */
  async function lossGradientOf(image, target, width, height, lambda) {
    const pixels = width * height;
    const camData = new ArrayBuffer(80);
    new Float32Array(camData).set([width, height], 16);
    new Uint32Array(camData)[19] = pixels;
    const camBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(camBuf, 0, camData);
    const iBuf = buffer(pixels * 12);
    device.queue.writeBuffer(iBuf, 0, Float32Array.from(image));
    const tBuf = buffer(pixels * 12);
    device.queue.writeBuffer(tBuf, 0, Float32Array.from(target));
    const loss = lossGradient({ camBuf, iBuf, pixels }, tBuf, { width, height }, lambda);
    const out = await read(loss.gradOut, pixels * 12);
    for (const b of [camBuf, iBuf, tBuf, ...loss.temporary]) b.destroy();
    return out;
  }

  return { device, render, gradients, train, trainScene, lossGradientOf };
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
