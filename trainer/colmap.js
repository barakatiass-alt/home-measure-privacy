// Reads a COLMAP text dataset (cameras.txt, images.txt, points3D.txt), as written by the
// app's "Scan photos for photoreal 3D" export, into cameras for the GPU trainer.

const lines = (text) => text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith('#'));

/** cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS... (PINHOLE: fx fy cx cy; SIMPLE_PINHOLE: f cx cy). */
export function parseCameras(text) {
  const cams = new Map();
  for (const l of lines(text)) {
    const [id, model, w, h, ...p] = l.split(/\s+/);
    const n = p.map(Number);
    const [fx, fy, cx, cy] = model === 'SIMPLE_PINHOLE' || model === 'SIMPLE_RADIAL' ? [n[0], n[0], n[1], n[2]] : n;
    cams.set(Number(id), { width: Number(w), height: Number(h), fx, fy, cx, cy });
  }
  return cams;
}

/** images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, each followed by a line of 2D points. */
export function parseImages(text) {
  const out = [];
  const all = text.split(/\r?\n/).filter((l) => !l.startsWith('#'));
  for (let i = 0; i < all.length; i++) {
    const parts = all[i].trim().split(/\s+/);
    if (parts.length < 10) continue;
    const [id, qw, qx, qy, qz, tx, ty, tz, cam, ...name] = parts;
    out.push({ id: Number(id), q: [qw, qx, qy, qz].map(Number), t: [tx, ty, tz].map(Number), cameraId: Number(cam), name: name.join(' ') });
    i++; // skip the 2D points line
  }
  return out;
}

/** points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK... */
export function parsePoints(text) {
  const ls = lines(text);
  const xyz = new Float32Array(ls.length * 3);
  const rgb = new Float32Array(ls.length * 3);
  ls.forEach((l, i) => {
    const p = l.split(/\s+/).map(Number);
    xyz.set([p[1], p[2], p[3]], i * 3);
    rgb.set([p[4] / 255, p[5] / 255, p[6] / 255], i * 3);
  });
  return { xyz, rgb, count: ls.length };
}

/** World-to-camera [R | t] (row-major 3 x 4) from COLMAP's quaternion and translation. */
export function viewFromPose([w, x, y, z], [tx, ty, tz]) {
  const n = Math.hypot(w, x, y, z) || 1;
  [w, x, y, z] = [w / n, x / n, y / n, z / n];
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), tx,
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), ty,
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), tz,
  ];
}

/** Trainer camera for an image, scaled to `width` pixels across. */
export function trainerCamera(image, camera, width) {
  const s = width / camera.width;
  return {
    view: viewFromPose(image.q, image.t),
    fx: camera.fx * s, fy: camera.fy * s, cx: camera.cx * s, cy: camera.cy * s,
    width: Math.round(camera.width * s), height: Math.round(camera.height * s),
  };
}

/** Camera centre in world space (for picking starting views and sizing the scene). */
export function cameraCentre(view) {
  // C = -R^T t
  const [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz] = view;
  return [-(r00 * tx + r10 * ty + r20 * tz), -(r01 * tx + r11 * ty + r21 * tz), -(r02 * tx + r12 * ty + r22 * tz)];
}
