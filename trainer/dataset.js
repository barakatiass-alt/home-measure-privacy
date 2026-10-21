// Loads the iPhone's "Scan photos for photoreal 3D" zip (COLMAP text + JPEG photos)
// into training views: a camera and the photo as RGB bytes at the training size.
import { unzipSync } from '../vendor/fflate/browser.js';
import { parseCameras, parseImages, parsePoints, trainerCamera } from './colmap.js';

const text = (bytes) => new TextDecoder().decode(bytes);

/** Finds a file in the zip whatever folder it was zipped under. */
function find(files, suffix) {
  const key = Object.keys(files).find((k) => k.replace(/\\/g, '/').endsWith(suffix));
  return key ? files[key] : null;
}

/** Decodes a photo and scales it to `width` pixels across; returns RGB bytes. */
async function decode(bytes, width, height) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const rgb = new Uint8Array(width * height * 3);
  for (let p = 0; p < width * height; p++) {
    rgb[p * 3] = rgba[p * 4];
    rgb[p * 3 + 1] = rgba[p * 4 + 1];
    rgb[p * 3 + 2] = rgba[p * 4 + 2];
  }
  return rgb;
}

/**
 * Reads the zip. Uses at most `maxViews` photos (evenly spread over the walk) at
 * `width` pixels across. Returns { views, points, photoCount }.
 */
export async function loadDataset(zipBytes, { width = 480, maxViews = 300, onProgress } = {}) {
  const files = unzipSync(new Uint8Array(zipBytes));
  const camText = find(files, 'sparse/0/cameras.txt');
  const imgText = find(files, 'sparse/0/images.txt');
  if (!camText || !imgText) throw new Error('This zip has no COLMAP cameras. Export it from Home Measure: Share, Scan photos for photoreal 3D.');
  const cameras = parseCameras(text(camText));
  const images = parseImages(text(imgText));
  const pointsFile = find(files, 'sparse/0/points3D.txt');
  const points = pointsFile ? parsePoints(text(pointsFile)) : { xyz: new Float32Array(0), rgb: new Float32Array(0), count: 0 };
  const step = Math.max(1, images.length / maxViews);
  const chosen = [];
  for (let i = 0; i < images.length; i += step) chosen.push(images[Math.floor(i)]);
  const views = [];
  for (const [k, image] of chosen.entries()) {
    const photo = find(files, `images/${image.name}`);
    const camera = cameras.get(image.cameraId);
    if (!photo || !camera) continue;
    const cam = trainerCamera(image, camera, width);
    views.push({ cam, image: await decode(photo, cam.width, cam.height), name: image.name });
    onProgress?.((k + 1) / chosen.length);
  }
  if (!views.length) throw new Error('No photos found in the zip.');
  return { views, points, photoCount: images.length };
}
