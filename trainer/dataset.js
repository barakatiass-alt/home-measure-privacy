// Loads the iPhone's "Scan photos for photoreal 3D" zip (COLMAP text + JPEG photos)
// into training views: a camera and the photo as RGB bytes at the training size.
import { unzipSync } from '../vendor/fflate/browser.js';
import { parseCameras, parseImages, parsePoints, trainerCamera } from './colmap.js';
import { chooseSharp, sharpness } from './sharpness.js';

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

/** Width of the grey copy a photo's sharpness is measured on. */
const SCORE_WIDTH = 160;

/** Sharpness of a photo, measured on a small grey copy (see sharpness.js). */
async function photoSharpness(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const width = SCORE_WIDTH;
  const height = Math.max(1, Math.round((bitmap.height / bitmap.width) * width));
  const ctx = new OffscreenCanvas(width, height).getContext('2d');
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const grey = new Uint8Array(width * height);
  for (let p = 0; p < grey.length; p++) grey[p] = (rgba[p * 4] * 77 + rgba[p * 4 + 1] * 150 + rgba[p * 4 + 2] * 29) >> 8;
  return sharpness(grey, width, height);
}


/**
 * Reads the zip. Uses at most `maxViews` photos at `width` pixels across: the sharpest
 * of each stretch of the walk, without the very blurry ones (see chooseSharp).
 * Returns { views, points, photoCount }.
 */
export async function loadDataset(zipBytes, { width = 480, maxViews = 300, onProgress } = {}) {
  const files = unzipSync(new Uint8Array(zipBytes));
  const camText = find(files, 'sparse/0/cameras.txt');
  const imgText = find(files, 'sparse/0/images.txt');
  if (!camText || !imgText) throw new Error('This zip has no COLMAP cameras. Export it from Home Measure: Share, Scan photos for photoreal 3D.');
  const cameras = parseCameras(text(camText));
  const images = parseImages(text(imgText)).filter((image) => find(files, `images/${image.name}`) && cameras.get(image.cameraId));
  const pointsFile = find(files, 'sparse/0/points3D.txt');
  const points = pointsFile ? parsePoints(text(pointsFile)) : { xyz: new Float32Array(0), rgb: new Float32Array(0), count: 0 };
  // Scoring every photo is about a third of the reading time; progress covers both.
  const scores = [];
  for (const [k, image] of images.entries()) {
    scores.push(await photoSharpness(find(files, `images/${image.name}`)));
    onProgress?.(((k + 1) / images.length) * 0.3);
  }
  const chosen = chooseSharp(scores, maxViews).map((i) => images[i]);
  const views = [];
  for (const [k, image] of chosen.entries()) {
    const cam = trainerCamera(image, cameras.get(image.cameraId), width);
    views.push({ cam, image: await decode(find(files, `images/${image.name}`), cam.width, cam.height), name: image.name });
    onProgress?.(0.3 + ((k + 1) / chosen.length) * 0.7);
  }
  if (!views.length) throw new Error('No photos found in the zip.');
  return { views, points, photoCount: images.length };
}
