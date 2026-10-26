// Picks the photos to train on: blurry photos (the phone moving while it took them)
// teach the model blur, so the sharpest photo of each part of the walk is used, and
// photos far blurrier than the rest are left out.

/** A photo below this share of the walk's median sharpness is left out. */
const BLUR_SHARE = 0.35;
/** Never leave out more than this share of the photos (a dim, plain home is soft all over). */
const MAX_DROPPED = 0.25;

/**
 * Sharpness of a grey image: the variance of its Laplacian (edges are crisp when the
 * photo is sharp, smeared when it is blurred). Only comparable between photos of one walk.
 * @param {Uint8Array | Uint8ClampedArray} grey  one byte a pixel
 */
export function sharpness(grey, width, height) {
  let sum = 0;
  let sum2 = 0;
  let n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = grey[i - 1] + grey[i + 1] + grey[i - width] + grey[i + width] - 4 * grey[i];
      sum += lap;
      sum2 += lap * lap;
      n++;
    }
  }
  if (!n) return 0;
  const mean = sum / n;
  return sum2 / n - mean * mean;
}

/**
 * Which photos to train on, as indexes in walk order: the sharpest of each of `maxViews`
 * stretches of the walk (neighbouring photos see the same place), then without the ones
 * far blurrier than the walk's median.
 * @param {number[]} scores  sharpness of each photo, in walk order
 */
export function chooseSharp(scores, maxViews) {
  const step = Math.max(1, scores.length / maxViews);
  const chosen = [];
  for (let start = 0; start < scores.length; start += step) {
    const from = Math.floor(start);
    const to = Math.min(scores.length, Math.floor(start + step));
    let best = from;
    for (let i = from + 1; i < to; i++) if (scores[i] > scores[best]) best = i;
    chosen.push(best);
  }
  const sorted = scores.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor((sorted.length - 1) / 2)] ?? 0;
  const blurry = chosen.filter((i) => scores[i] < median * BLUR_SHARE).sort((a, b) => scores[a] - scores[b]);
  const drop = new Set(blurry.slice(0, Math.floor(chosen.length * MAX_DROPPED)));
  return chosen.filter((i) => !drop.has(i));
}
