// Starting size of each splat from how close its neighbours are, as 3D Gaussian Splatting
// does: the root mean square distance to the 3 nearest points. Dense LiDAR points (3 cm
// apart) start small and sharp. Points with none near (the 25 cm wall grid of older
// scans) keep the earlier fixed size.

/** Points are looked up in cubes of this size (metres); neighbours further than this may be missed. */
const CELL = 0.1;

/**
 * Per point starting size in metres, clamped to [min, max]. A point with no neighbour
 * within reach gets `fallback`. xyz holds x, y, z per point.
 */
export function neighbourSizes(xyz, count, { min = 0.005, max = 0.1, fallback = 0.06 } = {}) {
  // A number per cube (16 bits an axis: ±3 km at 10 cm), faster than string keys.
  const key = (i, j, k) => ((i + 32768) * 65536 + (j + 32768)) * 65536 + (k + 32768);
  const cells = new Map();
  const cellOf = (p) => [0, 1, 2].map((a) => Math.floor(xyz[p * 3 + a] / CELL));
  for (let p = 0; p < count; p++) {
    const k = key(...cellOf(p));
    const list = cells.get(k);
    if (list) list.push(p);
    else cells.set(k, [p]);
  }
  const sizes = new Float32Array(count);
  const best = [Infinity, Infinity, Infinity];
  for (let p = 0; p < count; p++) {
    best.fill(Infinity);
    const [ci, cj, ck] = cellOf(p);
    const x = xyz[p * 3], y = xyz[p * 3 + 1], z = xyz[p * 3 + 2];
    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        for (let dk = -1; dk <= 1; dk++) {
          const list = cells.get(key(ci + di, cj + dj, ck + dk));
          if (!list) continue;
          for (const q of list) {
            if (q === p) continue;
            const dx = xyz[q * 3] - x, dy = xyz[q * 3 + 1] - y, dz = xyz[q * 3 + 2] - z;
            const d2 = dx * dx + dy * dy + dz * dz;
            // Keep the three smallest, in order.
            if (d2 >= best[2]) continue;
            if (d2 < best[0]) [best[0], best[1], best[2]] = [d2, best[0], best[1]];
            else if (d2 < best[1]) [best[1], best[2]] = [d2, best[1]];
            else best[2] = d2;
          }
        }
      }
    }
    const found = best.filter(Number.isFinite);
    const size = found.length ? Math.sqrt(found.reduce((a, b) => a + b, 0) / found.length) : fallback;
    sizes[p] = Math.min(max, Math.max(min, size));
  }
  return sizes;
}

/** Indices 0..n-1 in a random order (Fisher-Yates). */
export function shuffled(n, random = Math.random) {
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
