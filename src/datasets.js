// 2-D toy datasets in the square [-1, 1]². Each point is { x: [x1, x2], y: 0 | 1 }.
import { gaussian, rng } from "./nn.js";

const clamp = (v) => Math.max(-1, Math.min(1, v));

const GENERATORS = {
  circle(n, noise, rand) {
    return Array.from({ length: n }, (_, i) => {
      const inside = i % 2 === 0;
      const r = inside ? rand() * 0.45 : 0.65 + rand() * 0.3;
      const t = rand() * 2 * Math.PI;
      return { x: [clamp(r * Math.cos(t) + gaussian(rand) * noise), clamp(r * Math.sin(t) + gaussian(rand) * noise)], y: inside ? 1 : 0 };
    });
  },
  xor(n, noise, rand) {
    return Array.from({ length: n }, (_, i) => {
      const y = i % 2; // class 1 = same-sign quadrants, class 0 = opposite-sign quadrants
      const pad = 0.08; // keep points off the axes so the classes are separable
      const signA = rand() < 0.5 ? -1 : 1;
      const signB = y ? signA : -signA;
      const a = (pad + rand() * (1 - pad)) * signA;
      const b = (pad + rand() * (1 - pad)) * signB;
      return { x: [clamp(a + gaussian(rand) * noise), clamp(b + gaussian(rand) * noise)], y };
    });
  },
  gauss(n, noise, rand) {
    return Array.from({ length: n }, (_, i) => {
      const y = i % 2;
      const c = y ? 0.45 : -0.45;
      const sd = 0.2 + noise;
      return { x: [clamp(c + gaussian(rand) * sd), clamp(c + gaussian(rand) * sd)], y };
    });
  },
  moons(n, noise, rand) {
    return Array.from({ length: n }, (_, i) => {
      const y = i % 2;
      const t = rand() * Math.PI;
      const x1 = y ? 1 - Math.cos(t) : Math.cos(t);
      const x2 = y ? 0.5 - Math.sin(t) : Math.sin(t);
      return { x: [clamp((x1 - 0.5) * 0.65 + gaussian(rand) * noise), clamp((x2 - 0.25) * 0.8 + gaussian(rand) * noise)], y };
    });
  },
  spiral(n, noise, rand) {
    return Array.from({ length: n }, (_, i) => {
      const y = i % 2;
      const r = (Math.floor(i / 2) / (n / 2)) * 0.95;
      const t = 1.75 * r * 2 * Math.PI + (y ? Math.PI : 0);
      return { x: [clamp(r * Math.sin(t) + gaussian(rand) * noise), clamp(r * Math.cos(t) + gaussian(rand) * noise)], y };
    });
  },
};

export const DATASETS = {
  circle: "Circle",
  xor: "XOR",
  gauss: "Two clusters",
  moons: "Two moons",
  spiral: "Spiral",
};

// Returns { train, test } with a deterministic shuffle and split
export function makeDataset(name, { n = 300, noise = 0.05, seed = 1, testRatio = 0.25 } = {}) {
  const rand = rng(seed);
  const points = GENERATORS[name](n, noise, rand);
  for (let i = points.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [points[i], points[j]] = [points[j], points[i]];
  }
  const cut = Math.round(points.length * (1 - testRatio));
  return { train: points.slice(0, cut), test: points.slice(cut) };
}
