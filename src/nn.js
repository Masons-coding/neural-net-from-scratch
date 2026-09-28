// A multilayer perceptron written from scratch: forward pass, backpropagation and
// mini-batch gradient descent, with no ML libraries. Binary classifier with a sigmoid output
// trained on binary cross-entropy loss.

// Seedable PRNG (mulberry32) so training runs are reproducible
export function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal sample via Box–Muller
export function gaussian(rand) {
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Derivatives are written in terms of the activation's OUTPUT a = f(z), which is what
// backprop has on hand.
export const ACTIVATIONS = {
  tanh: { f: Math.tanh, df: (a) => 1 - a * a },
  relu: { f: (z) => (z > 0 ? z : 0), df: (a) => (a > 0 ? 1 : 0) },
  sigmoid: { f: sigmoid, df: (a) => a * (1 - a) },
  linear: { f: (z) => z, df: () => 1 },
};

const EPS = 1e-12;
export const bce = (p, y) => -(y * Math.log(p + EPS) + (1 - y) * Math.log(1 - p + EPS));

export class Network {
  // sizes: e.g. [2, 8, 8, 1] = 2 inputs, two hidden layers of 8, one output
  constructor(sizes, { activation = "tanh", seed = 1 } = {}) {
    if (sizes.at(-1) !== 1) throw new Error("Output layer must have exactly 1 neuron");
    this.sizes = sizes;
    this.activation = activation;
    const rand = rng(seed);
    this.W = [];
    this.b = [];
    for (let l = 1; l < sizes.length; l++) {
      const fanIn = sizes[l - 1];
      // He initialisation for ReLU layers, Xavier/Glorot for everything else
      const scale = activation === "relu" && l < sizes.length - 1 ? Math.sqrt(2 / fanIn) : Math.sqrt(1 / fanIn);
      this.W.push(Array.from({ length: sizes[l] }, () => Array.from({ length: fanIn }, () => gaussian(rand) * scale)));
      this.b.push(new Array(sizes[l]).fill(activation === "relu" ? 0.01 : 0));
    }
  }

  get parameterCount() {
    return this.W.reduce((n, layer, l) => n + layer.length * layer[0].length + this.b[l].length, 0);
  }

  // Returns the activations of every layer, input first
  forward(x) {
    const acts = [x];
    for (let l = 0; l < this.W.length; l++) {
      const act = l === this.W.length - 1 ? ACTIVATIONS.sigmoid : ACTIVATIONS[this.activation];
      const prev = acts[l];
      const bias = this.b[l];
      acts.push(
        this.W[l].map((row, j) => {
          let z = bias[j];
          for (let i = 0; i < row.length; i++) z += row[i] * prev[i];
          return act.f(z);
        })
      );
    }
    return acts;
  }

  predict(x) {
    return this.forward(x).at(-1)[0];
  }

  // Backpropagation for one example. With a sigmoid output and cross-entropy loss the
  // output error simplifies to dL/dz = prediction − target.
  gradients(x, y) {
    const acts = this.forward(x);
    const L = this.W.length;
    const gW = new Array(L);
    const gb = new Array(L);
    let delta = [acts[L][0] - y];
    for (let l = L - 1; l >= 0; l--) {
      gW[l] = delta.map((d) => acts[l].map((a) => d * a));
      gb[l] = [...delta];
      if (l > 0) {
        const { df } = ACTIVATIONS[this.activation];
        const W = this.W[l];
        // Chain rule: push the error back through the weights, then through the activation
        delta = acts[l].map((a, i) => {
          let sum = 0;
          for (let j = 0; j < delta.length; j++) sum += delta[j] * W[j][i];
          return df(a) * sum;
        });
      }
    }
    return { gW, gb, loss: bce(acts[L][0], y) };
  }

  // Average gradients over a mini-batch (plus the L2 penalty's gradient on weights)
  batchGradients(batch, l2 = 0) {
    const gW = this.W.map((layer) => layer.map((row) => row.map(() => 0)));
    const gb = this.b.map((layer) => layer.map(() => 0));
    let loss = 0;
    for (const { x, y } of batch) {
      const g = this.gradients(x, y);
      loss += g.loss;
      for (let l = 0; l < gW.length; l++) {
        for (let j = 0; j < gW[l].length; j++) {
          gb[l][j] += g.gb[l][j] / batch.length;
          for (let i = 0; i < gW[l][j].length; i++) gW[l][j][i] += g.gW[l][j][i] / batch.length;
        }
      }
    }
    if (l2) gW.forEach((layer, l) => layer.forEach((row, j) => row.forEach((_, i) => (row[i] += l2 * this.W[l][j][i]))));
    return { gW, gb, loss: loss / batch.length };
  }

  // One optimisation step on a mini-batch.
  //   sgd      : θ ← θ − η·g
  //   momentum : v ← βv + g;  θ ← θ − η·v                 (smooths noisy gradients)
  //   adam     : per-parameter step sizes from bias-corrected running means of g and g²
  trainBatch(batch, learningRate, l2 = 0, optimizer = "sgd") {
    const { gW, gb, loss } = this.batchGradients(batch, l2);
    const params = [...this.W.flatMap((layer) => layer), ...this.b]; // arrays we update in place
    const grads = [...gW.flatMap((layer) => layer), ...gb];

    if (optimizer === "sgd") {
      params.forEach((p, k) => p.forEach((_, i) => (p[i] -= learningRate * grads[k][i])));
      return loss;
    }

    if (!this.opt || this.opt.type !== optimizer) {
      this.opt = { type: optimizer, t: 0, m: grads.map((g) => g.map(() => 0)), v: grads.map((g) => g.map(() => 0)) };
    }
    const s = this.opt;
    s.t++;
    if (optimizer === "momentum") {
      const beta = 0.9;
      params.forEach((p, k) =>
        p.forEach((_, i) => {
          s.m[k][i] = beta * s.m[k][i] + grads[k][i];
          p[i] -= learningRate * s.m[k][i];
        })
      );
    } else if (optimizer === "adam") {
      const [b1, b2, eps] = [0.9, 0.999, 1e-8];
      const c1 = 1 - b1 ** s.t;
      const c2 = 1 - b2 ** s.t;
      params.forEach((p, k) =>
        p.forEach((_, i) => {
          const g = grads[k][i];
          s.m[k][i] = b1 * s.m[k][i] + (1 - b1) * g;
          s.v[k][i] = b2 * s.v[k][i] + (1 - b2) * g * g;
          p[i] -= (learningRate * (s.m[k][i] / c1)) / (Math.sqrt(s.v[k][i] / c2) + eps);
        })
      );
    } else {
      throw new Error(`Unknown optimizer: ${optimizer}`);
    }
    return loss;
  }

  // One pass over the data in shuffled mini-batches
  trainEpoch(data, { learningRate = 0.1, batchSize = 10, l2 = 0, optimizer = "sgd", rand = Math.random } = {}) {
    const order = [...data];
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let s = 0; s < order.length; s += batchSize) this.trainBatch(order.slice(s, s + batchSize), learningRate, l2, optimizer);
    return this.loss(data);
  }

  loss(data) {
    return data.length ? data.reduce((s, { x, y }) => s + bce(this.predict(x), y), 0) / data.length : 0;
  }

  accuracy(data) {
    return data.length ? data.filter(({ x, y }) => (this.predict(x) >= 0.5 ? 1 : 0) === y).length / data.length : 0;
  }
}
