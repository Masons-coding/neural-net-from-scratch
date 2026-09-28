import { test } from "node:test";
import assert from "node:assert/strict";

import { ACTIVATIONS, Network, bce, rng } from "../src/nn.js";
import { DATASETS, makeDataset } from "../src/datasets.js";

// Numerically estimate dLoss/dparam with central differences and compare to backprop
function gradientCheck(activation) {
  const net = new Network([2, 4, 3, 1], { activation, seed: 3 });
  const x = [0.3, -0.7];
  const y = 1;
  const { gW, gb } = net.gradients(x, y);
  const h = 1e-5;
  let worst = 0;
  const check = (get, set, analytic) => {
    const orig = get();
    set(orig + h);
    const plus = bce(net.predict(x), y);
    set(orig - h);
    const minus = bce(net.predict(x), y);
    set(orig);
    const numeric = (plus - minus) / (2 * h);
    worst = Math.max(worst, Math.abs(numeric - analytic) / Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic)));
  };
  net.W.forEach((layer, l) =>
    layer.forEach((row, j) =>
      row.forEach((_, i) => check(() => net.W[l][j][i], (v) => (net.W[l][j][i] = v), gW[l][j][i]))
    )
  );
  net.b.forEach((layer, l) => layer.forEach((_, j) => check(() => net.b[l][j], (v) => (net.b[l][j] = v), gb[l][j])));
  return worst;
}

for (const activation of ["tanh", "sigmoid", "relu"]) {
  test(`backprop gradients match numerical gradients (${activation})`, () => {
    assert.ok(gradientCheck(activation) < 1e-4);
  });
}

test("activation derivatives are consistent with the functions", () => {
  for (const [name, { f, df }] of Object.entries(ACTIVATIONS)) {
    for (const z of [-1.3, -0.2, 0.4, 2.1]) {
      const numeric = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6;
      assert.ok(Math.abs(numeric - df(f(z))) < 1e-4, `${name} at ${z}`);
    }
  }
});

test("a single training step on a batch lowers its loss", () => {
  const net = new Network([2, 5, 1], { seed: 1 });
  const batch = [
    { x: [0.5, 0.5], y: 1 },
    { x: [-0.5, -0.5], y: 0 },
  ];
  const before = net.loss(batch);
  net.trainBatch(batch, 0.5);
  assert.ok(net.loss(batch) < before);
});

test("a network with no hidden layer (logistic regression) separates two clusters", () => {
  const { train, test: holdout } = makeDataset("gauss", { seed: 4 });
  const net = new Network([2, 1], { seed: 2 });
  const rand = rng(9);
  for (let e = 0; e < 60; e++) net.trainEpoch(train, { learningRate: 0.5, rand });
  assert.ok(net.accuracy(holdout) > 0.95);
});

test("a hidden layer is required for XOR, and learns it", () => {
  const { train, test: holdout } = makeDataset("xor", { seed: 5, noise: 0 });
  const rand = rng(1);

  const linear = new Network([2, 1], { seed: 2 });
  for (let e = 0; e < 100; e++) linear.trainEpoch(train, { learningRate: 0.3, rand });
  assert.ok(linear.accuracy(holdout) < 0.75, "a linear model cannot solve XOR");

  const deep = new Network([2, 8, 8, 1], { activation: "tanh", seed: 2 });
  for (let e = 0; e < 150; e++) deep.trainEpoch(train, { learningRate: 0.3, rand });
  assert.ok(deep.accuracy(holdout) > 0.95, `accuracy ${deep.accuracy(holdout)}`);
});

test("learns the circle dataset with ReLU", () => {
  const { train, test: holdout } = makeDataset("circle", { seed: 6 });
  const net = new Network([2, 8, 8, 1], { activation: "relu", seed: 4 });
  const rand = rng(2);
  for (let e = 0; e < 150; e++) net.trainEpoch(train, { learningRate: 0.1, rand });
  assert.ok(net.accuracy(holdout) > 0.93, `accuracy ${net.accuracy(holdout)}`);
});

test("Adam learns the spiral (a notoriously hard 2-D problem)", () => {
  const { train, test: holdout } = makeDataset("spiral", { seed: 1 });
  const net = new Network([2, 16, 16, 1], { activation: "relu", seed: 7 });
  const rand = rng(1);
  for (let e = 0; e < 250; e++) net.trainEpoch(train, { learningRate: 0.01, optimizer: "adam", rand });
  assert.ok(net.accuracy(holdout) > 0.9, `accuracy ${net.accuracy(holdout)}`);
});

test("momentum and Adam both reduce loss, and unknown optimizers are rejected", () => {
  const { train } = makeDataset("moons", { seed: 2 });
  for (const [optimizer, lr] of [["momentum", 0.03], ["adam", 0.01]]) {
    const net = new Network([2, 6, 1], { seed: 1 });
    const before = net.loss(train);
    const rand = rng(4);
    for (let e = 0; e < 20; e++) net.trainEpoch(train, { learningRate: lr, optimizer, rand });
    assert.ok(net.loss(train) < before * 0.8, optimizer);
  }
  assert.throws(() => new Network([2, 1]).trainBatch(train.slice(0, 2), 0.1, 0, "rmsprop"));
});

test("training is reproducible with a fixed seed", () => {
  const run = () => {
    const { train } = makeDataset("moons", { seed: 3 });
    const net = new Network([2, 6, 1], { seed: 3 });
    const rand = rng(3);
    for (let e = 0; e < 5; e++) net.trainEpoch(train, { rand });
    return net.loss(train);
  };
  assert.equal(run(), run());
});

test("datasets are balanced, in range, and split train/test 75/25", () => {
  for (const name of Object.keys(DATASETS)) {
    const { train, test: holdout } = makeDataset(name, { n: 200 });
    assert.equal(train.length, 150);
    assert.equal(holdout.length, 50);
    const all = [...train, ...holdout];
    assert.equal(all.filter((p) => p.y === 1).length, 100, name);
    assert.ok(all.every((p) => p.x.every((v) => v >= -1 && v <= 1)), name);
  }
});

test("parameter count and shape validation", () => {
  assert.equal(new Network([2, 4, 1]).parameterCount, 2 * 4 + 4 + 4 * 1 + 1);
  assert.throws(() => new Network([2, 3]));
});
