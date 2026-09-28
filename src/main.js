import { Network, rng } from "./nn.js";
import { DATASETS, makeDataset } from "./datasets.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

const CYAN = [0, 173, 255];
const ORANGE = [227, 79, 38];
const BG = [4, 10, 28];
const RES = 72; // decision-boundary sampling resolution
const MAX_EPOCHS = 3000;

const state = {
  net: null,
  train: [],
  test: [],
  epoch: 0,
  history: [],
  playing: false,
  frame: 0,
  dataSeed: 1,
  shuffle: rng(7),
};

for (const [key, name] of Object.entries(DATASETS)) $("dataset").add(new Option(name, key));
$("dataset").value = "spiral";

/* ---------------- Model + data ---------------- */
function parseLayers() {
  const text = $("layers").value.trim();
  const sizes = text === "" ? [] : text.split(/[\s,]+/).map(Number);
  const ok = sizes.length <= 5 && sizes.every((n) => Number.isInteger(n) && n >= 1 && n <= 16);
  $("layers-error").hidden = ok;
  $("layers-error").textContent = ok ? "" : "Use up to 5 layers of 1–16 neurons, e.g. 8, 8";
  return ok ? sizes : null;
}

function buildNetwork() {
  const hidden = parseLayers();
  if (!hidden) return false;
  state.net = new Network([2, ...hidden, 1], { activation: $("activation").value, seed: Math.floor(Math.random() * 1e9) });
  state.epoch = 0;
  state.history = [];
  $("arch").textContent = `· ${[2, ...hidden, 1].join(" → ")} · ${state.net.parameterCount} params`;
  return true;
}

function buildData() {
  const { train, test } = makeDataset($("dataset").value, { n: 300, noise: +$("noise").value, seed: state.dataSeed });
  state.train = train;
  state.test = test;
}

function reset() {
  setPlaying(false);
  if (buildNetwork()) renderAll();
}

/* ---------------- Training loop ---------------- */
function trainEpoch() {
  const opts = { learningRate: +$("lr").value, batchSize: +$("batch").value, l2: +$("l2").value, optimizer: $("optimizer").value, rand: state.shuffle };
  const trainLoss = state.net.trainEpoch(state.train, opts);
  state.epoch++;
  state.history.push([trainLoss, state.net.loss(state.test)]);
  if (state.history.length > 600) state.history.shift();
}

function loop() {
  if (!state.playing) return;
  const steps = state.train.length > 150 ? 1 : 2;
  for (let i = 0; i < steps; i++) trainEpoch();
  state.frame++;
  renderStats();
  if (state.frame % 2 === 0) renderBoundary();
  if (state.frame % 3 === 0) renderLoss();
  if (state.frame % 12 === 0) renderNetwork();
  if (state.epoch >= MAX_EPOCHS) {
    setPlaying(false);
    renderAll();
    return;
  }
  requestAnimationFrame(loop);
}

function setPlaying(on) {
  state.playing = on;
  $("play").textContent = on ? "❚❚ Pause" : "▶ Train";
  if (on) requestAnimationFrame(loop);
}

/* ---------------- Rendering ---------------- */
function stat(label, value, good = false) {
  return el("div", { className: "stat" }, el("span", { className: "stat__label", textContent: label }), el("span", { className: `stat__value${good ? " good" : ""}`, textContent: value }));
}

function renderStats() {
  const { net, train, test } = state;
  const last = state.history.at(-1) ?? [net.loss(train), net.loss(test)];
  const trainAcc = net.accuracy(train);
  const testAcc = net.accuracy(test);
  $("stats").replaceChildren(
    stat("Epoch", state.epoch.toLocaleString()),
    stat("Train loss", last[0].toFixed(3)),
    stat("Test loss", last[1].toFixed(3)),
    stat("Train accuracy", `${(trainAcc * 100).toFixed(1)}%`, trainAcc > 0.95),
    stat("Test accuracy", `${(testAcc * 100).toFixed(1)}%`, testAcc > 0.95)
  );
}

// Offscreen low-res probability map, scaled up smoothly onto the visible canvas
const heat = document.createElement("canvas");
heat.width = RES;
heat.height = RES;
const heatCtx = heat.getContext("2d");
const heatImg = heatCtx.createImageData(RES, RES);

const toCanvas = (v, size) => ((v + 1) / 2) * size;
const fromCanvas = (px, size) => (px / size) * 2 - 1;

function renderBoundary() {
  const canvas = $("boundary");
  const ctx = canvas.getContext("2d");
  const { width: W, height: H } = canvas;

  for (let py = 0; py < RES; py++) {
    for (let px = 0; px < RES; px++) {
      const p = state.net.predict([fromCanvas(px + 0.5, RES), -fromCanvas(py + 0.5, RES)]);
      const conf = Math.abs(p - 0.5) * 2; // 0 at the boundary, 1 when certain
      const color = p >= 0.5 ? CYAN : ORANGE;
      const alpha = 0.08 + conf * 0.32;
      const i = (py * RES + px) * 4;
      for (let c = 0; c < 3; c++) heatImg.data[i + c] = BG[c] * (1 - alpha) + color[c] * alpha;
      heatImg.data[i + 3] = 255;
    }
  }
  heatCtx.putImageData(heatImg, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heat, 0, 0, W, H);

  // Axes
  ctx.strokeStyle = "rgba(169,180,208,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W / 2, 0);
  ctx.lineTo(W / 2, H);
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  const drawPoint = ({ x, y }, hollow) => {
    const cx = toCanvas(x[0], W);
    const cy = toCanvas(-x[1], H);
    ctx.beginPath();
    ctx.arc(cx, cy, 4.2, 0, Math.PI * 2);
    const color = y ? "#00adff" : "#e34f26";
    if (hollow) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  };
  state.train.forEach((pt) => drawPoint(pt, false));
  if ($("show-test").checked) state.test.forEach((pt) => drawPoint(pt, true));
}

function renderLoss() {
  const canvas = $("loss");
  const ctx = canvas.getContext("2d");
  const { width: W, height: H } = canvas;
  ctx.clearRect(0, 0, W, H);
  const h = state.history;
  ctx.fillStyle = "#a9b4d0";
  ctx.font = "11px ui-monospace, Consolas, monospace";
  if (h.length < 2) {
    ctx.fillText("Press ▶ Train to start", 14, H / 2);
    return;
  }
  const max = Math.max(0.1, ...h.flat());
  const pad = 22;
  ctx.fillText(max.toFixed(2), 4, 12);
  ctx.fillText("0", 4, H - 6);
  const line = (idx, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    h.forEach((pt, i) => {
      const x = pad + (i / (h.length - 1)) * (W - pad - 8);
      const y = 8 + (1 - pt[idx] / max) * (H - 20);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  line(1, "#e34f26");
  line(0, "#00adff");
}

const svg = (tag, attrs = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};

function renderNetwork() {
  const { net } = state;
  const sizes = net.sizes;
  const W = 480;
  const H = 230;
  const colX = (l) => 30 + (l / (sizes.length - 1)) * (W - 60);
  const rowY = (i, n) => (n === 1 ? H / 2 : 20 + (i / (n - 1)) * (H - 40));
  const maxW = Math.max(0.5, ...net.W.flat(2).map(Math.abs));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Network weights" });

  net.W.forEach((layer, l) =>
    layer.forEach((row, j) =>
      row.forEach((w, i) => {
        root.append(
          svg("line", {
            x1: colX(l),
            y1: rowY(i, sizes[l]),
            x2: colX(l + 1),
            y2: rowY(j, sizes[l + 1]),
            stroke: w >= 0 ? "#00adff" : "#e34f26",
            "stroke-width": (0.3 + (Math.abs(w) / maxW) * 3.2).toFixed(2),
            "stroke-opacity": (0.25 + (Math.abs(w) / maxW) * 0.7).toFixed(2),
          })
        );
      })
    )
  );
  sizes.forEach((n, l) => {
    for (let i = 0; i < n; i++) {
      root.append(svg("circle", { cx: colX(l), cy: rowY(i, n), r: n > 10 ? 5 : 7, fill: "#0f1a3f", stroke: l === 0 || l === sizes.length - 1 ? "#ffffff" : "#00adff", "stroke-width": 1.5 }));
    }
  });
  const labels = [["x₁", 0, 0], ["x₂", 0, 1]];
  labels.forEach(([text, l, i]) => {
    const t = svg("text", { x: colX(l) - 22, y: rowY(i, 2) + 4 });
    t.textContent = text;
    root.append(t);
  });
  const out = svg("text", { x: colX(sizes.length - 1) + 12, y: H / 2 + 4 });
  out.textContent = "ŷ";
  root.append(out);
  $("network").replaceChildren(root);
}

function renderAll() {
  renderStats();
  renderBoundary();
  renderLoss();
  renderNetwork();
}

/* ---------------- Controls ---------------- */
$("play").addEventListener("click", () => {
  if (state.epoch >= MAX_EPOCHS) reset();
  setPlaying(!state.playing);
});
$("step").addEventListener("click", () => {
  setPlaying(false);
  trainEpoch();
  renderAll();
});
$("reset").addEventListener("click", reset);
$("regen").addEventListener("click", () => {
  state.dataSeed++;
  buildData();
  reset();
});
$("dataset").addEventListener("change", () => {
  buildData();
  reset();
});
$("noise").addEventListener("input", (e) => {
  $("noise-out").textContent = Number(e.target.value).toFixed(2);
  buildData();
  renderAll();
});
$("layers").addEventListener("change", reset);
$("activation").addEventListener("change", reset);
// Sensible starting learning rate for each optimizer
$("optimizer").addEventListener("change", (e) => {
  $("lr").value = { adam: "0.01", momentum: "0.03", sgd: "0.3" }[e.target.value];
});
$("show-test").addEventListener("change", renderBoundary);

// Click to add training points: plain click = class 1 (blue), shift/right click = class 0 (orange)
function addPoint(e, cls) {
  const canvas = $("boundary");
  const rect = canvas.getBoundingClientRect();
  const x = fromCanvas(((e.clientX - rect.left) / rect.width) * canvas.width, canvas.width);
  const y = -fromCanvas(((e.clientY - rect.top) / rect.height) * canvas.height, canvas.height);
  state.train.push({ x: [x, y], y: cls });
  renderBoundary();
  renderStats();
}
$("boundary").addEventListener("click", (e) => addPoint(e, e.shiftKey ? 0 : 1));
$("boundary").addEventListener("contextmenu", (e) => {
  e.preventDefault();
  addPoint(e, 0);
});

buildData();
reset();
