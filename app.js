// app.js — WTA Match Outcome predictor
// Loads TensorFlow.js (global tf), dataset from wta_data.csv, trains GRU model, visualizes metrics.

import { DataLoader } from "./data-loader.js";
import { ModelMLP } from "./gru.js";

const tf = window.tf; // Use global TensorFlow.js loaded via <script>

let loader = null;
let model = null;
let dataset = null;
let lossChart = null;
let cmChart = null;
let metadata = null;

const els = {
  trainBtn: document.getElementById("trainBtn"),
  evalBtn: document.getElementById("evalBtn"),
  saveBtn: document.getElementById("saveBtn"),
  loadModelBtn: document.getElementById("loadModelBtn"),
  logs: document.getElementById("logs"),
  info: document.getElementById("info"),
  lossCanvas: document.getElementById("lossChart"),
  cmCanvas: document.getElementById("cmChart"),
  predictPanel: document.getElementById("predictPanel"),
  predictForm: document.getElementById("predictForm"),
  predictBtn: document.getElementById("predictBtn"),
  predictOut: document.getElementById("predictOut"),
};

function log(msg) {
  const time = new Date().toLocaleTimeString();
  els.logs.textContent += `[${time}] ${msg}\n`;
  els.logs.scrollTop = els.logs.scrollHeight;
}

function enableTraining(enabled) {
  els.trainBtn.disabled = !enabled;
  els.evalBtn.disabled = !enabled;
}

function showPredictPanel(show) {
  els.predictPanel.style.display = show ? "block" : "none";
}

async function initTensorFlowBackend() {
  const backendPriority = ["webgl", "wasm", "cpu"];
  const wasmBase = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.20.0/dist/";

  if (tf.wasm?.setWasmPaths) {
    tf.wasm.setWasmPaths(wasmBase);
  }

  for (const backend of backendPriority) {
    try {
      const ok = await tf.setBackend(backend);
      if (!ok) continue;
      await tf.ready();
      if (tf.getBackend() === backend) return backend;
    } catch (err) {
      console.warn(`Failed to set TensorFlow backend to ${backend}:`, err);
    }
  }

  await tf.ready();
  return tf.getBackend();
}

async function parseAndInit(text) {
  try {
    loader = new DataLoader();
    dataset = await loader.loadCSVText(text);
    metadata = dataset.metadata || {};
    const playerCount = metadata.players ? metadata.players.length : loader.players.length;
    els.info.textContent = `Dataset loaded — Train: ${dataset.X_train.shape[0]}, Test: ${dataset.X_test.shape[0]}, Features: ${dataset.featureNames.length}, Players indexed: ${playerCount}`;
    log("Dataset loaded successfully.");
    enableTraining(true);
    buildPredictForm();
  } catch (err) {
    console.error(err);
    els.info.textContent = `Dataset error: ${err.message}`;
    log(`Dataset error: ${err.message}`);
    enableTraining(false);
  }
}

async function autoLoadCSV() {
  const url = `./wta_data.csv?v=${Date.now()}`;
  try {
    console.log("🔍 Fetching CSV from:", url);
    const res = await fetch(url, { cache: "no-store" });
    console.log("✅ HTTP status:", res.status);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    console.log("📄 CSV length:", text.length);
    console.log("📄 First 200 chars:", text.slice(0, 200));

    if (!text || text.trim().length === 0) throw new Error("CSV is empty");
    await parseAndInit(text);
  } catch (err) {
    console.error("❌ Auto-load failed:", err);
    log(`Auto-load failed: ${err.message}`);
    els.info.textContent = "Failed to auto-load wta_data.csv from root.";
  }
}


function buildPredictForm() {
  if (!loader || !dataset) return;
  const form = els.predictForm;
  form.innerHTML = "";

  const players = loader.players || (metadata && metadata.players ? metadata.players : []);
  if (!players.length) {
    const warn = document.createElement("div");
    warn.textContent = "Player directory could not be built from dataset.";
    form.appendChild(warn);
    return;
  }

  const datalist = document.createElement("datalist");
  datalist.id = "playerOptions";
  datalist.innerHTML = players.map((p) => `<option value="${p}"></option>`).join("");
  form.appendChild(datalist);

  const playerRow = (label, name) => {
    const div = document.createElement("div");
    div.className = "form-row";
    div.innerHTML = `<label>${label}</label><input type="search" name="${name}" list="playerOptions" placeholder="Start typing a player" autocomplete="off" required />`;
    return div;
  };

  form.appendChild(playerRow("Player 1", "player1"));
  form.appendChild(playerRow("Player 2", "player2"));

  if (loader.categoricalCols.includes("Surface")) {
    const levels = loader.catLevels.Surface || (metadata && metadata.surfaces ? metadata.surfaces : []);
    const div = document.createElement("div");
    div.className = "form-row";
    const opts = levels.map((l) => `<option value="${l}">${l}</option>`).join("");
    div.innerHTML = `<label>Surface</label><select name="Surface">${opts}</select>`;
    form.appendChild(div);
  }

  if (loader.categoricalCols.includes("Court")) {
    const levels = loader.catLevels.Court || (metadata && metadata.courts ? metadata.courts : []);
    if (levels.length) {
      const div = document.createElement("div");
      div.className = "form-row";
      const opts = levels.map((l) => `<option value="${l}">${l}</option>`).join("");
      div.innerHTML = `<label>Court</label><select name="Court">${opts}</select>`;
      form.appendChild(div);
    }
  }

  if (loader.categoricalCols.includes("Round")) {
    const levels = loader.catLevels.Round || (metadata && metadata.rounds ? metadata.rounds : []);
    if (levels.length) {
      const div = document.createElement("div");
      div.className = "form-row";
      const opts = levels.map((l) => `<option value="${l}">${l}</option>`).join("");
      div.innerHTML = `<label>Round</label><select name="Round">${opts}</select>`;
      form.appendChild(div);
    }
  }

  const hint = document.createElement("div");
  hint.className = "form-hint";
  hint.textContent = "Players are matched case-insensitively; choose surface/context and hit predict.";
  form.appendChild(hint);
}

async function trainModel() {
  if (!dataset) return alert("Dataset not loaded yet.");
  model = new ModelMLP(dataset.featureNames.length);
  model.build();
  log("Training started...");
  const losses = [], valAcc = [];

  const history = await model.train(dataset.X_train, dataset.y_train, {
    epochs: 16,
    batchSize: 256,
    validationSplit: 0.2,
    patience: 3,
    onEpochEnd: (epoch, logs) => {
      const val = logs.val_acc ?? logs.val_accuracy ?? 0;
      log(`Epoch ${epoch + 1}: loss=${Number(logs.loss).toFixed(4)} val_acc=${Number(val).toFixed(4)}`);
      losses.push(Number(logs.loss));
      valAcc.push(Number(val));
      drawLossChart(losses, valAcc);
    }
  });

  const epochsRan = history?.epoch?.length ?? losses.length;
  log(`Training complete after ${epochsRan} epoch(s).`);
  els.saveBtn.disabled = false;
  els.evalBtn.disabled = false;
  showPredictPanel(true);
}

async function evaluateModel() {
  if (!dataset || !model) return alert("Train the model first.");
  log("Evaluating on test set...");
  const { loss, acc } = await model.evaluate(dataset.X_test, dataset.y_test);
  log(`Test Loss=${loss.toFixed(4)} | Accuracy=${acc.toFixed(4)}`);
  const cm = await model.confusionMatrix(dataset.X_test, dataset.y_test);
  drawConfusionMatrix(cm);
}

function drawLossChart(losses, valAcc) {
  const ctx = els.lossCanvas.getContext("2d");
  if (lossChart) lossChart.destroy();
  lossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: losses.map((_, i) => `E${i + 1}`),
      datasets: [
        { label: "Loss", data: losses, borderColor: "#6aa8ff", tension: 0.2 },
        { label: "Val Accuracy", data: valAcc, borderColor: "#50fa7b", tension: 0.2 }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}

function drawConfusionMatrix({ tp, tn, fp, fn }) {
  const ctx = els.cmCanvas.getContext("2d");
  if (cmChart) cmChart.destroy();
  cmChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Actual 0", "Actual 1"],
      datasets: [
        { label: "Pred 0", data: [tn, fn], backgroundColor: "#6aa8ff" },
        { label: "Pred 1", data: [fp, tp], backgroundColor: "#ff6384" }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: true } }
    }
  });
}

async function handlePredict(e) {
  e.preventDefault();
  if (!model || !loader) return alert("Train or load a model first.");
  const formData = new FormData(els.predictForm);
  const player1 = (formData.get("player1") || "").toString();
  const player2 = (formData.get("player2") || "").toString();
  const surface = formData.get("Surface") || null;
  const court = formData.get("Court") || null;
  const round = formData.get("Round") || null;
  let prepared;
  try {
    prepared = loader.vectorizeFromPlayers({ player1, player2, surface, court, round });
  } catch (err) {
    alert(err.message);
    return;
  }

  const vec = prepared.vector;
  const x = tf.tensor2d([Array.from(vec)], [1, vec.length], "float32");
  const yProb = model.predictProba(x);
  const prob = (await yProb.data())[0];
  const pred = prob >= 0.5 ? 1 : 0;
  const p1Name = prepared.players?.player1 ?? player1;
  const p2Name = prepared.players?.player2 ?? player2;
  const winnerText = pred === 1 ? `${p1Name} wins` : `${p1Name} loses`;
  let message = `Prediction for ${p1Name} vs ${p2Name}: ${winnerText} (p=${prob.toFixed(3)})`;
  if (prepared.missingFeatures?.length) {
    log(`Prediction fallback — missing engineered features: ${prepared.missingFeatures.join(", ")}`);
    message += `\nNote: used default values for ${prepared.missingFeatures.join(", ")}.`;
  }
  els.predictOut.textContent = message;
  x.dispose(); yProb.dispose();
}

// Buttons
els.trainBtn.addEventListener("click", trainModel);
els.evalBtn.addEventListener("click", evaluateModel);
els.saveBtn.addEventListener("click", async () => {
  if (model) { await model.save(); log("Model saved to browser storage."); }
});
els.loadModelBtn.addEventListener("click", async () => {
  try {
    const m = new ModelMLP(0);
    await m.load();
    model = m;
    log("Model loaded from browser storage.");
    showPredictPanel(true);
  } catch {
    alert("No saved model found or load failed.");
  }
});
els.predictBtn.addEventListener("click", handlePredict);

async function bootstrap() {
  console.log("🚀 App initialized — preparing TensorFlow backend");
  enableTraining(false);
  showPredictPanel(false);
  const backend = await initTensorFlowBackend();
  log(`TensorFlow backend in use: ${backend}`);
  console.log(`✅ TensorFlow backend set to: ${backend}`);
  await autoLoadCSV();
}

bootstrap();
