// app.js — WTA Match Outcome predictor
// Loads TensorFlow.js (global tf), dataset from wta_data.csv, trains MLP model, visualizes metrics.

import { DataLoader } from "./data-loader.js";
import { ModelMLP } from "./gru.js";

const tf = window.tf; // Use global TensorFlow.js loaded via <script>
const LOG_MAX_LINES = 400;

let loader = null;
let model = null;
let dataset = null;
let lossChart = null;
let cmChart = null;

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
  fileInput: document.getElementById("fileInput"),
  loadFileBtn: document.getElementById("loadFileBtn"),
  epochsInput: document.getElementById("epochsInput"),
  batchSizeInput: document.getElementById("batchSizeInput"),
  valSplitInput: document.getElementById("valSplitInput"),
  layer1Input: document.getElementById("layer1Units"),
  layer2Input: document.getElementById("layer2Units"),
  dropoutInput: document.getElementById("dropoutRate"),
  clearLogsBtn: document.getElementById("clearLogsBtn"),
};

function log(msg) {
  const time = new Date().toLocaleTimeString();
  els.logs.textContent += `[${time}] ${msg}\n`;
  const lines = els.logs.textContent.split("\n");
  if (lines.length > LOG_MAX_LINES) {
    const trimmed = lines.slice(-LOG_MAX_LINES).join("\n");
    els.logs.textContent = trimmed.endsWith("\n") ? trimmed : `${trimmed}\n`;
  }
  els.logs.scrollTop = els.logs.scrollHeight;
}

function enableTraining(enabled) {
  els.trainBtn.disabled = !enabled;
  els.evalBtn.disabled = !enabled || !model;
  els.saveBtn.disabled = !enabled || !model;
}

function showPredictPanel(show) {
  els.predictPanel.style.display = show ? "block" : "none";
}

async function parseAndInit(text) {
  try {
    disposeDataset();
    if (model) {
      model.dispose();
      model = null;
    }
    if (lossChart) { lossChart.destroy(); lossChart = null; }
    if (cmChart) { cmChart.destroy(); cmChart = null; }
    loader = new DataLoader();
    dataset = await loader.loadCSVText(text);
    els.info.textContent = `Dataset loaded — Train: ${dataset.X_train.shape[0]}, Test: ${dataset.X_test.shape[0]}, Features: ${dataset.featureNames.length}`;
    log("Dataset loaded successfully.");
    enableTraining(true);
    buildPredictForm();
    els.saveBtn.disabled = true;
    showPredictPanel(false);
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
    els.info.textContent = "Failed to auto-load wta_data.csv from project root. Use manual upload below.";
  }
}

function buildPredictForm() {
  if (!loader || !dataset) return;
  const form = els.predictForm;
  form.innerHTML = "";

  loader.numericCols.forEach((c) => {
    const div = document.createElement("div");
    div.className = "form-row";
    div.innerHTML = `<label>${c}</label><input type="number" step="any" name="${c}" required placeholder="Enter ${c}" />`;
    form.appendChild(div);
  });

  loader.categoricalCols.forEach((col) => {
    const levels = loader.catLevels[col] || [];
    const div = document.createElement("div");
    div.className = "form-row";
    const opts = ["<option value=\"\">Select…</option>", ...levels.map((l) => `<option value="${l}">${l}</option>`)];
    div.innerHTML = `<label>${col}</label><select name="${col}" required>${opts.join("")}</select>`;
    form.appendChild(div);
  });
}

async function trainModel() {
  if (!dataset) return alert("Dataset not loaded yet.");
  if (model) model.dispose();
  const hyper = readHyperparameters();
  model = new ModelMLP(dataset.featureNames.length, hyper.architecture);
  model.build();
  log("Training started...");
  const losses = [], valAcc = [];
  enableTraining(false);
  try {
    await model.train(dataset.X_train, dataset.y_train, {
      epochs: hyper.training.epochs,
      batchSize: hyper.training.batchSize,
      validationSplit: hyper.training.validationSplit,
      onEpochEnd: (epoch, logs) => {
        const val = logs.val_acc ?? logs.val_accuracy ?? 0;
        log(`Epoch ${epoch + 1}: loss=${Number(logs.loss).toFixed(4)} val_acc=${Number(val).toFixed(4)}`);
        losses.push(Number(logs.loss));
        valAcc.push(Number(val));
        drawLossChart(losses, valAcc);
      }
    });

    log("Training complete.");
    els.saveBtn.disabled = false;
    els.evalBtn.disabled = false;
    showPredictPanel(true);
  } catch (err) {
    log(`Training failed: ${err.message}`);
    alert(err.message);
  } finally {
    enableTraining(true);
  }
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
  if (!els.predictForm.reportValidity()) return;
  const formData = new FormData(els.predictForm);
  const userInput = {};
  for (const [k, v] of formData.entries()) userInput[k] = v;
  try {
    const vec = loader.vectorizeForPredict(userInput);
    const x = tf.tensor2d([Array.from(vec)], [1, vec.length], "float32");
    const yProb = model.predictProba(x);
    const prob = (await yProb.data())[0];
    const pred = prob >= 0.5 ? 1 : 0;
    els.predictOut.textContent = `Predicted: ${pred === 1 ? "Player 1 wins" : "Player 1 loses"} (p=${prob.toFixed(3)})`;
    x.dispose(); yProb.dispose();
  } catch (err) {
    log(`Prediction failed: ${err.message}`);
    alert(err.message);
  }
}

// Buttons
els.trainBtn.addEventListener("click", trainModel);
els.evalBtn.addEventListener("click", evaluateModel);
els.saveBtn.addEventListener("click", async () => {
  if (model) { await model.save(); log("Model saved to browser storage."); }
});
els.loadModelBtn.addEventListener("click", async () => {
  try {
    const m = new ModelMLP(dataset ? dataset.featureNames.length : 0);
    await m.load();
    model = m;
    log("Model loaded from browser storage.");
    showPredictPanel(true);
    enableTraining(Boolean(dataset));
    if (!dataset || !loader) {
      log("Load a dataset to enable predictions with the restored model.");
    }
  } catch {
    alert("No saved model found or load failed.");
  }
});
els.predictBtn.addEventListener("click", handlePredict);
els.loadFileBtn.addEventListener("click", handleManualFileLoad);
els.clearLogsBtn.addEventListener("click", () => {
  els.logs.textContent = "";
});

// Init
console.log("🚀 App initialized — calling autoLoadCSV()");
enableTraining(false);
showPredictPanel(false);
autoLoadCSV();
console.log("✅ autoLoadCSV() call placed after init");

function readHyperparameters() {
  const epochs = clampInt(els.epochsInput.value, 1, 200, 20);
  const batchSize = clampInt(els.batchSizeInput.value, 8, 1024, 256);
  const valSplitRaw = Number.parseFloat(els.valSplitInput.value);
  const validationSplit = Number.isFinite(valSplitRaw) ? Math.min(Math.max(valSplitRaw, 0.05), 0.4) : 0.2;
  const layer1 = clampInt(els.layer1Input.value, 4, 512, 128);
  const layer2 = clampInt(els.layer2Input.value, 0, 512, 64);
  const dropout = clampFloat(els.dropoutInput.value, 0, 0.8, 0.3);
  const hiddenUnits = layer2 > 0 ? [layer1, layer2] : [layer1];
  return {
    training: { epochs, batchSize, validationSplit },
    architecture: { hiddenUnits, dropout }
  };
}

function clampInt(value, min, max, fallback) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

function clampFloat(value, min, max, fallback) {
  const n = Number.parseFloat(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

async function handleManualFileLoad() {
  if (!els.fileInput.files || els.fileInput.files.length === 0) {
    return alert("Select a CSV file first.");
  }
  const file = els.fileInput.files[0];
  try {
    const text = await file.text();
    log(`Manual CSV load: ${file.name} (${text.length} chars)`);
    await parseAndInit(text);
  } catch (err) {
    log(`Manual load failed: ${err.message}`);
    alert(err.message);
  }
}

function disposeDataset() {
  if (!dataset) return;
  try {
    dataset.X_train?.dispose();
    dataset.y_train?.dispose();
    dataset.X_test?.dispose();
    dataset.y_test?.dispose();
  } catch (err) {
    console.warn("Failed to dispose dataset tensors", err);
  }
  dataset = null;
}
