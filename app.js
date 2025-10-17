// app.js
// Loads wta_data.csv from project root (auto), with manual file upload fallback.
// Trains MLP, evaluates, draws charts, and provides a simple prediction form.

import { DataLoader } from "./data-loader.js";
import { ModelMLP } from "./gru.js";

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
  csvInput: document.getElementById("csvInput"),
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

async function parseAndInit(text) {
  try {
    loader = new DataLoader();
    dataset = await loader.loadCSVText(text);
    els.info.textContent = `Dataset loaded — Train: ${dataset.X_train.shape[0]}, Test: ${dataset.X_test.shape[0]}, Features: ${dataset.featureNames.length}`;
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
  // Try to fetch from repo root; add cache buster to avoid stale CDN cache
  const url = `./wta_data.csv?v=${Date.now()}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    if (!text || text.trim().length === 0) throw new Error("CSV is empty");
    await parseAndInit(text);
  } catch (err) {
    log(`Auto-load failed: ${err.message}. You can choose a CSV file manually.`);
    els.info.textContent = "Auto-load failed. Please choose wta_data.csv with the file picker.";
  }
}

function buildPredictForm() {
  if (!loader || !dataset) return;
  const form = els.predictForm;
  form.innerHTML = "";

  // Numeric inputs
  loader.numericCols.forEach((c) => {
    const div = document.createElement("div");
    div.className = "form-row";
    div.innerHTML = `<label>${c}</label><input type="number" step="any" name="${c}" required />`;
    form.appendChild(div);
  });

  // Categorical dropdowns with fitted levels
  loader.categoricalCols.forEach((col) => {
    const levels = loader.catLevels[col] || [];
    const div = document.createElement("div");
    div.className = "form-row";
    const opts = levels.map((l) => `<option value="${l}">${l}</option>`).join("");
    div.innerHTML = `<label>${col}</label><select name="${col}">${opts}</select>`;
    form.appendChild(div);
  });
}

async function trainModel() {
  if (!dataset) return alert("Dataset not loaded yet.");
  model = new ModelMLP(dataset.featureNames.length);
  model.build();
  log("Training started...");
  const losses = [], valAcc = [];

  await model.train(dataset.X_train, dataset.y_train, {
    epochs: 20,
    batchSize: 256,
    validationSplit: 0.2,
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
  try {
    const formData = new FormData(els.predictForm);
    const userInput = {};
    for (const [k, v] of formData.entries()) userInput[k] = v;
    const vec = loader.vectorizeForPredict(userInput);
    const x = tf.tensor2d([Array.from(vec)], [1, vec.length], "float32");
    const yProb = model.predictProba(x);
    const prob = (await yProb.data())[0];
    const pred = prob >= 0.5 ? 1 : 0;
    els.predictOut.textContent = `Predicted: ${pred === 1 ? "Player 1 wins" : "Player 1 loses"} (p=${prob.toFixed(3)})`;
    x.dispose(); yProb.dispose();
  } catch (err) {
    alert(`Prediction error: ${err.message}`);
  }
}

// --- Events ---
els.trainBtn.addEventListener("click", trainModel);
els.evalBtn.addEventListener("click", evaluateModel);
els.saveBtn.addEventListener("click", async () => {
  if (model) { await model.save(); log("Model saved to browser storage."); }
});
els.loadModelBtn.addEventListener("click", async () => {
  try {
    const m = new ModelMLP(0);
    await m.load();
    // Recreate inputDim using current dataset (if available)
    if (dataset && dataset.featureNames) {
      // If loaded model input shape mismatches, rebuild not necessary here since tfjs stores it.
    }
    model = m;
    log("Model loaded from browser storage.");
    showPredictPanel(true);
  } catch (e) {
    alert("No saved model found or load failed.");
  }
});

// Manual CSV upload fallback
els.csvInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    await parseAndInit(text);
  } catch (err) {
    alert(`Failed to read CSV: ${err.message}`);
  }
});

// Init
enableTraining(false);
showPredictPanel(false);
autoLoadCSV();
