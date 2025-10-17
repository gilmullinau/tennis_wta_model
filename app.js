// app.js
// Binds UI, uses DataLoader + ModelMLP, renders charts, enables predict-after-train.

import { DataLoader } from "./data-loader.js";
import { ModelMLP } from "./gru.js";

let loader = null;
let model = null;
let dataset = null;

// Charts
let lossChart = null;
let cmChart = null;

const els = {
  fileInput: document.getElementById("fileInput"),
  loadBtn: document.getElementById("loadBtn"),
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
  predictOut: document.getElementById("predictOut")
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

async function loadCSV() {
  try {
    const file = els.fileInput.files?.[0];
    if (!file) {
      alert("Please choose a CSV file first.");
      return;
    }
    log(`Loading CSV: ${file.name}`);
    loader = new DataLoader();
    dataset = await loader.loadCSV(file);
    log(`Loaded dataset. Train: ${dataset.X_train.shape[0]} rows, Test: ${dataset.X_test.shape[0]} rows.`);
    els.info.textContent = `Features: ${dataset.featureNames.length} | Train: ${dataset.X_train.shape[0]} | Test: ${dataset.X_test.shape[0]}`;
    enableTraining(true);
    showPredictPanel(false);
  } catch (err) {
    console.error(err);
    alert("Failed to load CSV: " + err.message);
    log(`Error: ${err.message}`);
  }
}

async function trainModel() {
  if (!dataset) {
    alert("Load a dataset first.");
    return;
  }
  if (model) {
    model.model.dispose();
    model = null;
  }
  model = new ModelMLP(dataset.featureNames.length);
  model.build();
  log("Training started...");
  const losses = [];
  const valAcc = [];

  await model.train(dataset.X_train, dataset.y_train, {
    epochs: 20,
    batchSize: 256,
    validationSplit: 0.2,
    onEpochEnd: (epoch, logs) => {
      log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} val_acc=${(logs.val_acc ?? logs.val_accuracy ?? 0).toFixed(4)}`);
      losses.push(logs.loss);
      valAcc.push(logs.val_acc ?? logs.val_accuracy ?? 0);
      drawLossChart(losses, valAcc);
    }
  });
  log("Training completed.");
  els.saveBtn.disabled = false;
  els.evalBtn.disabled = false;
  showPredictPanel(true);
}

async function evaluateModel() {
  if (!dataset || !model) {
    alert("Load data and train the model first.");
    return;
  }
  log("Evaluating on test set...");
  const { loss, acc } = await model.evaluate(dataset.X_test, dataset.y_test);
  log(`Test Loss=${loss.toFixed(4)} | Test Accuracy=${acc.toFixed(4)}`);
  const cm = await model.confusionMatrix(dataset.X_test, dataset.y_test, 0.5);
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
        { label: "Loss", data: losses, tension: 0.2 },
        { label: "Val Accuracy", data: valAcc, tension: 0.2 }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}

function drawConfusionMatrix({ tp, tn, fp, fn }) {
  const ctx = els.cmCanvas.getContext("2d");
  const data = [
    [tn, fp],
    [fn, tp]
  ];
  const labels = ["Pred 0", "Pred 1"];
  const bk = (v) => v;

  // Convert to flat dataset for Chart.js matrix-like via stacked bars or bubble workaround
  // Simpler: show as 2 bar groups (Actual 0 and Actual 1) with two stacks.
  if (cmChart) cmChart.destroy();
  cmChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Actual 0", "Actual 1"],
      datasets: [
        { label: "Pred 0", data: [data[0][0], data[1][0]] },
        { label: "Pred 1", data: [data[0][1], data[1][1]] }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: { enabled: true },
        legend: { position: "bottom" }
      },
      scales: { y: { beginAtZero: true, precision: 0 } }
    }
  });
  log(`Confusion Matrix: TN=${tn}, FP=${fp}, FN=${fn}, TP=${tp}`);
}

// Save/load model to localStorage
async function saveModel() {
  if (!model) return;
  await model.save();
  log("Model saved to browser storage.");
}
async function loadSavedModel() {
  try {
    const m = new ModelMLP(0);
    await m.load();
    model = m;
    log("Model loaded from browser storage.");
    showPredictPanel(true);
  } catch (e) {
    alert("No saved model found.");
  }
}

// Predict panel
function buildPredictForm() {
  // Build dynamic form only once, based on loader artifacts
  if (!loader || !dataset) return;
  const form = els.predictForm;
  form.innerHTML = "";

  // Numeric inputs
  const numeric = loader.numericCols;
  numeric.forEach((c) => {
    const div = document.createElement("div");
    div.className = "form-row";
    div.innerHTML = `
      <label>${c}</label>
      <input type="number" step="any" name="${c}" required />
    `;
    form.appendChild(div);
  });

  // Categorical selects
  loader.categoricalCols.forEach((col) => {
    const levels = loader.catLevels[col] || [];
    const div = document.createElement("div");
    div.className = "form-row";
    const opts = levels.map((l) => `<option value="${l}">${l}</option>`).join("");
    div.innerHTML = `
      <label>${col}</label>
      <select name="${col}">${opts}</select>
    `;
    form.appendChild(div);
  });
}

async function handlePredict() {
  if (!model || !loader) {
    alert("Train or load a model first.");
    return;
  }
  const formData = new FormData(els.predictForm);
  const userInput = {};
  for (const [k, v] of formData.entries()) {
    userInput[k] = v;
  }
  try {
    const vec = loader.vectorizeForPredict(userInput);
    const x = tf.tensor2d([Array.from(vec)]);
    const yProb = model.predictProba(x);
    const prob = (await yProb.data())[0];
    x.dispose();
    yProb.dispose();
    const pred = prob >= 0.5 ? 1 : 0;
    els.predictOut.textContent = `Predicted: ${pred === 1 ? "Player 1 wins" : "Player 1 loses"} (p=${prob.toFixed(3)})`;
  } catch (e) {
    alert("Prediction failed: " + e.message);
  }
}

// Event listeners
els.loadBtn.addEventListener("click", async () => {
  await loadCSV();
  buildPredictForm();
});
els.trainBtn.addEventListener("click", trainModel);
els.evalBtn.addEventListener("click", evaluateModel);
els.saveBtn.addEventListener("click", saveModel);
els.loadModelBtn.addEventListener("click", loadSavedModel);
els.predictBtn.addEventListener("click", (e) => {
  e.preventDefault();
  handlePredict();
});

// Initial state
enableTraining(false);
showPredictPanel(false);
