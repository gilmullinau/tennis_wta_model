import { DataLoader } from "./data-loader.js";
import { ModelMLP } from "./gru.js";

let loader = null;
let model = null;
let dataset = null;

// Charts
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

async function autoLoadCSV() {
  try {
    loader = new DataLoader();
    log("Loading default dataset (wta_data.csv)...");
    const response = await fetch("./wta_data.csv");
    const blob = await response.blob();
    const file = new File([blob], "wta_data.csv");
    dataset = await loader.loadCSV(file);

    els.info.textContent = `Loaded wta_data.csv — Train: ${dataset.X_train.shape[0]}, Test: ${dataset.X_test.shape[0]}`;
    log("Dataset successfully loaded.");
    enableTraining(true);
    buildPredictForm();
  } catch (err) {
    log("Error loading dataset: " + err.message);
    alert("Failed to load wta_data.csv. Please check that the file exists in the project root.");
  }
}

async function trainModel() {
  if (!dataset) return alert("Dataset not loaded yet.");

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
        { label: "Loss", data: losses, tension: 0.2 },
        { label: "Val Accuracy", data: valAcc, tension: 0.2 }
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
        { label: "Pred 0", data: [tn, fn] },
        { label: "Pred 1", data: [fp, tp] }
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

function buildPredictForm() {
  if (!loader || !dataset) return;
  const form = els.predictForm;
  form.innerHTML = "";

  loader.numericCols.forEach((c) => {
    const div = document.createElement("div");
    div.className = "form-row";
    div.innerHTML = `<label>${c}</label><input type="number" step="any" name="${c}" required />`;
    form.appendChild(div);
  });

  loader.categoricalCols.forEach((col) => {
    const levels = loader.catLevels[col] || [];
    const div = document.createElement("div");
    div.className = "form-row";
    const opts = levels.map((l) => `<option value="${l}">${l}</option>`).join("");
    div.innerHTML = `<label>${col}</label><select name="${col}">${opts}</select>`;
    form.appendChild(div);
  });
}

async function handlePredict(e) {
  e.preventDefault();
  if (!model || !loader) return alert("Train or load a model first.");
  const formData = new FormData(els.predictForm);
  const userInput = {};
  for (const [k, v] of formData.entries()) userInput[k] = v;
  const vec = loader.vectorizeForPredict(userInput);
  const x = tf.tensor2d([Array.from(vec)]);
  const yProb = model.predictProba(x);
  const prob = (await yProb.data())[0];
  const pred = prob >= 0.5 ? 1 : 0;
  els.predictOut.textContent = `Predicted: ${pred === 1 ? "Player 1 wins" : "Player 1 loses"} (p=${prob.toFixed(3)})`;
  x.dispose(); yProb.dispose();
}

// Event bindings
els.trainBtn.addEventListener("click", trainModel);
els.evalBtn.addEventListener("click", evaluateModel);
els.saveBtn.addEventListener("click", async () => {
  if (model) {
    await model.save();
    log("Model saved.");
  }
});
els.loadModelBtn.addEventListener("click", async () => {
  const m = new ModelMLP(0);
  await m.load();
  model = m;
  log("Model loaded from storage.");
  showPredictPanel(true);
});
els.predictBtn.addEventListener("click", handlePredict);

enableTraining(false);
showPredictPanel(false);
autoLoadCSV(); // auto-load CSV at startup
