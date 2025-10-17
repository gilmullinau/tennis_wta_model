// data-loader.js
// Reads local CSV (as text), preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// split train/test, returns tf.Tensors for model training.

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js";

export class DataLoader {
  constructor() {
    this.numericCols = [
      "rank_diff", "pts_diff", "odd_diff",
      "h2h_advantage", "last_winner", "surface_winrate_adv", "year"
    ];
    this.categoricalCols = ["Surface", "Court", "Round"];
    this.dropCols = [
      "Tournament", "Date", "Best of", "Player_1", "Player_2", "Winner", "Score"
    ];
    this.labelCol = "y";
    this.catLevels = {};
    this.scaler = { mean: {}, std: {} };
    this.featureNames = [];
  }

  // NEW: direct text-based loading (no FileReader)
  async loadCSVText(csvText) {
    const delimiter = this._detectDelimiter(csvText);
    const rows = this._parseCSV(csvText, delimiter);
    if (rows.length === 0) throw new Error("Empty CSV file.");
    const headers = rows[0];
    const dataRows = rows.slice(1);

    const raw = dataRows.map((r) => {
      const obj = {};
      headers.forEach((h, i) => (obj[h.trim()] = r[i] === undefined ? "" : r[i]));
      return obj;
    });

    if (!headers.includes(this.labelCol)) {
      throw new Error(`Label column "${this.labelCol}" not found in CSV.`);
    }

    // drop leakage columns & cast numerics
    for (const row of raw) {
      this.dropCols.forEach((c) => delete row[c]);
      row[this.labelCol] = this._toNumber(row[this.labelCol]);
      for (const c of this.numericCols) {
        if (row[c] === undefined) continue;
        row[c] = this._toNumber(row[c]);
      }
    }

    // remove rows with NaNs
    const filtered = raw.filter((row) => {
      if (row[this.labelCol] === null || isNaN(row[this.labelCol])) return false;
      for (const c of this.numericCols) {
        if (row[c] === undefined || row[c] === null || isNaN(row[c])) return false;
      }
      return true;
    });

    // fit categorical levels and build design matrix
    this._fitCategoricals(filtered);
    const { X, y, featureNames } = this._buildDesignMatrix(filtered);
    this.featureNames = featureNames;

    // normalize numeric columns
    this._fitScaler(X, featureNames);
    const Xscaled = this._transformWithScaler(X, featureNames);

    // split train/test
    const { X_train, y_train, X_test, y_test } = this._trainTestSplitStratified(Xscaled, y, 0.2, 42);

    const xTrainTensor = tf.tensor2d(X_train);
    const yTrainTensor = tf.tensor2d(y_train.map((v) => [v]));
    const xTestTensor = tf.tensor2d(X_test);
    const yTestTensor = tf.tensor2d(y_test.map((v) => [v]));

    return {
      X_train: xTrainTensor, y_train: yTrainTensor,
      X_test: xTestTensor, y_test: yTestTensor,
      featureNames: this.featureNames,
      artifacts: {
        catLevels: this.catLevels, scaler: this.scaler,
        numericCols: this.numericCols, categoricalCols: this.categoricalCols
      }
    };
  }

  vectorizeForPredict(userInput) {
    const rowObj = {};
    for (const c of this.numericCols) {
      rowObj[c] = this._toNumber(userInput[c]);
      if (rowObj[c] === null || isNaN(rowObj[c])) {
        throw new Error(`Numeric input "${c}" missing or invalid.`);
      }
    }
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      const provided = (userInput[col] ?? "").toString();
      for (const lvl of levels) {
        const key = `${col}__${lvl}`;
        rowObj[key] = provided === lvl ? 1 : 0;
      }
    }
    const vec = this.featureNames.map((f) => {
      let v = rowObj[f] ?? 0;
      if (this.numericCols.includes(f)) {
        const mean = this.scaler.mean[f] ?? 0;
        const std = this.scaler.std[f] ?? 1;
        v = std === 0 ? 0 : (v - mean) / std;
      }
      return v;
    });
    return Float32Array.from(vec);
  }

  // ---------- helpers ----------
  _detectDelimiter(text) {
    const firstLine = text.split(/\r?\n/)[0] || "";
    const comma = (firstLine.match(/,/g) || []).length;
    const semicolon = (firstLine.match(/;/g) || []).length;
    return semicolon > comma ? ";" : ",";
  }

  _parseCSV(text, delimiter) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    const rows = [];
    for (const line of lines) {
      const row = [];
      let current = "", inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
          else inQuotes = !inQuotes;
        } else if (ch === delimiter && !inQuotes) {
          row.push(current); current = "";
        } else current += ch;
      }
      row.push(current);
      rows.push(row);
    }
    return rows;
  }

  _toNumber(v) {
    if (v === undefined || v === null) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  _fitCategoricals(rows) {
    for (const col of this.categoricalCols) {
      const set = new Set();
      rows.forEach((r) => {
        const val = (r[col] ?? "").toString().trim();
        if (val.length > 0) set.add(val);
      });
      this.catLevels[col] = Array.from(set.values()).sort();
    }
  }

  _buildDesignMatrix(rows) {
    const featureNames = [...this.numericCols];
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      for (const lvl of levels) featureNames.push(`${col}__${lvl}`);
    }
    const X = [], y = [];
    for (const r of rows) {
      const rowArr = [];
      for (const c of this.numericCols) rowArr.push(r[c]);
      for (const col of this.categoricalCols) {
        const levels = this.catLevels[col] || [];
        const val = (r[col] ?? "").toString();
        for (const lvl of levels) rowArr.push(val === lvl ? 1 : 0);
      }
      X.push(rowArr);
      y.push(Math.round(r[this.labelCol]));
    }
    return { X, y, featureNames };
  }

  _fitScaler(X, featureNames) {
    for (const c of this.numericCols) {
      const idx = featureNames.indexOf(c);
      if (idx === -1) continue;
      const n = X.length;
      let sum = 0, sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = X[i][idx];
        sum += v; sumSq += v * v;
      }
      const mean = sum / Math.max(1, n);
      const variance = Math.max(0, sumSq / Math.max(1, n) - mean * mean);
      const std = Math.sqrt(variance);
      this.scaler.mean[c] = mean;
      this.scaler.std[c] = std;
    }
  }

  _transformWithScaler(X, featureNames) {
    const X2 = X.map((row) => row.slice());
    for (const c of this.numericCols) {
      const idx = featureNames.indexOf(c);
      if (idx === -1) continue;
      const mean = this.scaler.mean[c] ?? 0;
      const std = this.scaler.std[c] ?? 1;
      for (let i = 0; i < X2.length; i++) {
        const v = X2[i][idx];
        X2[i][idx] = std === 0 ? 0 : (v - mean) / std;
      }
    }
    return X2;
  }

  _trainTestSplitStratified(X, y, testSize = 0.2, seed = 42) {
    const idx0 = [], idx1 = [];
    for (let i = 0; i < y.length; i++) (y[i] === 1 ? idx1 : idx0).push(i);
    const rng = this._mulberry32(seed);
    const shuffle = (arr) => arr.sort(() => rng() - 0.5);
    shuffle(idx0); shuffle(idx1);
    const n0test = Math.max(1, Math.floor(idx0.length * testSize));
    const n1test = Math.max(1, Math.floor(idx1.length * testSize));
    const testIdx = new Set([...idx0.slice(0, n0test), ...idx1.slice(0, n1test)]);
    const X_train = [], y_train = [], X_test = [], y_test = [];
    for (let i = 0; i < y.length; i++) {
      if (testIdx.has(i)) { X_test.push(X[i]); y_test.push(y[i]); }
      else { X_train.push(X[i]); y_train.push(y[i]); }
    }
    return { X_train, y_train, X_test, y_test };
  }

  _mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
}
