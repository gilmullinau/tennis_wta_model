// data-loader.js
// Reads CSV text, preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// stratified split train/test, returns tf.Tensors and feature metadata.
const tf = window.tf;

export class DataLoader {
  constructor() {
    this.numericCols = [
      "rank_diff", "pts_diff", "odd_diff",
      "h2h_advantage", "last_winner", "surface_winrate_adv", "year"
    ];
    this.categoricalCols = ["Surface", "Court", "Round"];
    this.dropCols = [
      "Tournament", "Date", "Best of", "Player_1", "Player_2", "Winner", "Score",
      "Rank_1","Rank_2","Pts_1","Pts_2","Odd_1","Odd_2"
    ];
    this.labelCol = "y";
    this.catLevels = {};
    this.scaler = { mean: {}, std: {} };
    this.featureNames = [];
  }

  async loadCSVText(csvText) {
    const delimiter = this._detectDelimiter(csvText);
    const rows = this._parseCSV(csvText, delimiter);
    if (rows.length === 0) throw new Error("Empty CSV file.");
    const headers = rows[0].map(h => (h ?? "").trim());
    const dataRows = rows.slice(1);

    const raw = dataRows.map((r) => {
      const obj = {};
      headers.forEach((h, i) => (obj[h] = r[i] === undefined ? "" : r[i]));
      return obj;
    });

    if (!headers.includes(this.labelCol)) {
      throw new Error(`Label column "${this.labelCol}" not found in CSV.`);
    }

    const missingNumeric = this.numericCols.filter((c) => !headers.includes(c));
    const missingCategorical = this.categoricalCols.filter((c) => !headers.includes(c));
    if (missingNumeric.length > 0 || missingCategorical.length > 0) {
      const missing = [];
      if (missingNumeric.length > 0) missing.push(`numeric: ${missingNumeric.join(", ")}`);
      if (missingCategorical.length > 0) missing.push(`categorical: ${missingCategorical.join(", ")}`);
      throw new Error(`Missing expected columns — ${missing.join("; ")}`);
    }

    // Drop leakage columns; cast types
    for (const row of raw) {
      for (const c of this.dropCols) if (c in row) delete row[c];
      row[this.labelCol] = this._toNumber(row[this.labelCol]);
      for (const c of this.numericCols) {
        row[c] = this._toNumber(row[c]);
      }
    }

    // Filter invalid rows
    const filtered = raw.filter((row) => {
      if (!Number.isFinite(row[this.labelCol])) return false;
      for (const c of this.numericCols) {
        if (!Number.isFinite(row[c])) return false;
      }
      return true;
    });
    if (filtered.length < 10) throw new Error(`Too few valid rows: ${filtered.length}`);

    // Stratified split on raw rows to avoid leakage
    const { trainRows, testRows } = this._splitRowsStratified(filtered, 0.2, 42);

    // Fit categorical levels and scalers only on training data
    this._fitCategoricals(trainRows);
    const { X: X_train_raw, y: y_train, featureNames } = this._buildDesignMatrix(trainRows);
    this.featureNames = featureNames;
    this._fitScaler(X_train_raw, featureNames);

    const X_train_scaled = this._transformWithScaler(X_train_raw, featureNames);
    const { X: X_test_raw, y: y_test } = this._buildDesignMatrix(testRows, featureNames);
    const X_test_scaled = this._transformWithScaler(X_test_raw, featureNames);

    // To tensors
    const xTrainTensor = tf.tensor2d(X_train_scaled, [X_train_scaled.length, featureNames.length], "float32");
    const yTrainTensor = tf.tensor2d(y_train.map(v => [v]), [y_train.length, 1], "float32");
    const xTestTensor = tf.tensor2d(X_test_scaled, [X_test_scaled.length, featureNames.length], "float32");
    const yTestTensor = tf.tensor2d(y_test.map(v => [v]), [y_test.length, 1], "float32");

    return {
      X_train: xTrainTensor, y_train: yTrainTensor,
      X_test: xTestTensor, y_test: yTestTensor,
      featureNames: this.featureNames,
      artifacts: {
        catLevels: this.catLevels,
        scaler: this.scaler,
        numericCols: this.numericCols,
        categoricalCols: this.categoricalCols,
        featureNames: this.featureNames
      }
    };
  }

  vectorizeForPredict(userInput) {
    const rowObj = {};
    for (const c of this.numericCols) {
      const v = this._toNumber(userInput[c]);
      if (!Number.isFinite(v)) throw new Error(`Numeric input "${c}" missing or invalid.`);
      rowObj[c] = v;
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
    if (v === undefined || v === null || v === "") return NaN;
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }

  _fitCategoricals(rows) {
    for (const col of this.categoricalCols) {
      const set = new Set();
      for (const r of rows) {
        const val = (r[col] ?? "").toString().trim();
        if (val.length > 0) set.add(val);
      }
      this.catLevels[col] = Array.from(set.values()).sort();
    }
  }

  _buildDesignMatrix(rows, featureNames = null) {
    const resolvedFeatureNames = featureNames ? featureNames.slice() : this._featureNamesFromArtifacts();
    const X = [], y = [];
    for (const r of rows) {
      const rowArr = [];
      for (const feat of resolvedFeatureNames) {
        if (this.numericCols.includes(feat)) {
          rowArr.push(r[feat]);
        } else {
          const [col, lvl] = feat.split("__");
          const val = (r[col] ?? "").toString();
          rowArr.push(val === lvl ? 1 : 0);
        }
      }
      X.push(rowArr);
      y.push(Math.round(r[this.labelCol]));
    }
    return { X, y, featureNames: resolvedFeatureNames };
  }

  _featureNamesFromArtifacts() {
    const featureNames = [...this.numericCols];
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      for (const lvl of levels) featureNames.push(`${col}__${lvl}`);
    }
    return featureNames;
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

  _splitRowsStratified(rows, testSize = 0.2, seed = 42) {
    const idxByClass = new Map();
    for (let i = 0; i < rows.length; i++) {
      const label = Math.round(rows[i][this.labelCol]);
      if (!idxByClass.has(label)) idxByClass.set(label, []);
      idxByClass.get(label).push(i);
    }

    const rng = this._mulberry32(seed);
    const shuffle = (arr) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    };

    const total = rows.length;
    const targetTest = Math.max(1, Math.round(total * testSize));
    const testIdx = new Set();

    let assigned = 0;
    const classes = Array.from(idxByClass.keys());
    for (const cls of classes) {
      const arr = shuffle(idxByClass.get(cls));
      const clsTotal = arr.length;
      if (clsTotal === 0) continue;
      const desired = Math.max(clsTotal > 0 ? 1 : 0, Math.round(clsTotal * testSize));
      const take = Math.min(clsTotal, desired);
      for (let i = 0; i < take; i++) {
        if (assigned >= targetTest) break;
        testIdx.add(arr[i]);
        assigned++;
      }
    }

    // If still short on test samples (e.g., due to rounding), fill from remaining
    if (assigned < targetTest) {
      const allIdx = Array.from({ length: total }, (_, i) => i);
      shuffle(allIdx);
      for (const idx of allIdx) {
        if (assigned >= targetTest) break;
        if (!testIdx.has(idx)) {
          testIdx.add(idx);
          assigned++;
        }
      }
    }

    const trainRows = [];
    const testRows = [];
    for (let i = 0; i < rows.length; i++) {
      if (testIdx.has(i)) testRows.push(rows[i]);
      else trainRows.push(rows[i]);
    }
    return { trainRows, testRows };
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
