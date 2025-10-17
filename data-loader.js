// data-loader.js
// Reads CSV text, preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// stratified split train/test, returns tf.Tensors and feature metadata. Also builds a player
// directory so the UI can derive feature vectors from player names and contextual inputs.
import { PlayerCatalog } from "./player-features.js";

const tf = window.tf;

export class DataLoader {
  constructor() {
    this.numericCandidates = [
      "rank_diff",
      "pts_diff",
      "odd_diff",
      "elo_diff",
      "surface_elo_diff",
      "form_diff",
      "surface_form_diff",
      "games_margin_diff",
      "sets_margin_diff",
      "rest_days_diff",
      "tiebreak_rate_diff",
      "rank_trend_diff",
      "points_trend_diff",
      "h2h_advantage",
      "last_winner",
      "last_winner_indicator",
      "surface_winrate_adv",
      "year",
    ];
    this.categoricalCandidates = ["Surface", "Court", "Round"];
    this.dropCols = [
      "Tournament", "Date", "Best of", "Player_1", "Player_2", "Winner", "Score",
      "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2",
      "tournament", "date", "best_of", "player_1", "player_2", "winner", "score",
      "rank_1", "rank_2", "pts_1", "pts_2", "odd_1", "odd_2"
    ];

    this.numericCols = [];
    this.categoricalCols = [];
    this.labelCol = "y";
    this.catLevels = {};
    this.scaler = { mean: {}, std: {} };
    this.featureNames = [];

    // Metadata for player-driven predictions
    this.players = [];
    this.surfaces = [];
    this.rounds = [];
    this.courts = [];
    this.latestYear = null;

    this.headerAliases = new Map();
    this.catalog = null;
  }

  async loadCSVText(csvText) {
    const delimiter = this._detectDelimiter(csvText);
    const rows = this._parseCSV(csvText, delimiter);
    if (rows.length === 0) throw new Error("Empty CSV file.");
    const headers = rows[0].map((h) => (h ?? "").trim());
    const dataRows = rows.slice(1);

    headers.forEach((h) => {
      const key = (h ?? "").toString().trim().toLowerCase();
      if (key && !this.headerAliases.has(key)) this.headerAliases.set(key, h);
    });

    const headerSet = new Set(headers);
    this.numericCols = this.numericCandidates.filter((c) => headerSet.has(c));
    this.categoricalCols = this.categoricalCandidates.filter((c) => headerSet.has(c));
    if (this.numericCols.length === 0) {
      throw new Error("No numeric feature columns found in CSV.");
    }

    const raw = dataRows.map((row) => {
      const obj = {};
      headers.forEach((h, idx) => {
        obj[h] = row[idx] === undefined ? "" : row[idx];
      });
      return obj;
    });

    if (!headers.includes(this.labelCol)) {
      throw new Error(`Label column "${this.labelCol}" not found in CSV.`);
    }

    // Build player catalog and metadata before mutating rows
    this.catalog = new PlayerCatalog(raw, { headerAliases: this.headerAliases });
    this.players = this.catalog.listPlayers();
    this.surfaces = this.catalog.listSurfaces();
    this.rounds = this.catalog.listRounds();
    this.courts = this.catalog.listCourts();
    this.latestYear = this.catalog.getLatestYear();

    // Drop leakage columns and cast numeric values
    for (const row of raw) {
      for (const c of this.dropCols) {
        if (c in row) delete row[c];
      }
      row[this.labelCol] = this._toNumber(row[this.labelCol]);
      for (const c of this.numericCols) {
        if (row[c] === undefined) continue;
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
    if (filtered.length < 10) {
      throw new Error(`Too few valid rows: ${filtered.length}`);
    }

    // Prepare categorical encoders
    this._fitCategoricals(filtered);
    const { X, y, featureNames } = this._buildDesignMatrix(filtered);
    this.featureNames = featureNames;

    // Fit/transform scaler
    this._fitScaler(X, featureNames);
    const Xscaled = this._transformWithScaler(X, featureNames);

    const { X_train, y_train, X_test, y_test } = this._trainTestSplitStratified(Xscaled, y, 0.2, 42);

    const xTrainTensor = tf.tensor2d(X_train, [X_train.length, featureNames.length], "float32");
    const yTrainTensor = tf.tensor2d(y_train.map((v) => [v]), [y_train.length, 1], "float32");
    const xTestTensor = tf.tensor2d(X_test, [X_test.length, featureNames.length], "float32");
    const yTestTensor = tf.tensor2d(y_test.map((v) => [v]), [y_test.length, 1], "float32");

    return {
      X_train: xTrainTensor,
      y_train: yTrainTensor,
      X_test: xTestTensor,
      y_test: yTestTensor,
      featureNames: this.featureNames,
      artifacts: {
        catLevels: this.catLevels,
        scaler: this.scaler,
        numericCols: this.numericCols,
        categoricalCols: this.categoricalCols,
      },
      metadata: {
        players: this.players,
        surfaces: this.surfaces,
        rounds: this.rounds,
        courts: this.courts,
      },
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

  vectorizeFromPlayers({ player1, player2, surface = null, court = null, round = null, year = null } = {}) {
    if (!this.catalog) throw new Error("Player metadata unavailable. Reload dataset.");

    const surfaceLevel = surface ?? this._defaultCategorical("Surface", this.catalog.defaultSurface());
    const courtLevel = court ?? this._defaultCategorical("Court", this.catalog.defaultCourt());
    const roundLevel = round ?? this._defaultCategorical("Round", this.catalog.defaultRound());

    const prepared = this.catalog.prepareFeatures({
      player1,
      player2,
      features: this.numericCols,
      context: {
        surface: surfaceLevel,
        court: courtLevel,
        round: roundLevel,
        year,
      },
    });

    const catInputs = {};
    if (this.categoricalCols.includes("Surface")) catInputs.Surface = prepared.context.surface;
    if (this.categoricalCols.includes("Court")) catInputs.Court = prepared.context.court;
    if (this.categoricalCols.includes("Round")) catInputs.Round = prepared.context.round;

    const vec = this.vectorizeForPredict({ ...prepared.numericValues, ...catInputs });
    return {
      vector: vec,
      players: prepared.players,
      surface: prepared.context.surface,
      court: prepared.context.court,
      round: prepared.context.round,
      missingFeatures: prepared.missing,
    };
  }

  _defaultCategorical(col, fallback = "") {
    const levels = this.catLevels[col] || [];
    return levels.length ? levels[0] : fallback;
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
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          if (inQuotes && line[i + 1] === '"') {
            current += '"';
            i++;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (ch === delimiter && !inQuotes) {
          row.push(current);
          current = "";
        } else {
          current += ch;
        }
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

  _buildDesignMatrix(rows) {
    const featureNames = [...this.numericCols];
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      for (const lvl of levels) featureNames.push(`${col}__${lvl}`);
    }
    const X = [];
    const y = [];
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
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = X[i][idx];
        sum += v;
        sumSq += v * v;
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
    const idx0 = [];
    const idx1 = [];
    for (let i = 0; i < y.length; i++) (y[i] === 1 ? idx1 : idx0).push(i);
    const rng = this._mulberry32(seed);
    const shuffle = (arr) => arr.sort(() => rng() - 0.5);
    shuffle(idx0);
    shuffle(idx1);
    const n0test = Math.max(1, Math.floor(idx0.length * testSize));
    const n1test = Math.max(1, Math.floor(idx1.length * testSize));
    const testIdx = new Set([...idx0.slice(0, n0test), ...idx1.slice(0, n1test)]);
    const X_train = [];
    const y_train = [];
    const X_test = [];
    const y_test = [];
    for (let i = 0; i < y.length; i++) {
      if (testIdx.has(i)) {
        X_test.push(X[i]);
        y_test.push(y[i]);
      } else {
        X_train.push(X[i]);
        y_train.push(y[i]);
      }
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
