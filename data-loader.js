// data-loader.js
// Reads CSV text, preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// stratified split train/test, returns tf.Tensors and feature metadata. Also builds a player
// directory so the UI can derive feature vectors from player names and contextual inputs.
const tf = window.tf;

export class DataLoader {
  constructor() {
    this.numericCandidates = [
      "rank_diff", "pts_diff", "odd_diff",
      "h2h_advantage", "last_winner", "last_winner_indicator",
      "surface_winrate_adv", "year"
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
    this.playerMeta = new Map();
    this.playerLookup = new Map();
    this.h2h = new Map();
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

    // Build player directory and metadata before mutating rows
    this._buildPlayerDirectory(raw);

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
    if (!player1 || !player2) throw new Error("Both players must be provided.");
    const p1 = this._resolvePlayer(player1);
    const p2 = this._resolvePlayer(player2);
    if (!p1) throw new Error(`Player "${player1}" not found in dataset.`);
    if (!p2) throw new Error(`Player "${player2}" not found in dataset.`);
    if (p1 === p2) throw new Error("Choose two different players.");

    const chosenSurface = surface ?? this._defaultSurface();
    const chosenCourt = court ?? this._defaultCategorical("Court");
    const chosenRound = round ?? this._defaultCategorical("Round");

    const ctx = {
      surface: chosenSurface,
      court: chosenCourt,
      round: chosenRound,
      year,
    };

    const numericValues = {};
    const missing = [];
    for (const feature of this.numericCols) {
      const value = this._deriveFeature(feature, p1, p2, ctx);
      if (Number.isFinite(value)) {
        numericValues[feature] = value;
      } else {
        numericValues[feature] = 0;
        missing.push(feature);
      }
    }

    if (this.numericCols.includes("year")) {
      numericValues.year = Number.isFinite(year) ? year : (this.latestYear ?? new Date().getFullYear());
    }

    const catInputs = {};
    if (this.categoricalCols.includes("Surface")) catInputs.Surface = chosenSurface;
    if (this.categoricalCols.includes("Court")) catInputs.Court = chosenCourt;
    if (this.categoricalCols.includes("Round")) catInputs.Round = chosenRound;

    const vec = this.vectorizeForPredict({ ...numericValues, ...catInputs });
    return {
      vector: vec,
      players: { player1: p1, player2: p2 },
      surface: chosenSurface,
      court: chosenCourt,
      round: chosenRound,
      missingFeatures: missing,
    };
  }

  _buildPlayerDirectory(rows) {
    this.playerMeta = new Map();
    this.playerLookup = new Map();
    this.h2h = new Map();
    this.players = [];
    this.surfaces = [];
    this.rounds = [];
    this.courts = [];
    this.latestYear = null;

    const surfaceSet = new Set();
    const roundSet = new Set();
    const courtSet = new Set();

    const parsed = rows
      .map((row, idx) => {
        const player1 = this._stringValue(row, ["player_1", "Player_1"]);
        const player2 = this._stringValue(row, ["player_2", "Player_2"]);
        if (!player1 || !player2) return null;
        const dateStr = this._stringValue(row, ["date", "Date"]);
        const date = dateStr ? new Date(dateStr) : null;
        const ts = date && Number.isFinite(date.getTime()) ? date.getTime() : Number.POSITIVE_INFINITY;
        const surface = this._stringValue(row, ["surface", "Surface"]) || null;
        const round = this._stringValue(row, ["round", "Round"]) || null;
        const court = this._stringValue(row, ["court", "Court"]) || null;
        const winner = this._stringValue(row, ["winner", "Winner"]);

        return {
          row,
          idx,
          ts,
          player1: this._canonical(player1),
          player2: this._canonical(player2),
          winner: this._canonical(winner),
          surface,
          round,
          court,
          date,
        };
      })
      .filter(Boolean)
      .sort((a, b) => (a.ts === b.ts ? a.idx - b.idx : a.ts - b.ts));

    for (const item of parsed) {
      const { row, player1, player2, winner, surface, round, court, date } = item;
      if (surface) surfaceSet.add(surface);
      if (round) roundSet.add(round);
      if (court) courtSet.add(court);
      if (date instanceof Date && Number.isFinite(date.getFullYear())) {
        this.latestYear = Math.max(this.latestYear ?? 0, date.getFullYear());
      }

      const entry1 = this._ensurePlayerEntry(player1);
      const entry2 = this._ensurePlayerEntry(player2);
      const winnerCanonical = winner || "";

      this._updatePlayerEntry(entry1, row, true, surface, winnerCanonical === player1);
      this._updatePlayerEntry(entry2, row, false, surface, winnerCanonical === player2);
      this._updateHeadToHead(player1, player2, winnerCanonical);
    }

    this.players = Array.from(this.playerMeta.keys()).sort((a, b) => a.localeCompare(b));
    this.surfaces = surfaceSet.size ? Array.from(surfaceSet.values()).sort((a, b) => a.localeCompare(b)) : [];
    this.rounds = roundSet.size ? Array.from(roundSet.values()).sort((a, b) => a.localeCompare(b)) : [];
    this.courts = courtSet.size ? Array.from(courtSet.values()).sort((a, b) => a.localeCompare(b)) : [];
  }

  _ensurePlayerEntry(name) {
    if (!name) return null;
    if (!this.playerMeta.has(name)) {
      this.playerMeta.set(name, {
        name,
        lastRank: null,
        lastPts: null,
        lastOdd: null,
        surfaceStats: new Map(),
        totalWins: 0,
        totalMatches: 0,
      });
      this.playerLookup.set(name.toLowerCase(), name);
    }
    return this.playerMeta.get(name);
  }

  _updatePlayerEntry(entry, row, isPlayer1, surface, won) {
    if (!entry) return;
    const rank = this._numericValue(row, isPlayer1 ? ["rank_1", "Rank_1"] : ["rank_2", "Rank_2"]);
    const pts = this._numericValue(row, isPlayer1 ? ["pts_1", "Pts_1"] : ["pts_2", "Pts_2"]);
    const odd = this._numericValue(row, isPlayer1 ? ["odd_1", "Odd_1"] : ["odd_2", "Odd_2"]);

    if (Number.isFinite(rank)) entry.lastRank = rank;
    if (Number.isFinite(pts)) entry.lastPts = pts;
    if (Number.isFinite(odd)) entry.lastOdd = odd;

    if (surface) {
      const stats = entry.surfaceStats.get(surface) || { wins: 0, matches: 0 };
      stats.matches += 1;
      if (won) stats.wins += 1;
      entry.surfaceStats.set(surface, stats);
    }

    entry.totalMatches += 1;
    if (won) entry.totalWins += 1;
  }

  _updateHeadToHead(player1, player2, winner) {
    const update = (left, right) => {
      if (!left || !right) return;
      const key = `${left}|||${right}`;
      const record = this.h2h.get(key) || { wins: 0, losses: 0, lastWinner: "", advantage: 0 };
      if (winner && winner === left) {
        record.wins += 1;
        record.lastWinner = left;
      } else if (winner && winner === right) {
        record.losses += 1;
        record.lastWinner = right;
      }
      const total = record.wins + record.losses;
      record.advantage = total > 0 ? (record.wins - record.losses) / total : 0;
      this.h2h.set(key, record);
    };

    update(player1, player2);
    update(player2, player1);
  }

  _deriveFeature(feature, player1, player2, ctx) {
    const entry1 = this.playerMeta.get(player1);
    const entry2 = this.playerMeta.get(player2);
    if (!entry1 || !entry2) return NaN;

    switch (feature) {
      case "rank_diff": {
        const a = this._finite(entry2.lastRank);
        const b = this._finite(entry1.lastRank);
        return this._diff(a, b);
      }
      case "pts_diff": {
        const a = this._finite(entry1.lastPts);
        const b = this._finite(entry2.lastPts);
        return this._diff(a, b);
      }
      case "odd_diff": {
        const a = this._finite(entry2.lastOdd);
        const b = this._finite(entry1.lastOdd);
        return this._diff(a, b);
      }
      case "surface_winrate_adv": {
        const rate1 = this._surfaceWinrate(entry1, ctx.surface);
        const rate2 = this._surfaceWinrate(entry2, ctx.surface);
        if (!Number.isFinite(rate1) || !Number.isFinite(rate2)) return NaN;
        return rate1 - rate2;
      }
      case "h2h_advantage": {
        const stats = this._getHeadToHead(player1, player2);
        return stats.advantage ?? 0;
      }
      case "last_winner": {
        const stats = this._getHeadToHead(player1, player2);
        if (!stats.lastWinner) return 0;
        return stats.lastWinner === player1 ? 1 : 0;
      }
      case "last_winner_indicator": {
        const stats = this._getHeadToHead(player1, player2);
        if (!stats.lastWinner) return 0;
        if (stats.lastWinner === player1) return 1;
        if (stats.lastWinner === player2) return -1;
        return 0;
      }
      case "year":
        return ctx.year ?? this.latestYear ?? new Date().getFullYear();
      default:
        return NaN;
    }
  }

  _surfaceWinrate(entry, surface) {
    if (!entry) return NaN;
    if (surface && entry.surfaceStats.has(surface)) {
      const stats = entry.surfaceStats.get(surface);
      if (stats.matches > 0) return stats.wins / stats.matches;
    }
    if (entry.totalMatches > 0) {
      return entry.totalWins / entry.totalMatches;
    }
    return NaN;
  }

  _getHeadToHead(player1, player2) {
    const key = `${player1}|||${player2}`;
    return this.h2h.get(key) || { advantage: 0, lastWinner: "" };
  }

  _defaultSurface() {
    if (this.surfaces && this.surfaces.length) return this.surfaces[0];
    return "Hard";
  }

  _defaultCategorical(col) {
    const levels = this.catLevels[col] || [];
    return levels.length ? levels[0] : "";
  }

  _resolvePlayer(name) {
    if (!name) return null;
    const trimmed = name.trim();
    if (!trimmed) return null;
    if (this.playerMeta.has(trimmed)) return trimmed;
    const lower = trimmed.toLowerCase();
    if (this.playerLookup.has(lower)) return this.playerLookup.get(lower);
    for (const candidate of this.players) {
      if (candidate.toLowerCase().includes(lower)) {
        return candidate;
      }
    }
    return null;
  }

  _stringValue(row, candidates) {
    for (const key of candidates) {
      const alias = this.headerAliases.get(key.toLowerCase()) || key;
      if (alias in row && row[alias] != null) {
        const val = row[alias].toString().trim();
        if (val.length > 0) return val;
      }
    }
    return "";
  }

  _numericValue(row, candidates) {
    for (const key of candidates) {
      const alias = this.headerAliases.get(key.toLowerCase()) || key;
      if (alias in row) {
        const num = this._toNumber(row[alias]);
        if (Number.isFinite(num)) return num;
      }
    }
    return NaN;
  }

  _canonical(name) {
    return (name || "").toString().trim();
  }

  _finite(value) {
    return Number.isFinite(value) ? value : NaN;
  }

  _diff(a, b) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return NaN;
    return a - b;
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
