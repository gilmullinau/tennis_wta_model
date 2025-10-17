// data-loader.js
// Reads CSV text, preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// stratified split train/test, returns tf.Tensors and feature metadata. Also builds player indexes
// to allow selecting players in the prediction UI.
const tf = window.tf;

export class DataLoader {
  constructor() {
    this.numericCandidates = [
      "rank_diff", "pts_diff", "odd_diff",
      "elo_diff", "surface_elo_diff",
      "form_diff", "surface_form_diff",
      "games_margin_diff", "sets_margin_diff",
      "rest_days_diff", "tiebreak_rate_diff",
      "rank_trend_diff", "points_trend_diff",
      "h2h_advantage", "last_winner_indicator", "last_winner",
      "surface_winrate_adv", "year"
    ];
    this.categoricalCandidates = ["Surface", "Court", "Round"];
    this.dropCols = [
      "Tournament", "Date", "Best of", "Player_1", "Player_2", "Winner", "Score",
      "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2", "match_id",
      "tournament", "date", "best_of", "player_1", "player_2", "winner", "score",
      "rank_1", "rank_2", "pts_1", "pts_2", "odd_1", "odd_2"
    ];
    this.numericCols = [];
    this.categoricalCols = [];
    this.labelCol = "y";
    this.catLevels = {};
    this.scaler = { mean: {}, std: {} };
    this.featureNames = [];
    this.headerAliases = new Map();

    // Player metadata for prediction UI
    this.players = [];
    this.surfaces = [];
    this.rounds = [];
    this.courts = [];
    this.latestYear = null;
    this.playerLookup = new Map(); // lowercase -> canonical name
    this.playerMeta = new Map();
    this.h2h = new Map();

    this.historyWindow = 10;
    this.surfaceHistoryWindow = 6;

    this.metricSources = {
      rank: { p1: ["rank_1", "rank1"], p2: ["rank_2", "rank2"] },
      pts: { p1: ["pts_1", "points_1"], p2: ["pts_2", "points_2"] },
      odd: { p1: ["odd_1", "odds_1"], p2: ["odd_2", "odds_2"] },
      elo: { p1: ["elo_p1", "elo1", "elo_pre", "elo_player1"], p2: ["elo_p2", "elo2", "elo_player2", "elo_opponent"] },
      surfaceElo: { p1: ["surface_elo_p1", "surface_elo1"], p2: ["surface_elo_p2", "surface_elo2"] },
      form: { p1: ["p1_form", "form_p1"], p2: ["p2_form", "form_p2"] },
      surfaceForm: { p1: ["p1_surface_form", "surface_form_p1"], p2: ["p2_surface_form", "surface_form_p2"] },
      gamesMargin: { p1: ["p1_games_margin", "games_margin_p1"], p2: ["p2_games_margin", "games_margin_p2"] },
      setsMargin: { p1: ["p1_sets_margin", "sets_margin_p1"], p2: ["p2_sets_margin", "sets_margin_p2"] },
      restDays: { p1: ["p1_rest_days", "rest_days_p1"], p2: ["p2_rest_days", "rest_days_p2"] },
      tiebreakRate: { p1: ["p1_tiebreak_rate", "tiebreak_rate_p1"], p2: ["p2_tiebreak_rate", "tiebreak_rate_p2"] },
      rankTrend: { p1: ["p1_rank_trend", "rank_trend_p1"], p2: ["p2_rank_trend", "rank_trend_p2"] },
      pointsTrend: { p1: ["p1_points_trend", "points_trend_p1"], p2: ["p2_points_trend", "points_trend_p2"] },
    };
  }

  async loadCSVText(csvText) {
    const delimiter = this._detectDelimiter(csvText);
    const rows = this._parseCSV(csvText, delimiter);
    if (rows.length === 0) throw new Error("Empty CSV file.");
    const headers = rows[0].map((h) => (h ?? "").trim());
    const dataRows = rows.slice(1);

    const headerSet = new Set(headers);
    headers.forEach((h) => {
      const key = (h ?? "").toString().trim().toLowerCase();
      if (key && !this.headerAliases.has(key)) this.headerAliases.set(key, h);
    });
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

    // Build rich metadata (players, surfaces, recent form) before mutating rows
    this._buildPlayerMetadata(raw);

    // Drop leakage columns; cast types
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
    if (filtered.length < 10) throw new Error(`Too few valid rows: ${filtered.length}`);

    // Fit categorical levels and build design matrix
    this._fitCategoricals(filtered);
    const { X, y, featureNames } = this._buildDesignMatrix(filtered);
    this.featureNames = featureNames;

    // Normalize numeric columns
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
    const ctx = { surface: chosenSurface, court: chosenCourt, round: chosenRound, year };

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

  _buildPlayerMetadata(rows) {
    this.playerMeta = new Map();
    this.playerLookup = new Map();
    this.h2h = new Map();
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
        const ts = date && Number.isFinite(date.getTime()) ? date : null;
        const surface = this._stringValue(row, ["surface", "Surface"]) || null;
        const round = this._stringValue(row, ["round", "Round"]) || null;
        const court = this._stringValue(row, ["court", "Court"]) || null;
        const winner = this._stringValue(row, ["winner", "Winner"]);
        return { row, player1, player2, date: ts, idx, surface, round, court, winner };
      })
      .filter(Boolean)
      .sort((a, b) => {
        if (a.date && b.date) {
          const diff = a.date - b.date;
          if (diff !== 0) return diff;
        } else if (a.date && !b.date) {
          return -1;
        } else if (!a.date && b.date) {
          return 1;
        }
        return a.idx - b.idx;
      });

    for (const item of parsed) {
      const { player1, player2, row, date, surface, round, court, winner } = item;
      if (surface) surfaceSet.add(surface);
      if (round) roundSet.add(round);
      if (court) courtSet.add(court);
      if (date instanceof Date && Number.isFinite(date.getFullYear())) {
        this.latestYear = Math.max(this.latestYear ?? 0, date.getFullYear());
      }

      this._updatePlayerEntry(player1, row, true, date, surface, winner === player1);
      this._updatePlayerEntry(player2, row, false, date, surface, winner === player2);
      this._updateHeadToHead(player1, player2, winner);
    }

    this.players = Array.from(this.playerMeta.keys()).sort((a, b) => a.localeCompare(b));
    this.surfaces = surfaceSet.size ? Array.from(surfaceSet.values()).sort((a, b) => a.localeCompare(b)) : [];
    this.rounds = roundSet.size ? Array.from(roundSet.values()).sort((a, b) => a.localeCompare(b)) : [];
    this.courts = courtSet.size ? Array.from(courtSet.values()).sort((a, b) => a.localeCompare(b)) : [];
  }

  _updatePlayerEntry(name, row, isPlayer1, date, surface, won) {
    const canonical = name.trim();
    if (!this.playerMeta.has(canonical)) {
      this.playerMeta.set(canonical, {
        name: canonical,
        lastMatchDate: null,
        latest: {},
        rankHistory: [],
        pointsHistory: [],
        recent: [],
        surfaceRecent: new Map(),
        surfaceStats: new Map(),
      });
      this.playerLookup.set(canonical.toLowerCase(), canonical);
    }
    const entry = this.playerMeta.get(canonical);
    if (!entry) return;

    for (const [metric, sources] of Object.entries(this.metricSources)) {
      const candidates = (isPlayer1 ? sources.p1 : sources.p2) || [];
      const value = this._numericValue(row, candidates);
      if (!Number.isFinite(value)) continue;
      if (metric === "surfaceElo") {
        const sEntry = this._ensureSurfaceEntry(entry, surface);
        sEntry.latest.surfaceElo = value;
      } else if (metric === "surfaceForm") {
        const sEntry = this._ensureSurfaceEntry(entry, surface);
        sEntry.latest.surfaceForm = value;
      } else {
        entry.latest[metric] = value;
      }
      if (metric === "rank") {
        entry.rankHistory.push(value);
        if (entry.rankHistory.length > 5) entry.rankHistory.shift();
        if (entry.rankHistory.length >= 2) {
          const n = entry.rankHistory.length;
          entry.latest.rankTrend = entry.rankHistory[n - 1] - entry.rankHistory[n - 2];
        }
      }
      if (metric === "pts") {
        entry.pointsHistory.push(value);
        if (entry.pointsHistory.length > 5) entry.pointsHistory.shift();
        if (entry.pointsHistory.length >= 2) {
          const n = entry.pointsHistory.length;
          entry.latest.pointsTrend = entry.pointsHistory[n - 1] - entry.pointsHistory[n - 2];
        }
      }
    }

    if (date instanceof Date && Number.isFinite(date.getTime())) {
      if (entry.lastMatchDate instanceof Date) {
        const diffDays = Math.max(0, Math.round((date - entry.lastMatchDate) / (1000 * 60 * 60 * 24)));
        entry.latest.restDays = Math.min(60, diffDays);
      }
      entry.lastMatchDate = date;
    }

    const historyRecord = this._historyRecord(row, isPlayer1, won, surface);
    if (historyRecord) {
      entry.recent.push(historyRecord);
      if (entry.recent.length > this.historyWindow) entry.recent.shift();
      this._recomputeAverages(entry);
      const sEntry = this._ensureSurfaceEntry(entry, surface);
      sEntry.recent.push(historyRecord);
      if (sEntry.recent.length > this.surfaceHistoryWindow) sEntry.recent.shift();
      this._recomputeSurfaceAverages(sEntry);
      const stats = this._ensureSurfaceStats(entry, surface);
      stats.matches += 1;
      stats.wins += won ? 1 : 0;
    }
  }

  _historyRecord(row, isPlayer1, won, surface) {
    const gamesDelta = this._numericValue(row, ["games_delta"]);
    const setsDelta = this._numericValue(row, ["sets_delta"]);
    const tieBreaks = this._numericValue(row, ["tie_breaks_played", "tie_breaks", "tiebreaks"]);
    const record = {
      won: won ? 1 : 0,
      gamesDelta: Number.isFinite(gamesDelta) ? (isPlayer1 ? gamesDelta : -gamesDelta) : null,
      setsDelta: Number.isFinite(setsDelta) ? (isPlayer1 ? setsDelta : -setsDelta) : null,
      tiebreak: Number.isFinite(tieBreaks) ? (tieBreaks > 0 ? 1 : 0) : null,
      surface,
    };
    return record;
  }

  _recomputeAverages(entry) {
    const { recent } = entry;
    if (!recent.length) return;
    entry.latest.form = this._avg(recent.map((r) => r.won), 0, 3);
    entry.latest.gamesMargin = this._avg(
      recent.map((r) => r.gamesDelta).filter((v) => Number.isFinite(v)),
      0,
      3
    );
    entry.latest.setsMargin = this._avg(
      recent.map((r) => r.setsDelta).filter((v) => Number.isFinite(v)),
      0,
      3
    );
    entry.latest.tiebreakRate = this._avg(
      recent.map((r) => r.tiebreak).filter((v) => Number.isFinite(v)),
      0,
      3
    );
  }

  _ensureSurfaceEntry(entry, surface) {
    const key = surface || "Unknown";
    if (!entry.surfaceRecent.has(key)) {
      entry.surfaceRecent.set(key, {
        recent: [],
        latest: {},
      });
    }
    return entry.surfaceRecent.get(key);
  }

  _ensureSurfaceStats(entry, surface) {
    const key = surface || "Unknown";
    if (!entry.surfaceStats.has(key)) {
      entry.surfaceStats.set(key, { wins: 0, matches: 0 });
    }
    return entry.surfaceStats.get(key);
  }

  _recomputeSurfaceAverages(surfaceEntry) {
    const { recent } = surfaceEntry;
    if (!recent.length) return;
    surfaceEntry.latest.surfaceForm = this._avg(recent.map((r) => r.won), 0, 2);
    surfaceEntry.latest.tiebreakRate = this._avg(
      recent.map((r) => r.tiebreak).filter((v) => Number.isFinite(v)),
      0,
      2
    );
  }

  _avg(values, fallback = 0, minCount = 1) {
    const filtered = values.filter((v) => Number.isFinite(v));
    if (filtered.length < minCount) return fallback;
    const sum = filtered.reduce((acc, v) => acc + v, 0);
    return sum / filtered.length;
  }

  _updateHeadToHead(player1, player2, winner) {
    const key12 = this._pairKey(player1, player2);
    const key21 = this._pairKey(player2, player1);
    const normalizedWinner = (winner ?? "").toLowerCase();

    const update = (key) => {
      const rec = this.h2h.get(key) || { wins: 0, losses: 0, lastWinnerIndicator: 0 };
      const [left, right] = key.split("|||");
      if (normalizedWinner) {
        if (normalizedWinner === left.toLowerCase()) {
          rec.wins += 1;
          rec.lastWinnerIndicator = 1;
        } else if (normalizedWinner === right.toLowerCase()) {
          rec.losses += 1;
          rec.lastWinnerIndicator = -1;
        }
      }
      const total = rec.wins + rec.losses;
      rec.advantage = total > 0 ? (rec.wins - rec.losses) / total : 0;
      this.h2h.set(key, rec);
    };

    update(key12);
    update(key21);
  }

  _pairKey(a, b) {
    return `${a}|||${b}`;
  }

  _deriveFeature(feature, player1, player2, ctx) {
    const entry1 = this.playerMeta.get(player1);
    const entry2 = this.playerMeta.get(player2);
    if (!entry1 || !entry2) return NaN;
    const surface = ctx.surface || this._defaultSurface();

    const diff = (metric, { surfaceSpecific = false } = {}) => {
      const a = this._metricValue(entry1, metric, surfaceSpecific ? surface : null);
      const b = this._metricValue(entry2, metric, surfaceSpecific ? surface : null);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return NaN;
      return a - b;
    };

    switch (feature) {
      case "rank_diff":
        return diff("rank");
      case "pts_diff":
        return diff("pts");
      case "odd_diff":
        return diff("odd");
      case "elo_diff":
        return diff("elo");
      case "surface_elo_diff":
        return diff("surfaceElo", { surfaceSpecific: true });
      case "form_diff":
        return diff("form");
      case "surface_form_diff":
        return diff("surfaceForm", { surfaceSpecific: true });
      case "games_margin_diff":
        return diff("gamesMargin");
      case "sets_margin_diff":
        return diff("setsMargin");
      case "rest_days_diff":
        return diff("restDays");
      case "tiebreak_rate_diff":
        return diff("tiebreakRate");
      case "rank_trend_diff":
        return diff("rankTrend");
      case "points_trend_diff":
        return diff("pointsTrend");
      case "surface_winrate_adv":
        return this._surfaceWinrateAdvantage(entry1, entry2, surface);
      case "h2h_advantage": {
        const stats = this._getHeadToHeadStats(player1, player2);
        return stats.advantage;
      }
      case "last_winner_indicator": {
        const stats = this._getHeadToHeadStats(player1, player2);
        return stats.lastWinnerIndicator ?? 0;
      }
      case "last_winner": {
        const stats = this._getHeadToHeadStats(player1, player2);
        return stats.lastWinnerIndicator > 0 ? 1 : 0;
      }
      case "year":
        return ctx.year ?? this.latestYear ?? new Date().getFullYear();
      default:
        return 0;
    }
  }

  _metricValue(entry, metric, surface = null) {
    if (surface && entry.surfaceRecent.has(surface)) {
      const surfaceEntry = entry.surfaceRecent.get(surface);
      if (surfaceEntry && surfaceEntry.latest && Number.isFinite(surfaceEntry.latest[metric])) {
        return surfaceEntry.latest[metric];
      }
    }
    if (entry.latest && Number.isFinite(entry.latest[metric])) return entry.latest[metric];
    if (surface) {
      const fallback = entry.surfaceRecent.get("Unknown");
      if (fallback && fallback.latest && Number.isFinite(fallback.latest[metric])) {
        return fallback.latest[metric];
      }
    }
    return NaN;
  }

  _surfaceWinrateAdvantage(entry1, entry2, surface) {
    const rate1 = this._surfaceWinrate(entry1, surface);
    const rate2 = this._surfaceWinrate(entry2, surface);
    if (!Number.isFinite(rate1) || !Number.isFinite(rate2)) return NaN;
    return rate1 - rate2;
  }

  _surfaceWinrate(entry, surface) {
    const stats = entry.surfaceStats.get(surface) || entry.surfaceStats.get("Unknown");
    if (stats && stats.matches > 0) {
      return stats.wins / stats.matches;
    }
    const totalWins = Array.from(entry.surfaceStats.values()).reduce((acc, s) => acc + s.wins, 0);
    const totalMatches = Array.from(entry.surfaceStats.values()).reduce((acc, s) => acc + s.matches, 0);
    if (totalMatches > 0) return totalWins / totalMatches;
    return NaN;
  }

  _getHeadToHeadStats(player1, player2) {
    const key = this._pairKey(player1, player2);
    return this.h2h.get(key) || { advantage: 0, lastWinnerIndicator: 0 };
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
    const exact = this.playerMeta.has(trimmed) ? trimmed : null;
    if (exact) return exact;
    const lower = trimmed.toLowerCase();
    if (this.playerLookup.has(lower)) return this.playerLookup.get(lower);
    // try partial match
    for (const candidate of this.playerMeta.keys()) {
      if (candidate.toLowerCase().includes(lower)) {
        return candidate;
      }
    }
    return null;
  }

  _stringValue(row, candidates) {
    for (const key of candidates) {
      if (key in row && row[key] !== undefined && row[key] !== null) {
        const val = row[key].toString().trim();
        if (val.length > 0) return val;
      }
      const alias = this.headerAliases.get(key.toLowerCase());
      if (alias && alias in row && row[alias] != null) {
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
