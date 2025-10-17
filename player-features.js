// player-features.js
// Builds player metadata (recent ranks, odds, surface records, head-to-head) from raw rows
// and derives numeric features for prediction based on selected competitors.

export class PlayerCatalog {
  constructor(rows, { headerAliases = new Map() } = {}) {
    this.headerAliases = headerAliases;
    this.playerMeta = new Map();
    this.playerLookup = new Map();
    this.h2h = new Map();
    this.players = [];
    this.surfaces = [];
    this.rounds = [];
    this.courts = [];
    this.latestYear = null;

    this._build(rows);
  }

  listPlayers() {
    return this.players.slice();
  }

  listSurfaces() {
    return this.surfaces.slice();
  }

  listRounds() {
    return this.rounds.slice();
  }

  listCourts() {
    return this.courts.slice();
  }

  getLatestYear() {
    return this.latestYear;
  }

  defaultSurface() {
    return this.surfaces.length ? this.surfaces[0] : "Hard";
  }

  defaultCourt() {
    return this.courts.length ? this.courts[0] : "";
  }

  defaultRound() {
    return this.rounds.length ? this.rounds[0] : "";
  }

  resolvePlayer(name) {
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

  prepareFeatures({ player1, player2, features, context }) {
    const resolved1 = this.resolvePlayer(player1);
    const resolved2 = this.resolvePlayer(player2);
    if (!resolved1) throw new Error(`Player "${player1}" not found in dataset.`);
    if (!resolved2) throw new Error(`Player "${player2}" not found in dataset.`);
    if (resolved1 === resolved2) throw new Error("Choose two different players.");

    const ctx = {
      surface: context.surface ?? this.defaultSurface(),
      court: context.court ?? this.defaultCourt(),
      round: context.round ?? this.defaultRound(),
      year:
        context.year ??
        this.latestYear ??
        new Date().getFullYear(),
    };

    const numericValues = {};
    const missing = [];
    for (const feature of features) {
      const value = this._deriveFeature(feature, resolved1, resolved2, ctx);
      if (Number.isFinite(value)) {
        numericValues[feature] = value;
      } else {
        numericValues[feature] = 0;
        missing.push(feature);
      }
    }

    if (features.includes("year")) {
      numericValues.year = ctx.year;
    }

    return {
      numericValues,
      missing,
      players: { player1: resolved1, player2: resolved2 },
      context: ctx,
    };
  }

  _build(rows) {
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
        const a = this._finite(entry2?.lastRank);
        const b = this._finite(entry1?.lastRank);
        return this._diff(a, b);
      }
      case "pts_diff": {
        const a = this._finite(entry1?.lastPts);
        const b = this._finite(entry2?.lastPts);
        return this._diff(a, b);
      }
      case "odd_diff": {
        const a = this._finite(entry2?.lastOdd);
        const b = this._finite(entry1?.lastOdd);
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
        const num = Number(row[alias]);
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
}
