#!/usr/bin/env python3
"""Preprocess the WTA historical dataset for the browser demo.

Steps implemented in this script:
1. Download (optional) and load the raw Kaggle dump.
2. Clean columns, normalise names and parse dates/scores.
3. Build player-level rolling aggregates (form, surface form, rest, games).
4. Compute Elo ratings (overall + surface specific).
5. Derive head-to-head features.
6. Assemble a wide dataframe with one row per match containing only
   pre-match information and save it as CSV ready for the JS frontend.

The resulting dataset keeps the original identifier columns so the UI can
later map player names to features automatically.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency for automated download
    import kagglehub  # type: ignore
except Exception:  # pragma: no cover - we don't require kagglehub at runtime
    kagglehub = None


DEFAULT_DATASET = "dissfya/wta-tennis-2007-2023-daily-update"
SCORE_PATTERN = re.compile(r"(\d+)-(\d+)")


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    min_year: int = 2010
    rolling_matches: int = 10
    rolling_surface_matches: int = 6
    rolling_days: int = 365
    elo_k: float = 32.0
    surface_elo_k: float = 24.0
    elo_decay: float = 0.01  # seasonal decay factor applied annually
    output: Path = Path("data/processed_matches.csv")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def download_dataset(dataset: str = DEFAULT_DATASET) -> Path:
    """Download the Kaggle dataset and return its local path.

    Raises RuntimeError if kagglehub is unavailable.
    """

    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is not installed. Install it or provide --input pointing "
            "to the extracted dataset."
        )
    return Path(kagglehub.dataset_download(dataset))


def find_matches_csv(root: Path) -> Path:
    """Pick the largest CSV file that looks like the match results table."""

    candidates = list(root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {root}.")

    preferred = [
        p for p in candidates if re.search(r"^wta.*\.csv$", p.name, re.I)
    ] or [p for p in candidates if "match" in p.name.lower()] or candidates
    return max(preferred, key=lambda p: p.stat().st_size)


def load_raw_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Normalise column names for easier processing downstream
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def parse_score(score: str) -> Tuple[int, int, int, int, int]:
    """Return (games_p1, games_p2, sets_p1, sets_p2, tie_breaks).

    Handles walkovers/retirements by returning zeros.
    """

    if not isinstance(score, str):
        return 0, 0, 0, 0, 0
    score = score.strip()
    if not score or "w/o" in score.lower() or "walkover" in score.lower():
        return 0, 0, 0, 0, 0

    games_1 = games_2 = sets_1 = sets_2 = tie_breaks = 0
    for left, right in SCORE_PATTERN.findall(score):
        g1, g2 = int(left), int(right)
        games_1 += g1
        games_2 += g2
        if g1 > g2:
            sets_1 += 1
        elif g2 > g1:
            sets_2 += 1
        if max(g1, g2) >= 7:
            tie_breaks += 1
    return games_1, games_2, sets_1, sets_2, tie_breaks


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    scores = df.get("score")
    parsed = scores.fillna("").map(parse_score)
    df[[
        "p1_games_won",
        "p2_games_won",
        "p1_sets_won",
        "p2_sets_won",
        "tie_breaks_played",
    ]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df["games_delta"] = df["p1_games_won"] - df["p2_games_won"]
    df["sets_delta"] = df["p1_sets_won"] - df["p2_sets_won"]
    return df


def compute_elo(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    surface_ratings: Dict[Tuple[str, str], float] = defaultdict(lambda: 1500.0)
    last_year = None

    elo_p1 = []
    elo_p2 = []
    elo_surface_p1 = []
    elo_surface_p2 = []

    for row in df.itertuples(index=False):
        year = row.date.year
        if last_year is not None and year > last_year:
            decay = math.exp(-config.elo_decay * (year - last_year))
            for player in list(ratings.keys()):
                ratings[player] = 1500.0 + (ratings[player] - 1500.0) * decay
            for key in list(surface_ratings.keys()):
                surface_ratings[key] = 1500.0 + (surface_ratings[key] - 1500.0) * decay
        last_year = year

        p1, p2 = row.player_1, row.player_2
        surface = getattr(row, "surface", "Unknown")
        r1, r2 = ratings[p1], ratings[p2]
        s1, s2 = surface_ratings[(p1, surface)], surface_ratings[(p2, surface)]

        elo_p1.append(r1)
        elo_p2.append(r2)
        elo_surface_p1.append(s1)
        elo_surface_p2.append(s2)

        expected = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
        outcome = 1.0 if row.winner == row.player_1 else 0.0
        ratings[p1] = r1 + config.elo_k * (outcome - expected)
        ratings[p2] = r2 + config.elo_k * ((1.0 - outcome) - (1.0 - expected))

        expected_surface = 1.0 / (1.0 + 10 ** ((s2 - s1) / 400))
        surface_ratings[(p1, surface)] = s1 + config.surface_elo_k * (outcome - expected_surface)
        surface_ratings[(p2, surface)] = s2 + config.surface_elo_k * ((1.0 - outcome) - (1.0 - expected_surface))

    df = df.copy()
    df["elo_p1"] = elo_p1
    df["elo_p2"] = elo_p2
    df["surface_elo_p1"] = elo_surface_p1
    df["surface_elo_p2"] = elo_surface_p2
    df["elo_diff"] = df["elo_p1"] - df["elo_p2"]
    df["surface_elo_diff"] = df["surface_elo_p1"] - df["surface_elo_p2"]
    return df


def expand_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long dataframe with one row per player per match."""

    base_cols = [
        "match_id",
        "date",
        "surface",
        "round",
        "tournament",
        "court",
        "best_of",
        "tie_breaks_played",
        "games_delta",
        "sets_delta",
    ]
    p1_cols = {
        "player": "player_1",
        "opponent": "player_2",
        "player_rank": "rank_1",
        "opponent_rank": "rank_2",
        "player_pts": "pts_1",
        "opponent_pts": "pts_2",
        "player_odds": "odd_1",
        "opponent_odds": "odd_2",
        "games_for": "p1_games_won",
        "games_against": "p2_games_won",
        "sets_for": "p1_sets_won",
        "sets_against": "p2_sets_won",
        "elo_pre": "elo_p1",
        "elo_surface_pre": "surface_elo_p1",
    }
    p2_cols = {
        "player": "player_2",
        "opponent": "player_1",
        "player_rank": "rank_2",
        "opponent_rank": "rank_1",
        "player_pts": "pts_2",
        "opponent_pts": "pts_1",
        "player_odds": "odd_2",
        "opponent_odds": "odd_1",
        "games_for": "p2_games_won",
        "games_against": "p1_games_won",
        "sets_for": "p2_sets_won",
        "sets_against": "p1_sets_won",
        "elo_pre": "elo_p2",
        "elo_surface_pre": "surface_elo_p2",
    }

    common_cols = base_cols + list(p1_cols.values()) + ["winner"]
    missing = [c for c in common_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required for long expansion: {missing}")

    first = df[base_cols + list(p1_cols.values()) + ["winner"]].copy()
    for new_col, src in p1_cols.items():
        first[new_col] = first[src]
    first["is_player_1"] = True
    first["won"] = (df["winner"] == df["player_1"]).astype(int)

    second = df[base_cols + list(p2_cols.values()) + ["winner"]].copy()
    for new_col, src in p2_cols.items():
        second[new_col] = second[src]
    second["is_player_1"] = False
    second["won"] = (df["winner"] == df["player_2"]).astype(int)

    long_df = pd.concat([first, second], ignore_index=True, sort=False)
    long_df["tie_breaks_played"] = long_df["tie_breaks_played"].fillna(0)
    long_df["games_delta_signed"] = np.where(
        long_df["is_player_1"], long_df["games_delta"], -long_df["games_delta"]
    )
    long_df["sets_delta_signed"] = np.where(
        long_df["is_player_1"], long_df["sets_delta"], -long_df["sets_delta"]
    )
    return long_df


def add_rolling_features(long_df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    long_df = long_df.sort_values(["player", "date", "match_id"]).copy()

    def _rolling(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
        return series.shift(1).rolling(window=window, min_periods=min_periods).mean()

    long_df["form_win_rate"] = (
        long_df.groupby("player", group_keys=False)["won"].apply(
            lambda s: _rolling(s, config.rolling_matches, min_periods=3)
        )
    )
    long_df["surface_form_win_rate"] = (
        long_df.groupby(["player", "surface"], group_keys=False)["won"].apply(
            lambda s: _rolling(s, config.rolling_surface_matches, min_periods=2)
        )
    )
    long_df["games_delta_avg"] = (
        long_df.groupby("player", group_keys=False)["games_delta_signed"].apply(
            lambda s: _rolling(s, config.rolling_matches, min_periods=3)
        )
    )
    long_df["sets_delta_avg"] = (
        long_df.groupby("player", group_keys=False)["sets_delta_signed"].apply(
            lambda s: _rolling(s, config.rolling_matches, min_periods=3)
        )
    )
    long_df["tiebreak_rate"] = (
        long_df.groupby("player", group_keys=False)["tie_breaks_played"].apply(
            lambda s: _rolling(s, config.rolling_matches, min_periods=3)
        )
    )
    long_df["rest_days"] = (
        long_df.groupby("player", group_keys=False)["date"].diff().dt.days.shift(0)
    )
    long_df["rest_days"] = long_df["rest_days"].fillna(30).clip(0, 60)

    # rolling rank/points trends (use simple difference to last match)
    long_df["rank_trend"] = (
        long_df.groupby("player", group_keys=False)["player_rank"].apply(
            lambda s: s.shift(1) - s.shift(2)
        )
    )
    long_df["points_trend"] = (
        long_df.groupby("player", group_keys=False)["player_pts"].apply(
            lambda s: s.shift(1) - s.shift(2)
        )
    )

    return long_df


def aggregate_back(df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    p1 = long_df[long_df["is_player_1"]].copy().set_index("match_id")
    p2 = long_df[~long_df["is_player_1"]].copy().set_index("match_id")
    p1 = p1.reindex(df.match_id)
    p2 = p2.reindex(df.match_id)

    selected_cols = [
        "form_win_rate",
        "surface_form_win_rate",
        "games_delta_avg",
        "sets_delta_avg",
        "tiebreak_rate",
        "rest_days",
        "rank_trend",
        "points_trend",
    ]
    suffix_map = {
        "form_win_rate": "form",
        "surface_form_win_rate": "surface_form",
        "games_delta_avg": "games_margin",
        "sets_delta_avg": "sets_margin",
        "tiebreak_rate": "tiebreak_rate",
        "rest_days": "rest_days",
        "rank_trend": "rank_trend",
        "points_trend": "points_trend",
    }

    for col in selected_cols:
        df[f"p1_{suffix_map[col]}"] = p1[col]
        df[f"p2_{suffix_map[col]}"] = p2[col]

    df["form_diff"] = df["p1_form"] - df["p2_form"]
    df["surface_form_diff"] = df["p1_surface_form"] - df["p2_surface_form"]
    df["games_margin_diff"] = df["p1_games_margin"] - df["p2_games_margin"]
    df["sets_margin_diff"] = df["p1_sets_margin"] - df["p2_sets_margin"]
    df["rest_days_diff"] = df["p1_rest_days"] - df["p2_rest_days"]
    df["tiebreak_rate_diff"] = df["p1_tiebreak_rate"] - df["p2_tiebreak_rate"]
    df["rank_trend_diff"] = df["p1_rank_trend"] - df["p2_rank_trend"]
    df["points_trend_diff"] = df["p1_points_trend"] - df["p2_points_trend"]

    return df


def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    h2h_wins: Dict[Tuple[str, str], int] = defaultdict(int)
    last_winner: Dict[Tuple[str, str], int] = {}

    adv_list = []
    last_winner_list = []

    for row in df.itertuples(index=False):
        p1, p2 = row.player_1, row.player_2
        key = (p1, p2)
        rev_key = (p2, p1)

        wins = h2h_wins[key]
        losses = h2h_wins[rev_key]
        total = wins + losses
        adv = (wins - losses) / total if total > 0 else 0.0
        adv_list.append(adv)
        last = last_winner.get(key)
        last_winner_list.append(1 if last == 1 else (-1 if last == -1 else 0))

        if row.winner == p1:
            h2h_wins[key] += 1
            last_winner[key] = 1
            last_winner[rev_key] = -1
        else:
            h2h_wins[rev_key] += 1
            last_winner[key] = -1
            last_winner[rev_key] = 1

    df = df.copy()
    df["h2h_advantage"] = adv_list
    df["last_winner_indicator"] = last_winner_list
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    df = df.rename(
        columns={
            "winner": "winner",
            "player1": "player_1",
            "player2": "player_2",
        }
    )

    required = {"player_1", "player_2", "winner", "date"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing required columns: {sorted(missing)}")

    df = df[df["player_1"].notna() & df["player_2"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df[df["date"].dt.year >= config.min_year]
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    df["y"] = (df["winner"] == df["player_1"]).astype(int)

    for col in ["rank_1", "rank_2", "pts_1", "pts_2", "odd_1", "odd_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_score_features(df)
    df = compute_elo(df, config)
    df = compute_head_to_head(df)

    long_df = expand_long(df)
    long_df = add_rolling_features(long_df, config)
    df = aggregate_back(df, long_df)

    numeric_diff_cols = [
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
        "last_winner_indicator",
    ]

    for col in numeric_diff_cols:
        if col not in df.columns and col.replace("_diff", "_adv") in df.columns:
            continue

    present_numeric = [c for c in numeric_diff_cols if c in df.columns]
    if present_numeric:
        df = df.dropna(subset=present_numeric, how="any")

    if "surface_form_diff" in df.columns and "surface_winrate_adv" not in df.columns:
        df["surface_winrate_adv"] = df["surface_form_diff"].clip(-1, 1)
    if "last_winner_indicator" in df.columns and "last_winner" not in df.columns:
        df["last_winner"] = (df["last_winner_indicator"] > 0).astype(int)

    df["year"] = df["date"].dt.year

    keep_columns = [
        "tournament",
        "date",
        "court",
        "surface",
        "round",
        "best_of",
        "player_1",
        "player_2",
        "winner",
        "score",
        "y",
        "year",
    ] + [c for c in numeric_diff_cols if c in df.columns]

    return df[keep_columns]


def run_pipeline(config: PreprocessConfig, input_path: Optional[Path] = None) -> pd.DataFrame:
    if input_path is None:
        dataset_root = download_dataset()
        input_path = find_matches_csv(dataset_root)
    elif input_path.is_dir():
        input_path = find_matches_csv(input_path)

    df = load_raw_matches(input_path)

    if "rank_diff" not in df.columns and {"rank_1", "rank_2"}.issubset(df.columns):
        df["rank_diff"] = pd.to_numeric(df["rank_1"], errors="coerce") - pd.to_numeric(df["rank_2"], errors="coerce")
    if "pts_diff" not in df.columns and {"pts_1", "pts_2"}.issubset(df.columns):
        df["pts_diff"] = pd.to_numeric(df["pts_1"], errors="coerce") - pd.to_numeric(df["pts_2"], errors="coerce")
    if "odd_diff" not in df.columns and {"odd_1", "odd_2"}.issubset(df.columns):
        df["odd_diff"] = pd.to_numeric(df["odd_1"], errors="coerce") - pd.to_numeric(df["odd_2"], errors="coerce")

    df_processed = preprocess(df, config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(config.output, index=False)
    return df_processed


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess WTA dataset for the JS demo")
    parser.add_argument("--input", type=Path, help="Path to raw CSV or directory with CSV files", default=None)
    parser.add_argument("--output", type=Path, help="Where to write the processed CSV", default=PreprocessConfig.output)
    parser.add_argument("--min-year", type=int, default=PreprocessConfig.min_year)
    parser.add_argument("--rolling-matches", type=int, default=PreprocessConfig.rolling_matches)
    parser.add_argument("--rolling-surface-matches", type=int, default=PreprocessConfig.rolling_surface_matches)
    parser.add_argument("--rolling-days", type=int, default=PreprocessConfig.rolling_days)
    parser.add_argument("--elo-k", type=float, default=PreprocessConfig.elo_k)
    parser.add_argument("--surface-elo-k", type=float, default=PreprocessConfig.surface_elo_k)
    parser.add_argument("--elo-decay", type=float, default=PreprocessConfig.elo_decay)
    return parser.parse_args(args=args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    config = PreprocessConfig(
        min_year=args.min_year,
        rolling_matches=args.rolling_matches,
        rolling_surface_matches=args.rolling_surface_matches,
        rolling_days=args.rolling_days,
        elo_k=args.elo_k,
        surface_elo_k=args.surface_elo_k,
        elo_decay=args.elo_decay,
        output=args.output,
    )
    df = run_pipeline(config, input_path=args.input)
    print(f"Processed dataset with {len(df)} rows saved to {config.output}")


if __name__ == "__main__":
    main()
