"""Data preparation & feature engineering pipeline for checkpoint 1.

This script ingests WTA/ATP match data, cleans and standardises fields,
engineers static and dynamic features, and builds fixed-length GRU-ready
sequences of past matches for each player.

Usage:
    python data_pipeline/checkpoint1_pipeline.py \
        --wta-path wta_data.csv \
        --atp-path data/raw/atp.csv  # optional \
        --output-dir data/processed \
        --sequence-length 20

Outputs inside the chosen output directory:
    cleaned_data.csv            -> cleaned & normalised match-level data
    feature_engineered_data.csv -> dataset with static + dynamic features
    gru_sequences.npz           -> numpy archive with sequences + masks
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    sequence_length: int = 20
    fatigue_window_days: int = 14
    surface_trend_matches: int = 5
    surface_familiarity_days: int = 30
    recent_window: int = 10


@dataclass
class SequenceArtifacts:
    X: np.ndarray
    mask: np.ndarray
    match_ids: np.ndarray
    feature_names: List[str]


ROUND_ORDER = [
    "Qualification", "1st Round", "2nd Round", "3rd Round", "4th Round",
    "Quarterfinals", "Semifinals", "Final"
]
ROUND_INDEX = {r: i for i, r in enumerate(ROUND_ORDER)}
SURFACE_MAP = {
    "hard": "hard",
    "hardcourt": "hard",
    "carpet": "hard",
    "indoor hard": "hard",
    "clay": "clay",
    "red clay": "clay",
    "grass": "grass",
}


def _normalise_name(name: str) -> str:
    if pd.isna(name):
        return ""
    clean = re.sub(r"\s+", " ", str(name).strip())
    return clean.title()


def _standardise_surface(surface: str) -> str:
    if pd.isna(surface):
        return "other"
    s = str(surface).strip().lower()
    for key, value in SURFACE_MAP.items():
        if key in s:
            return value
    return s if s in {"hard", "clay", "grass"} else "other"


def _round_index(round_name: str) -> int:
    if pd.isna(round_name):
        return -1
    name = str(round_name).strip()
    return ROUND_INDEX.get(name, -1)


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], errors="coerce")


def load_raw_data(wta_path: Path, atp_path: Path | None = None) -> pd.DataFrame:
    frames = []
    if wta_path and wta_path.exists():
        frames.append(pd.read_csv(wta_path).assign(tour="WTA"))
    if atp_path and atp_path.exists():
        frames.append(pd.read_csv(atp_path).assign(tour="ATP"))

    if not frames:
        raise FileNotFoundError("No input CSV files were found.")

    return pd.concat(frames, ignore_index=True)


def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Tournament": "tournament",
        "Date": "match_date",
        "Court": "court",
        "Surface": "surface",
        "Round": "round",
        "Player_1": "player_1",
        "Player_2": "player_2",
        "Winner": "winner",
        "Rank_1": "rank_1",
        "Rank_2": "rank_2",
        "Pts_1": "pts_1",
        "Pts_2": "pts_2",
        "Odd_1": "odd_1",
        "Odd_2": "odd_2",
    }
    df = df.rename(columns=rename_map)

    for col in ["player_1", "player_2", "winner"]:
        if col in df.columns:
            df[col] = df[col].apply(_normalise_name)

    if "surface" in df.columns:
        df["surface"] = df["surface"].apply(_standardise_surface)

    if "round" in df.columns:
        df["round_order"] = df["round"].apply(_round_index)

    if "match_date" in df.columns:
        df["match_date"] = _ensure_datetime(df, "match_date")

    required = ["player_1", "player_2", "winner", "match_date"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df.dropna(subset=required)
    df = df[df["player_1"] != df["player_2"]]

    subset_cols = [c for c in ["tournament", "match_date", "player_1", "player_2"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first")

    df["y"] = (df["winner"] == df["player_1"]).astype(int)
    df = df.sort_values("match_date").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))

    return df


def _win_rate(history: List[Dict], limit: int | None = None) -> float:
    if not history:
        return np.nan
    slice_hist = history[-limit:] if limit else history
    return float(np.mean([h["won"] for h in slice_hist]))


def _streak(history: List[Dict], target: int) -> int:
    streak = 0
    for h in reversed(history):
        if h["won"] == target:
            streak += 1
        else:
            break
    return streak


def _surface_win_rate(history: List[Dict], surface: str, limit: int | None = None) -> float:
    matches = [h for h in history if h["surface"] == surface]
    if limit:
        matches = matches[-limit:]
    if not matches:
        return np.nan
    return float(np.mean([m["won"] for m in matches]))


def _surface_count_recent(history: List[Dict], surface: str, current_date: pd.Timestamp, days: int) -> int:
    window_start = current_date - pd.Timedelta(days=days)
    return sum(1 for h in history if h["surface"] == surface and h["date"] >= window_start)


def _fatigue_index(history: List[Dict], current_date: pd.Timestamp, window_days: int) -> int:
    window_start = current_date - pd.Timedelta(days=window_days)
    return sum(1 for h in history if h["date"] >= window_start)


def _tournament_experience(history: List[Dict], tournament: str) -> int:
    return sum(1 for h in history if h.get("tournament") == tournament)


def _h2h_balance(h2h_store: Dict[frozenset, Dict[str, int]], p1: str, p2: str) -> int:
    key = frozenset({p1, p2})
    record = h2h_store.get(key, {})
    return record.get(p1, 0) - record.get(p2, 0)


def _update_h2h(h2h_store: Dict[frozenset, Dict[str, int]], p1: str, p2: str, winner: str) -> None:
    key = frozenset({p1, p2})
    if key not in h2h_store:
        h2h_store[key] = defaultdict(int)
    h2h_store[key][winner] += 1


def _surface_one_hot(surface: str) -> List[int]:
    options = ["hard", "clay", "grass", "other"]
    surface = surface if surface in options else "other"
    return [1 if surface == opt else 0 for opt in options]


def build_dynamic_features(df: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, SequenceArtifacts]:
    df = df.copy().sort_values("match_date").reset_index(drop=True)

    history: Dict[str, List[Dict]] = defaultdict(list)
    h2h_store: Dict[frozenset, Dict[str, int]] = {}
    seq_history: Dict[str, List[Dict]] = defaultdict(list)

    feature_rows = []
    seq_tensors = []
    seq_masks = []

    seq_feature_template = [
        "won", "rank", "opp_rank", "rank_diff", "round_index",
        "surface_hard", "surface_clay", "surface_grass", "surface_other",
        "days_since_last",
    ]
    seq_feature_names = [f"p1_{f}" for f in seq_feature_template] + [f"p2_{f}" for f in seq_feature_template]

    for _, row in df.iterrows():
        p1, p2 = row["player_1"], row["player_2"]
        surface = row.get("surface", "other")
        date = row.get("match_date")
        tournament = row.get("tournament", "")
        round_idx = row.get("round_order", -1)

        p1_history = history[p1]
        p2_history = history[p2]

        def player_features(player_history: List[Dict]) -> Dict[str, float]:
            last_date = player_history[-1]["date"] if player_history else pd.NaT
            days_since = (date - last_date).days if pd.notnull(last_date) else np.nan
            return {
                "recent_win_rate_10": _win_rate(player_history, config.recent_window),
                "days_since_last_match": days_since,
                "fatigue_index": _fatigue_index(player_history, date, config.fatigue_window_days),
                "win_streak": _streak(player_history, 1),
                "loss_streak": _streak(player_history, 0),
                "surface_trend": _surface_win_rate(player_history, surface, config.surface_trend_matches),
                "surface_familiarity": _surface_count_recent(player_history, surface, date, config.surface_familiarity_days),
                "surface_win_rate": _surface_win_rate(player_history, surface),
                "total_win_rate": _win_rate(player_history),
                "tournament_experience": _tournament_experience(player_history, tournament),
            }

        p1_feat = player_features(p1_history)
        p2_feat = player_features(p2_history)

        h2h_balance = _h2h_balance(h2h_store, p1, p2)

        feature_rows.append({
            "match_id": row["match_id"],
            "player_1": p1,
            "player_2": p2,
            "match_date": date,
            "surface": surface,
            "round_order": round_idx,
            "y": row["y"],
            "h2h_balance": h2h_balance,
            **{f"p1_{k}": v for k, v in p1_feat.items()},
            **{f"p2_{k}": v for k, v in p2_feat.items()},
        })

        # sequences built from previous matches only
        def make_seq(player_key: str, player_history_vectors: Dict[str, List[Dict]]):
            history_vectors = player_history_vectors[player_key]
            vectors = [h["vector"] for h in history_vectors][-config.sequence_length:]
            mask = [1.0] * len(vectors)
            if len(vectors) < config.sequence_length:
                pad_len = config.sequence_length - len(vectors)
                vectors = [np.zeros(len(seq_feature_template), dtype=np.float32)] * pad_len + vectors
                mask = [0.0] * pad_len + mask
            return np.stack(vectors, axis=0), np.array(mask, dtype=np.float32)

        p1_seq, p1_mask = make_seq(p1, seq_history)
        p2_seq, p2_mask = make_seq(p2, seq_history)
        combined_seq = np.concatenate([p1_seq, p2_seq], axis=1)
        combined_mask = np.minimum(p1_mask, p2_mask)
        seq_tensors.append(combined_seq.astype(np.float32))
        seq_masks.append(combined_mask.astype(np.float32))

        # update histories with current match outcome
        winner = row["winner"]
        p1_rank = row.get("rank_1") if "rank_1" in row else np.nan
        p2_rank = row.get("rank_2") if "rank_2" in row else np.nan
        rank_diff = (p1_rank - p2_rank) if not (pd.isna(p1_rank) or pd.isna(p2_rank)) else np.nan

        def append_history(player: str, won: int, rank: float, opp_rank: float, player_history: Dict[str, List[Dict]], days_since_last: float):
            vector = [
                float(won),
                float(rank) if not pd.isna(rank) else np.nan,
                float(opp_rank) if not pd.isna(opp_rank) else np.nan,
                float(rank_diff) if not pd.isna(rank_diff) else np.nan,
                float(round_idx),
                *_surface_one_hot(surface),
                float(days_since_last),
            ]
            player_history[player].append({
                "date": date,
                "surface": surface,
                "won": won,
                "tournament": tournament,
                "vector": np.array(vector, dtype=np.float32),
            })

        p1_won = 1 if winner == p1 else 0
        p2_won = 1 if winner == p2 else 0
        append_history(p1, p1_won, p1_rank, p2_rank, seq_history, p1_feat["days_since_last_match"])
        append_history(p2, p2_won, p2_rank, p1_rank, seq_history, p2_feat["days_since_last_match"])

        history[p1].append({"date": date, "surface": surface, "won": p1_won, "tournament": tournament})
        history[p2].append({"date": date, "surface": surface, "won": p2_won, "tournament": tournament})
        _update_h2h(h2h_store, p1, p2, winner)

    feature_df = pd.DataFrame(feature_rows)
    sequences = SequenceArtifacts(
        X=np.stack(seq_tensors, axis=0),
        mask=np.stack(seq_masks, axis=0),
        match_ids=df["match_id"].to_numpy(),
        feature_names=seq_feature_names,
    )
    return feature_df, sequences


def engineer_static_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"rank_1", "rank_2"}.issubset(result.columns):
        result["rank_diff"] = result["rank_1"] - result["rank_2"]
    if {"pts_1", "pts_2"}.issubset(result.columns):
        result["pts_diff"] = result["pts_1"] - result["pts_2"]
    if {"odd_1", "odd_2"}.issubset(result.columns):
        result["odd_diff"] = result["odd_1"] - result["odd_2"]
    # placeholders for demographics/ratings if user provides metadata later
    for col in ["player_1_age", "player_2_age", "player_1_height", "player_2_height", "player_1_country", "player_2_country"]:
        if col not in result.columns:
            result[col] = np.nan
    return result


def save_sequences(output_dir: Path, artifacts: SequenceArtifacts) -> Path:
    seq_path = output_dir / "gru_sequences.npz"
    np.savez(
        seq_path,
        X=artifacts.X,
        mask=artifacts.mask,
        match_ids=artifacts.match_ids,
        feature_names=np.array(artifacts.feature_names),
    )
    return seq_path


def run_pipeline(wta_path: Path, atp_path: Path | None, output_dir: Path, config: PipelineConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(wta_path, atp_path)
    cleaned = clean_and_standardise(raw_df)
    static_df = engineer_static_features(cleaned)
    dynamic_df, sequences = build_dynamic_features(static_df, config)

    cleaned_path = output_dir / "cleaned_data.csv"
    engineered_path = output_dir / "feature_engineered_data.csv"

    cleaned.to_csv(cleaned_path, index=False)
    dynamic_df.to_csv(engineered_path, index=False)
    seq_path = save_sequences(output_dir, sequences)

    summary = {
        "input_rows": len(raw_df),
        "cleaned_rows": len(cleaned),
        "engineered_rows": len(dynamic_df),
        "sequence_shape": list(sequences.X.shape),
        "sequence_feature_names": sequences.feature_names,
    }
    with open(output_dir / "checkpoint1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Cleaned data saved to {cleaned_path}")
    print(f"Feature engineered data saved to {engineered_path}")
    print(f"GRU sequences saved to {seq_path}")
    print("Summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint 1 data preparation")
    parser.add_argument("--wta-path", type=Path, default=Path("wta_data.csv"), help="Path to WTA CSV")
    parser.add_argument("--atp-path", type=Path, default=None, help="Optional path to ATP CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--fatigue-window-days", type=int, default=14)
    parser.add_argument("--surface-trend-matches", type=int, default=5)
    parser.add_argument("--surface-familiarity-days", type=int, default=30)
    parser.add_argument("--recent-window", type=int, default=10)
    args = parser.parse_args()

    config = PipelineConfig(
        sequence_length=args.sequence_length,
        fatigue_window_days=args.fatigue_window_days,
        surface_trend_matches=args.surface_trend_matches,
        surface_familiarity_days=args.surface_familiarity_days,
        recent_window=args.recent_window,
    )

    run_pipeline(args.wta_path, args.atp_path, args.output_dir, config)
