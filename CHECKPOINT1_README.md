# Checkpoint 1 — Data preparation & feature engineering

This repository now contains a reproducible pipeline for the first project checkpoint: cleaning the raw WTA/ATP match data, engineering static and dynamic features, and exporting GRU-ready sequences for the last 12–20 matches per player.

## How to run

```bash
python data_pipeline/checkpoint1_pipeline.py \
  --wta-path wta_data.csv \
  --atp-path /path/to/atp_matches.csv \  # optional
  --output-dir data/processed \
  --sequence-length 20
```

Adjust the `sequence-length` flag to anywhere between 12–20 to match the GRU requirements. The defaults also expose rolling window sizes for fatigue/surface trend calculations.

## Outputs

Running the pipeline writes the following artefacts to `data/processed/`:

- `cleaned_data.csv` — cleaned, de-duplicated, standardised match table (uniform surfaces, ISO dates, normalised player names, target label `y`).
- `feature_engineered_data.csv` — adds static (`rank_diff`, `pts_diff`, placeholders for age/height/country) and dynamic features (recent win rates, streaks, fatigue, surface trend/familiarity, tournament experience, H2H balance).
- `gru_sequences.npz` — numpy archive with tensors shaped `(num_matches, sequence_length, num_features*2)` where the second dimension holds left/right player histories; includes padding masks for matches with < sequence length history.
- `checkpoint1_summary.json` — quick metadata about row counts and sequence shapes.

## Notes & assumptions

- The pipeline is tolerant to ATP data being missing; it will run on WTA-only data if that is all that is available.
- Player demographics (age, height, country) are left as `NaN` placeholders until auxiliary metadata is supplied. The processing functions will automatically keep these columns for future joins.
- Surfaces are normalised to `hard`, `clay`, `grass`, or `other`; rounds are ordinally encoded to maintain ordering for sequence construction.
- Dynamic metrics are computed chronologically to avoid leakage and reflect each player’s form at the time of the match.

## EDA notebook

A lightweight notebook is available at `notebooks/checkpoint1_eda.ipynb`. It demonstrates loading the processed data, inspecting missingness, and computing simple correlations for feature sanity checks.
