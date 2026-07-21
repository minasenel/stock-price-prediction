# StockPriceML — LSTM vs GRU Stock Price Prediction

A PyTorch time-series regression pipeline that predicts stock closing prices and
compares two recurrent architectures — LSTM and GRU — head-to-head on the same
data. Given a sliding window of the last 60 daily closing prices, each model
predicts the next day's close; the two are then evaluated on identical held-out
test data and compared by MSE/RMSE, with results plotted and written to a
metrics report.

## How it works

1. **Data** — daily closing prices for a ticker (currently `AMZN`) are pulled
   via `yfinance`, chronologically split into train / validation / test, and
   scaled with a `MinMaxScaler` fit only on the training split (no look-ahead
   bias).
2. **Sequencing** — each split is windowed into (60-day sequence → next-day
   price) pairs for sequence-to-one prediction.
3. **Training** — an `LSTMModel` and a `GRUModel` (matching hidden size /
   layer count) are trained independently with early stopping on validation
   loss; the best checkpoint of each is saved to `models/`.
4. **Evaluation** — both models run on the same test set; predictions are
   inverse-scaled back to price space and scored with MSE/RMSE.
5. **Reporting** — a comparison plot (actual vs. LSTM vs. GRU) is saved to
   `outputs/prediction.png`, and a markdown metrics table is saved to
   `outputs/metrics.md`.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` currently pins no versions — if you hit a dependency
conflict, install into a fresh virtual environment.

## Usage

```bash
cd src
python main.py
```

Run from inside `src/` — output paths (`../outputs/prediction.png`,
`../outputs/metrics.md`, `../models/best_*.pth`) are relative to that working
directory, so running `python src/main.py` from the repo root will write to
the wrong location.

## Project structure

- `src/data_loader.py` — data download, scaling/splitting, and the
  `StockDataset` / `DataLoader` setup.
- `src/models.py` — `LSTMModel` and `GRUModel` definitions.
- `src/train.py` — training loop with validation tracking, early stopping,
  and checkpointing, plus test-set evaluation.
- `src/utils.py` — runs both models through the same test set and computes
  comparison metrics.
- `src/visualize.py` — comparison plot and metrics report generation.
- `src/main.py` — orchestrates the full pipeline end-to-end.

## Known limitations

- Single-feature (Close price only) input — no volume, technical indicators,
  or other features.
- No automated tests, linting, or CI configured.
- Dependency versions in `requirements.txt` are unpinned.
