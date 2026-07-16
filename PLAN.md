# Project Plan — LSTM vs GRU Stock Price Prediction

End-to-end PyTorch time-series regression pipeline predicting stock closing prices,
comparing LSTM and GRU architectures via MSE/RMSE. Delivered across six feature
branches, one phase each, reviewed and merged before the next begins.

## Phase 1/6 — Data Engineering & Preprocessing
`feature/data-preprocessing` — **merged**
Fixed look-ahead bias in `src/data_loader.py`: `MinMaxScaler` now fits on the training
split only (`fit_transform`) and transforms the test split (`transform`), instead of
fitting on the full series before splitting.

## Phase 2/6 — Model Architecture Construction
`feature/model-architectures` — **in review**
Added `GRUModel` to `src/models.py` alongside the existing `LSTMModel`, with a matching
constructor signature (`input_size`, `hidden_size`, `num_layers`, `output_size`) so
either model can be swapped into the pipeline interchangeably.

## Phase 3/6 — Training & Optimization Loop
`feature/training-loop`
Refactor `src/train.py`: add a validation split, per-epoch validation-loss tracking,
early stopping, and model checkpoint saving (`torch.save`) into `models/`.

## Phase 4/6 — Evaluation & Metrics
`feature/evaluation-metrics`
Add MSE/RMSE computation on inverse-scaled predictions; run LSTM and GRU through the
same test set for a direct side-by-side comparison.

## Phase 5/6 — Visualization & Reporting
`feature/visual-reports`
Extend plotting to show actual vs. LSTM vs. GRU predictions, saved under `outputs/`,
plus a simple metrics summary table comparing the two models.

## Phase 6/6 — Integration, Documentation & Final Pipeline
`feature/integration-docs`
Update `src/main.py` to orchestrate both models end-to-end through the refactored
pipeline; update `README.md` with setup/usage instructions; prepare for final merge.
