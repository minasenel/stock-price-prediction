# CLAUDE.md

This file provides development guidance, project constraints, and systemic rules to Claude Code (claude.ai/code) when working in this repository.

## Project Overview
StockPriceML is a deep learning-based time-series regression project designed to predict stock closing prices using PyTorch[cite: 1, 2]. The system implements a single-feature (Close price only), sliding-window sequence-to-one prediction strategy[cite: 2].

*Note: While the project's conceptual documentation references both LSTM and GRU models benchmarked on Amazon (AMZN) data, the current codebase implements the LSTM model only, and `main.py` is hardcoded to the `AAPL` ticker.*

## 1. Code Modification & Approval Policy
- **Strict Approval Required:** Do not modify, create, or delete any code files (`.py`, `.ipynb`, `.json`, etc.) without explicitly asking for permission, explaining the changes, and receiving direct approval from the user.
- **Plan Before Execution:** Always present a structured plan (`plan`) before writing any code. Clearly detail which files and lines will be modified, and what the logical rationale is. Never proceed to implementation until the plan is approved.

## 2. Token & Budget Economy
- **Minimal Code Output:** Focus directly on the exact user query. Avoid providing bloated code snippets, large boilerplate blocks, or unrequested features. Keep examples isolated and concise[cite: 1].
- **Targeted File Reading:** Minimize token consumption by avoiding full-repository scans. Only inspect the files directly relevant to the task at hand (e.g., targeting only the dataset processing script, specific model architectures, or training loops).

## 3. Project-Specific Technical Constraints (PyTorch & Time Series)
- **Modular Code Architecture:** Separate data workflows, neural architectures, and training execution into clean, individual modules (`preprocessing`, `models`, `training`/`evaluation`). Do not group the entire pipeline into a single macro-script.
- **Look-Ahead Bias Prevention:** When implementing data normalization (e.g., `MinMaxScaler`), the scaler must be fit **only** on the training split (`fit_transform`) and subsequently used to normalize the test split (`transform`)[cite: 2]. Never fit the scaler on the entire combined dataset[cite: 2].
  > ⚠️ **Current Discrepancy:** `src/data_loader.py`'s `prepare_data()` currently runs `fit_transform` on the entire dataset before splitting[cite: 2]. This look-ahead bias must be fixed via an approved feature branch plan.
- **Overfitting Management:** Always introduce validation tracking and Early Stopping logic into training execution pipelines to handle convergence dynamically instead of running high, fixed epoch cycles[cite: 2].
  > ⚠️ **Current Discrepancy:** `src/train.py`'s `train_model()` currently runs for a fixed number of epochs without validation loss monitoring or an early stopping mechanism[cite: 2].

## 4. Workflow & Branch Strategy
- Break down complex assignments into atomic, logical milestones. Move forward progressively by executing and committing changes one branch step at a time (Data Engineering, Architecture Construction, Optimization Loop, Assessment, and Visual Reports).

## Execution Commands
No testing harness, linting configuration, or CI workflow is currently initialized in this environment (verified: no `tests/` directory, no `pytest`/`ruff`/`flake8` rules, no active GitHub workflows)[cite: 2].

- **Dependency Installation:** `pip install -r requirements.txt` (Note: dependency versions are unpinned in this file)[cite: 2]
- **Pipeline Execution:** `cd src && python main.py`
  *(Note: Must be executed with `src/` as the current working directory, as relative charting paths map to `../outputs/prediction.png`[cite: 2])*

## Codebase Architecture
The system workflow is orchestrated end-to-end by `src/main.py` (`main()`, no CLI argument parsing)[cite: 2]. All hyperparameter attributes are stored as local hardcoded constants: ticker selection, chronological boundaries, `seq_length=60`, `batch_size=32`, `hidden_size=64`, `num_layers=2`, `lr=0.001`, `epochs=20`[cite: 2].

1. **`src/data_loader.py`**
   - `load_stock_data(...)`: Direct download interface wrapper wrapping `yfinance`[cite: 2].
   - `prepare_data(...)`: Extracts the single `Close` vector, scales via a `MinMaxScaler`, applies a chronological split, and encapsulates arrays into `StockDataset` objects. Returns `(train_dataset, test_dataset, scaler)`[cite: 2].
   - `StockDataset(Dataset)`: Custom PyTorch dataset structure mapping sliding sequences of width `seq_length` (`x`) to their subsequent next-step targets (`y`)[cite: 2].
   - `get_dataloaders(...)`: Packs datasets into streamable `DataLoader` instances (train split is shuffled, test split is sequential)[cite: 2].
2. **`src/models.py`**
   - `LSTMModel`: Stacked `nn.LSTM` module configured with `batch_first=True` feeding an independent `nn.Linear` dense regression head using the hidden state vector of the final temporal step[cite: 2]. No GRU equivalent exists yet[cite: 2].
3. **`src/train.py`**
   - `train_model(...)`: Standard iterative PyTorch training loop tracking and printing macro epoch loss values[cite: 2]. Model serialization saving states are missing[cite: 2].
   - `evaluate_model(...)`: Execution tracking under `eval()` and `no_grad()` contexts returning inference and true vector pairs as structured NumPy elements[cite: 2].
4. **`src/main.py`**
   - Structural orchestration layer: runtime hardware auto-selection (`cuda` → `mps` → `cpu`), dataset ingestion, model construction, learning parameters binding (`MSELoss` + `Adam`), model optimization, testing evaluation, inverse feature rescaling via `scaler`, and data visualization plotting to `../outputs/prediction.png`[cite: 2].

*Note: Helper modules `src/utils.py` and `src/__init__.py` exist as empty modules reserved for future helper implementations[cite: 2].*

## Known Gaps & Tech Debt
- Automated tests, code format verification linter rules, and automation pipeline CI systems are missing[cite: 2].
- Look-Ahead Bias risk is active inside `prepare_data()` due to an early full-dataset scaling calculation[cite: 2].
- The training controller lacks verification checkpoints, tracking diagnostics, and Early Stopping layers[cite: 2].
- No automated weight storage mechanism (`torch.save`) is written into the execution, leaving the root `models/` folder blank[cite: 2].
- Redundant and duplicate virtual environments exist in the active project directory (`.venv/` and `venv/`)[cite: 2].
- The global root `README.md` file is a placeholder header lacking environment instructions or architectural summaries[cite: 2].