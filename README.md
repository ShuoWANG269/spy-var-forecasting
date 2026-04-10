# SPY Value-at-Risk Forecasting

Forecasting Value-at-Risk (VaR) at quantile levels τ = 1%, 5%, 10% for the daily log returns of SPDR S&P 500 ETF (SPY) using multiple methods.

## Methods

| Category | Model | Description |
|----------|-------|-------------|
| Quantile Regression | Quantile Regression | Linear QR with HAR-RV-J features |
| GARCH | GJR-GARCH-t | Leverage term + Student-t distribution |
| Deep Learning | MLP | Chronopoulos et al. (2024, JFE) Deep Quantile Estimator |
| Deep Learning | LSTM | Buczynski & Chlebus (2023) GARCHNet architecture |

All models use a 500-day rolling window, pinball loss for training, and violation rate for evaluation.

## Directory Structure

```
v2/
├── README.md              # This file
├── report.md              # Comprehensive report (with figure references)
├── run_all.py             # Unified runner script (--group control)
├── plot_results.py        # Visualization plotting script
├── models/                # Model implementations
│   ├── garch.py           #   GARCH family
│   ├── historical.py      #   Historical simulation
│   ├── lstm.py            #   LSTM quantile regression
│   ├── nn.py              #   MLP quantile regression
│   ├── parametric.py      #   Parametric method
│   └── quantile_reg.py    #   Linear quantile regression
├── utils/                 # Utility functions
│   ├── data.py            #   Data loading and feature engineering
│   ├── backtest.py        #   Backtesting evaluation
│   └── training.py        #   Shared training framework (pinball loss + early stopping)
├── tests/                 # Test suite (103 tests)
├── results/               # Experiment outputs
│   ├── *.csv              #   VaR forecast results
│   ├── *_meta.json        #   Model metadata
│   ├── *_search_log.csv   #   Hyperparameter search logs
│   ├── checkpoints/       #   Model weights
│   └── *.png              #   Visualization charts
└── spy_data.csv -> ../spy_data.csv  # Data (symlink)
```

## Environment Setup

**Prerequisites**: Python >= 3.11. [uv](https://docs.astral.sh/uv/) is recommended.

```bash
# 1. Clone the repository
git clone <repo-url> && cd <repo-dir>

# 2. Create virtual environment and install dependencies
uv venv .venv
uv pip install -e ".[dev]"

# 3. Verify the environment
.venv/bin/python -c "import torch, arch, statsmodels; print('OK')"
```

If not using uv, pip works as well:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Reproduction Steps

### Step 1: Run Tests (~2 minutes)

```bash
.venv/bin/pytest v2/tests -v
```

Expected output: `103 passed`.

### Step 2: Run All Models (~100 minutes on CPU)

```bash
# Run everything (executes sequentially: traditional → garch500 → mlp → lstm)
.venv/bin/python v2/run_all.py

# Or run by group:
.venv/bin/python v2/run_all.py --group traditional   # ~1 min
.venv/bin/python v2/run_all.py --group garch500      # ~5 min
.venv/bin/python v2/run_all.py --group mlp           # ~30 min
.venv/bin/python v2/run_all.py --group lstm          # ~60 min
```

Outputs are written to `v2/results/` (CSV forecast results + JSON metadata + model checkpoints).

> **Note**: Hyperparameter search is not included in the default run (too time-consuming). To reproduce the search process:
> ```bash
> .venv/bin/python v2/run_all.py --group search       # ~several hours
> ```

### Step 3: Generate Visualizations (~10 seconds)

```bash
.venv/bin/python v2/plot_results.py
```

Generates 5 PNG files in `v2/results/`:

| Chart | File | Report Section |
|-------|------|----------------|
| EDA time series + distribution | `eda_returns.png` | §2 |
| Violation rate bar chart | `violation_rates.png` | §5.1 |
| VaR time series facet plot | `var_timeseries_tau0.05.png` | §5.2 |
| VaR overlay comparison | `var_overlay_tau0.01.png` | §5.2 |
| Quantile crossing | `quantile_crossing.png` | §5.3 |

### Step 4: View the Report

The report is in Markdown format with figures referenced via relative paths:

```bash
# Read directly
cat v2/report.md

# Or open with a Markdown preview tool (e.g., VS Code, Obsidian)
```

## Full Reproduction (One-liner)

```bash
# Complete reproduction from scratch
uv venv .venv && uv pip install -e ".[dev]" \
  && .venv/bin/pytest v2/tests -v \
  && .venv/bin/python v2/run_all.py \
  && .venv/bin/python v2/plot_results.py
```

## Key Results

| Model | τ=1% | τ=5% | τ=10% |
|-------|------|------|-------|
| GJR-GARCH-t | 1.40% | 6.28% | 10.92% |
| Quantile Regression | 1.82% | 5.10% | 9.57% |
| MLP | 1.64% | 4.42% | 8.45% |
| **LSTM** | **1.09%** | **5.17%** | **9.18%** |
| Theoretical | 1.0% | 5.0% | 10.0% |

See `report.md` for detailed analysis.
