# ML Assignment Project

## Setup (macOS, Apple Silicon)

### 1. Install Python 3.9+ (recommended: 3.10)

### 2. Create virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

```
pip install -r requirements.txt
```

## Run Smoke Test

```
python -m src.utils.smoke_test
```

## Run All Checks

```
pytest -q
```

## Project Structure

- src/utils/ : Core utilities (seed, logger, device, metrics, io, plots, smoke_test)
- configs/ : Config files
- outputs/ : Results (plots, tables, checkpoints)
- report/ : Reports
- scripts/ : Helper scripts
- tests/ : Pytest-based tests

## Acceptance Checklist

- `python -m src.utils.smoke_test` creates:
  - outputs/tables/smoke_metrics.csv
  - outputs/plots/smoke_plot.png
- `pytest -q` passes all tests
- metrics.py computes per-class + micro/macro metrics deterministically

---

## VERIFICATION

### Commands to Run

```
python -m src.utils.smoke_test
pytest -q
```

### Expected Outputs

- outputs/tables/smoke_metrics.csv (CSV with columns: class, precision, recall, f1, support)
- outputs/plots/smoke_plot.png (non-empty PNG file)
- All tests in tests/ pass (no errors)

### PASS/FAIL Criteria

- **PASS**: Both files above are created, contain valid data, and all tests pass.
- **FAIL**: Any file missing, empty, or any test fails.
