# Hydraulic System Condition Monitoring & Diagnostics

A machine learning pipeline to process raw multivariate sensor readouts from hydraulic test rigs, extract time-series statistical features, and train Random Forest diagnostic models across multiple hydraulic components.

Requires the UCI Condition monitoring of hydraulic system data. And then make your own data/ files to ensure the code works as intended. 

## Project Overview

The project operates in two stages:
1. **Feature Extraction & Preprocessing (`preprocess.py`):**
   - Extracts 6 summary statistics (`mean`, `std`, `max`, `median`, `skew`, `kurtosis`) per cycle across 16 hydraulic sensors (`PS1–PS6`, `FS1–FS2`, `TS1–TS4`, `VS1`, `CE`, `CP`, `SE`).
   - Merges sensor features with multi-target labels from `profile.txt`.
   - Scales numeric features with `StandardScaler` and outputs `data/processed/master_data.csv`.

2. **Diagnostic Modeling & Stress Testing (`diagnostics.py`):**
   - Evaluates Random Forest classifiers across 4 targets:
     - `Cooler_Condition`
     - `Valve_Condition`
     - `Pump_Leakage`
     - `Accumulator_Condition`
   - Executes tree convergence tests (`n_estimators` from 1 to 150) and pre-pruning analyses (`max_depth` from 1 to 20).
   - Performs a **Quantile Temperature Stress Test** (train on cycles with temperature $\le$ 80th percentile, evaluate on the hottest top 20%) to assess out-of-distribution robustness with and without temperature features (`TS*`).
   - Exports confusion matrices, importance plots, and metric spreadsheets.

---

## Project Structure

```text
├── data/
│   ├── raw/                   # Raw sensor .txt files & profile.txt
│   └── processed/             # Output folder for master_data.csv
├── results/
│   ├── plots/
│   │   ├── convergence/       # Convergence curves (Accuracy vs. Trees)
│   │   ├── pruning/           # Bias-variance trade-off curves
│   │   ├── importance/        # Aggregated sensor importance charts
│   │   └── matrices/          # Confusion matrix heatmaps
│   └── tables/                # Exported .xlsx / .csv performance summaries
├── preprocess.py              # Script 1: Feature extraction & scaling
├── diagnostics.py             # Script 2: Modeling, validation & stress testing
├── .gitignore
└── README.md
