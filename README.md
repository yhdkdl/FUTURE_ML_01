# FUTURE_ML_01 — Sales Forecasting System

A production-ready weekly sales forecasting pipeline built with Python
and Scikit-learn. Trained on 4 years of Superstore retail data to
predict future revenue and support business planning decisions.

---

## Project Structure
```
FUTURE_ML_01/
├── data/
│   ├── raw/                   ← original dataset (not committed)
│   └── processed/
│       ├── weekly_sales.csv   ← cleaned weekly aggregation
│       ├── features.csv       ← engineered feature set
│       └── forecast.csv       ← 12-week future forecast
├── src/
│   ├── data/
│   │   ├── loader.py          ← CSV ingestion
│   │   ├── cleaner.py         ← cleaning + weekly aggregation
│   │   └── eda.py             ← exploratory analysis
│   ├── features/
│   │   └── engineer.py        ← time-based feature engineering
│   ├── models/
│   │   ├── trainer.py         ← train/test split + model training
│   │   ├── evaluator.py       ← metrics + evaluation report
│   │   └── predictor.py       ← recursive future forecast
│   └── visualization/
│       └── charts.py          ← 4 business-friendly charts
├── outputs/
│   ├── models/
│   │   └── forecast_model.joblib
│   └── charts/
│       ├── 01_sales_history.png
│       ├── 02_forecast.png
│       ├── 03_seasonality.png
│       └── 04_model_performance.png
├── notebooks/                 ← exploratory notebooks (optional)
├── main.py                    ← single pipeline entry point
├── requirements.txt           ← unpinned dependencies
├── requirements_lock.txt      ← exact pinned versions
└── FORECAST_SUMMARY.md        ← business-facing report
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/FUTURE_ML_01.git
cd FUTURE_ML_01
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements_lock.txt
```

### 4. Add the dataset
Download the Superstore dataset from Kaggle and place it at:
```
data/raw/superstore.csv
```

### 5. Run the full pipeline
```bash
python main.py
```

---

## Pipeline Stages

| Stage | Module | What It Does |
|-------|--------|-------------|
| 1 | `loader.py` | Reads raw CSV into DataFrame |
| 2 | `cleaner.py` | Cleans data, aggregates to weekly sales |
| 3 | `eda.py` | Prints exploratory analysis to terminal |
| 4 | `engineer.py` | Builds lag, rolling, and calendar features |
| 5 | `trainer.py` | Seasonally-aware train/test split |
| 6 | `trainer.py` | Trains Random Forest Regressor |
| 7 | `evaluator.py` | MAE, RMSE, MAPE, R² evaluation |
| 8 | `predictor.py` | Recursive 12-week forecast |
| 9 | `charts.py` | Saves 4 business-friendly PNG charts |

---

## Model

**Algorithm:** Random Forest Regressor (Scikit-learn)

**Features used:**
- Calendar: year index, month, quarter, week of year
- Flags: is Q4, is quarter end, is back-to-school season
- Lag: last 1, 2, 4, and 52 weeks of sales
- Rolling: 4-week mean, 12-week mean, 4-week std deviation

**Evaluation (on held-out test weeks spanning all 4 years):**

| Metric | Value |
|--------|-------|
| MAE    | ~$5,385 |
| RMSE   | ~$7,539 |
| MAPE   | ~33% |
| R²     | ~0.46 |

---

## Output Charts

| Chart | Description |
|-------|-------------|
| `01_sales_history.png` | 4-year weekly sales with 8-week trend overlay |
| `02_forecast.png` | 12-week forecast with confidence band |
| `03_seasonality.png` | Monthly and quarterly sales breakdown |
| `04_model_performance.png` | Actual vs predicted + metric summary |

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production only — final merged result |
| `develop` | Integration branch |
| `feature/sprint-x` | Active development work |
| `hotfix/...` | Bug fixes merged back to develop |

---

## Dataset

Superstore Sales Dataset — Kaggle
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.14 | Core language |
| pandas | Data manipulation |
| scikit-learn | ML model |
| matplotlib | Charting |
| seaborn | Plot styling |
| joblib | Model persistence |