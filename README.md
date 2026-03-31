# FUTURE_ML_01 — Sales Forecasting System

A production-ready sales forecasting pipeline built with Python and Scikit-learn.

## Project Structure
```
FUTURE_ML_01/
├── data/
│   ├── raw/          ← original dataset (not committed to git)
│   └── processed/    ← cleaned, feature-engineered data
├── notebooks/        ← exploratory analysis
├── src/
│   ├── data/         ← loading & cleaning pipeline
│   ├── features/     ← time-based feature engineering
│   ├── models/       ← training & evaluation
│   └── visualization/← business-friendly charts
├── outputs/
│   ├── models/       ← saved trained models
│   └── charts/       ← exported forecast images
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
python main.py
```

## Branch Strategy
- `main` — production only
- `develop` — integration branch
- `feature/sprint-x` — active work