# WC 2026 Match Predictor

A machine learning project that predicts 2026 FIFA World Cup match outcomes using historical international football data and ELO ratings. Predictions are updated daily via GitHub Actions as the tournament progresses, and displayed on a live web dashboard.

## Data Sources

| Source | Content |
|--------|---------|
| [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Historical match results from 1872 to present |
| [football-data.org](https://www.football-data.org) | 2026 WC fixtures and live results |

## How It Works

1. Historical match data is used to compute ELO ratings and form statistics for every national team
2. Two XGBoost regressors predict expected goals for each team independently
3. A Poisson probability matrix converts expected goals into win/draw/loss probabilities and a most likely scoreline
4. As 2026 WC matches are played, results are fetched automatically, ELO ratings are updated, and predictions for remaining matches are recalculated
5. The web dashboard reads `predictions.csv` directly and reflects every update without redeployment

## Setup

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place the CSV files under `data/`:

```
data/
├── results.csv
├── shootouts.csv
├── goalscorers.csv
└── former_names.csv
```

Set your football-data.org API key as an environment variable:

```bash
export FOOTBALL_DATA_API_KEY=your_key_here
```

## Usage

### First run

```bash
python src/fetch_matches.py   # Fetch 2026 WC fixtures
python src/train.py           # Train the models
python src/predict.py         # Generate predictions
```

### Manual update

```bash
python src/update.py
```

### Automated updates via GitHub Actions

Add `FOOTBALL_DATA_API_KEY` as a repository secret under `Settings → Secrets → Actions`.

The workflow runs daily at 12:00 UTC and automatically:
- Fetches new match results from football-data.org
- Retrains the models with updated data
- Commits refreshed `predictions.csv` and `matches_2026.csv`

### Web dashboard (GitHub Pages)

1. Go to `Settings → Pages`
2. Set source to **Deploy from a branch**, select `main`, root `/`
3. The dashboard will be live at `https://<username>.github.io/<repo>/`

The page fetches `data/predictions.csv` on every load — no redeployment needed when predictions update.

## Model

| | |
|---|---|
| **Algorithm** | XGBoost Regressor (two models: home goals / away goals) |
| **Probability estimation** | Poisson distribution over expected goals |
| **Validation** | 5-fold cross-validation (RMSE) |

### Features

| Feature | Description |
|---------|-------------|
| `elo_diff` | ELO rating difference between the two teams |
| `elo_home_adj` | Home team ELO (home advantage not applied at neutral venues) |
| `form_home/away` | Average points over last 10 matches |
| `form_diff` | Form difference |
| `h2h_*` | Historical head-to-head win/draw/loss rates |
| `is_wc` | Whether the match is a World Cup fixture |
| `is_neutral` | Whether the match is played at a neutral venue |

## Outputs

| File | Description |
|------|-------------|
| `index.html` | Live web dashboard — reads predictions.csv dynamically |
| `data/matches_2026.csv` | Full 2026 WC schedule with results as they come in |
| `data/predictions.csv` | Latest predictions with probabilities and expected scorelines |
