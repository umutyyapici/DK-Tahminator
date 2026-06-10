# WC 2026 Match Predictor

A machine learning project that predicts 2026 FIFA World Cup match outcomes using historical international football data and ELO ratings. Predictions are updated daily via GitHub Actions as the tournament progresses, and displayed on a live web dashboard.

## Backtest Performance

Evaluated on 3,610 matches from major tournaments (2018–2025) across all confederations:

| Metric | Value |
|--------|-------|
| Accuracy | 60.8% |
| Exact Score | 13.3% |
| Log Loss | 0.861 |
| Brier Score | 0.168 |
| Baseline (always home) | 46.1% |

## Data Sources

| Source | Content |
|--------|---------|
| [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Historical match results from 1872 to present |
| [football-data.org](https://www.football-data.org) | 2026 WC fixtures and live results |

## How It Works

1. Historical match data is used to compute ELO ratings and form statistics for every national team
2. Two XGBoost regressors predict expected goals for each team independently
3. A Poisson probability matrix (0–8 goals) converts expected goals into win/draw/loss probabilities and a most likely scoreline
4. As 2026 WC matches are played, results are fetched automatically, ELO ratings are updated, and predictions for remaining matches are recalculated
5. Completed matches are compared against predictions to compute a live accuracy score
6. The web dashboard reads all output files dynamically — no redeployment needed

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

The workflow runs daily at 10:00 UTC (13:00 Turkey) and automatically:
- Fetches new match results from football-data.org
- Retrains the models with updated data
- Runs backtest evaluation
- Commits refreshed `predictions.csv`, `matches_2026.csv`, `accuracy.json`, and `backtest.json`

### Web dashboard (GitHub Pages)

1. Go to `Settings → Pages`
2. Set source to **Deploy from a branch**, select `main`, root `/`
3. The dashboard will be live at `https://<username>.github.io/<repo>/`

The page fetches output files on every load — no redeployment needed when data updates.

### Model evaluation (backtest)

```bash
python src/evaluate.py
```

Trains on historical data and tests on major tournaments from 2018 onward across all confederations (FIFA World Cup + qualification, UEFA Euro, UEFA Nations League, Copa América, African Cup of Nations, AFC Asian Cup, Gold Cup, CONCACAF Nations League, Oceania Nations Cup). Reports accuracy, log loss, Brier score, and exact score accuracy.

## Model

| | |
|---|---|
| **Algorithm** | XGBoost Regressor (two models: home goals / away goals) |
| **Probability estimation** | Poisson distribution over expected goals (0–8 goal matrix) |
| **Validation** | 5-fold cross-validation (RMSE) |

### Features

| Feature | Description |
|---------|-------------|
| `elo_diff` | ELO rating difference between the two teams |
| `elo_home_adj` | Home team ELO (home advantage not applied at neutral venues) |
| `form_home/away` | Average points over last 10 matches |
| `form_diff` | Form difference |
| `h2h_*` | Historical head-to-head win/draw/loss rates |
| `elo_momentum_home/away` | ELO change over last 5 matches (rising vs falling team) |
| `elo_momentum_diff` | Momentum difference between teams |
| `is_wc` | Whether the match is a World Cup fixture |
| `is_neutral` | Whether the match is played at a neutral venue |

## Outputs

| File | Description |
|------|-------------|
| `index.html` | Live web dashboard — reads all data files dynamically |
| `data/matches_2026.csv` | Full 2026 WC schedule with results as they come in |
| `data/predictions.csv` | Latest predictions with probabilities and expected scorelines |
| `data/accuracy.json` | Live accuracy stats — updated after each match day |
| `data/backtest.json` | Backtest results across 2018+ major tournaments (3,610 matches) |
