# WC 2026 Match Predictor

A machine learning project that predicts 2026 FIFA World Cup match outcomes using historical international football data and ELO ratings. Predictions are updated daily via GitHub Actions as the tournament progresses, and displayed on a live web dashboard.

## Backtest Performance

Evaluated on 3,610 matches from major tournaments (2018–2025) across all confederations:

| Metric | Value |
|--------|-------|
| Accuracy | 61.4% |
| Exact Score | 13.0% |
| Top-3 Score | 37.6% |
| Log Loss | 0.853 |
| Brier Score | 0.167 |
| Baseline (always home) | 46.1% |
| Dixon-Coles ρ (estimated) | -0.053 |

## Data Sources

| Source | Content |
|--------|---------|
| [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Historical match results from 1872 to present |
| [football-data.org](https://www.football-data.org) | 2026 WC fixtures and live results |

## How It Works

1. Historical match data is used to compute ELO ratings, confederation strength, and form statistics for every national team. ELO updates use a **per-tournament K-factor table** (e.g. World Cup finals = 60, World Cup qualifiers = 40, regional cups = 22, multi-sport games / CONIFA = 14) so that the rating impact of a match reflects its actual competitive level
2. Two XGBoost regressors predict expected goals for each team independently
3. A Poisson probability matrix (0–8 goals) converts expected goals into win/draw/loss probabilities and a most likely scoreline. A **Dixon-Coles low-score correction** (Dixon & Coles, 1997) is applied to the (0-0), (1-0), (0-1) and (1-1) cells using a correlation parameter ρ, which corrects the tendency of an independent Poisson model to underestimate draws in low-scoring matches
4. ρ is estimated automatically via **time-weighted** maximum likelihood on historical results during training (recent matches count more) and stored in `models/rho.json`
5. As 2026 WC matches are played, results are fetched automatically, ELO ratings are updated, and predictions for remaining matches are recalculated
6. Completed matches are compared against predictions to compute a live accuracy score
7. For each upcoming day, the match with the highest expected points is marked as that day's "joker" (`is_joker` column in `predictions.csv`, shown with a 🃏 badge on the dashboard) — predictions and joker picks are entered into the prediction game manually
8. The web dashboard reads all output files dynamically — no redeployment needed

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
| **Algorithm** | XGBoost Regressor (two models: home goals / away goals), tuned via `RandomizedSearchCV` |
| **Probability estimation** | Poisson distribution over expected goals (0–8 goal matrix) |
| **Low-score correlation** | Dixon-Coles τ adjustment on the 0-0/1-0/0-1/1-1 cells, ρ estimated via time-weighted maximum likelihood during training |
| **Validation** | 5-fold time-based cross-validation (`TimeSeriesSplit`, RMSE) — each fold trains only on past matches and tests on a later block, avoiding lookahead bias |

### Features

| Feature | Description |
|---------|-------------|
| `elo_diff` | ELO rating difference between the two teams |
| `elo_home_adj` | Home team ELO (home advantage not applied at neutral venues) |
| `elo_away` | Away team ELO |
| `elo_win_prob_home` | ELO-implied win probability for the home team (logistic transform of the ELO difference, including home advantage) |
| `form_home/away` | Average points over last 10 matches |
| `form_diff` | Form difference |
| `h2h_*` | Historical head-to-head win/draw/loss rates |
| `elo_momentum_home/away` | ELO change over last 5 matches (rising vs falling team) |
| `elo_momentum_diff` | Momentum difference between teams |
| `atk_home/away` | Average goals scored over last 10 matches |
| `def_home/away` | Average goals conceded over last 10 matches |
| `atk_vs_def_home/away` | Attack vs. opponent defense differential |
| `conf_strength_home/away` | Average ELO of all teams in the team's confederation (UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC) up to that point in time |
| `conf_strength_diff` | Confederation strength difference |
| `is_wc` | Whether the match is a World Cup fixture |
| `is_neutral` | Whether the match is played at a neutral venue |

### ELO tournament weights (K-factor)

Each match's impact on ELO ratings (`K` factor) is set per tournament based on its competitive level — World Cup finals carry the most weight, followed by continental championships, World Cup qualifiers, regional cups, friendlies, and multi-sport/amateur games at the bottom. See `TOURNAMENT_K` in `src/features.py` for the full table. Previously, qualification rounds inherited the same K-factor as the corresponding final tournament (e.g. World Cup qualifiers were weighted the same as the World Cup final, K=60); each tournament now has its own value.

## Outputs

| File | Description |
|------|-------------|
| `index.html` | Live web dashboard — reads all data files dynamically |
| `data/matches_2026.csv` | Full 2026 WC schedule with results as they come in |
| `data/predictions.csv` | Latest predictions with probabilities, expected scorelines, and the daily `is_joker` pick |
| `data/accuracy.json` | Live accuracy stats — updated after each match day |
| `data/backtest.json` | Backtest results across 2018+ major tournaments (3,610 matches) |
| `models/rho.json` | Fitted Dixon-Coles correlation parameter (ρ), re-estimated on every training run |
