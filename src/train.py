"""
train.py
--------
XGBoost + LightGBM regresyon modelleri egitir.
Ev sahibi ve deplasman gol sayilarini ayri ayri tahmin eder.
Yakın tarihli maclara daha yuksek ornek agirligi verilir.
Modelleri models/ klasorune kaydeder.
"""

import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import root_mean_squared_error

from features import build_features, FEATURE_COLS
from poisson_model import estimate_rho, save_rho, time_decay_weights

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_HOME_PATH = os.path.join(MODELS_DIR, "model_home.pkl")
MODEL_AWAY_PATH = os.path.join(MODELS_DIR, "model_away.pkl")
LGB_HOME_PATH   = os.path.join(MODELS_DIR, "model_home_lgb.pkl")
LGB_AWAY_PATH   = os.path.join(MODELS_DIR, "model_away_lgb.pkl")


def make_xgb_model():
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )


def make_lgb_model():
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        min_child_samples=15,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def train():
    # 1. Feature'lar
    print("[train] Feature'lar hesaplaniyor...")
    train_df, _ = build_features(
        kaggle_results_path=os.path.join(DATA_DIR, "results.csv"),
        wc2026_path=os.path.join(DATA_DIR, "matches_2026.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )

    train_df = train_df[train_df["date"] >= "1990-01-01"].copy()
    print(f"[train] Egitim seti: {len(train_df)} mac (1990 sonrasi)")

    X      = train_df[FEATURE_COLS].values
    y_home = train_df["home_score"].values
    y_away = train_df["away_score"].values

    # 2. Ornek agirliklari: yakin tarihli maclara daha fazla agirlik
    days_ago = (train_df["date"].max() - train_df["date"]).dt.days.values
    sample_weights = time_decay_weights(days_ago)
    print(f"[train] Ornek agirliklari: min={sample_weights.min():.3f}, "
          f"max={sample_weights.max():.3f}, ortalama={sample_weights.mean():.3f}")

    # 3. Zaman bazli CV (XGBoost, agirliksiz — referans icin)
    print("[train] 5-fold TimeSeriesSplit CV...")
    cv = TimeSeriesSplit(n_splits=5)

    cv_home = -cross_val_score(make_xgb_model(), X, y_home, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_away = -cross_val_score(make_xgb_model(), X, y_away, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"[train] XGB CV Ev   RMSE: {cv_home.mean():.3f} +/- {cv_home.std():.3f}")
    print(f"[train] XGB CV Dep  RMSE: {cv_away.mean():.3f} +/- {cv_away.std():.3f}")

    # 4. Final modelleri — ornek agirliklarla egit
    print("[train] XGBoost modelleri egitiliyor (agirlikli)...")
    model_home = make_xgb_model()
    model_away = make_xgb_model()
    model_home.fit(X, y_home, sample_weight=sample_weights)
    model_away.fit(X, y_away, sample_weight=sample_weights)

    print("[train] LightGBM modelleri egitiliyor (agirlikli)...")
    lgb_home = make_lgb_model()
    lgb_away = make_lgb_model()
    lgb_home.fit(X, y_home, sample_weight=sample_weights)
    lgb_away.fit(X, y_away, sample_weight=sample_weights)

    # 5. Egitim seti RMSE
    xgb_lh = model_home.predict(X)
    xgb_la = model_away.predict(X)
    lgb_lh = lgb_home.predict(X)
    lgb_la = lgb_away.predict(X)

    # Ensemble tahmini (basit ortalama)
    ens_lh = (xgb_lh + lgb_lh) / 2
    ens_la = (xgb_la + lgb_la) / 2

    print(f"[train] XGB egitim RMSE: ev={root_mean_squared_error(y_home, xgb_lh):.3f}  "
          f"dep={root_mean_squared_error(y_away, xgb_la):.3f}")
    print(f"[train] LGB egitim RMSE: ev={root_mean_squared_error(y_home, lgb_lh):.3f}  "
          f"dep={root_mean_squared_error(y_away, lgb_la):.3f}")
    print(f"[train] Ensemble RMSE : ev={root_mean_squared_error(y_home, ens_lh):.3f}  "
          f"dep={root_mean_squared_error(y_away, ens_la):.3f}")

    # 6. Dixon-Coles rho — ensemble tahminiyle zaman agirlikli MLE
    rho_weights = time_decay_weights(days_ago)
    rho = estimate_rho(ens_lh, ens_la, y_home, y_away, weights=rho_weights)
    print(f"[train] Dixon-Coles rho (ensemble, zaman agirlikli): {rho:.4f}")
    save_rho(rho)

    # 7. Kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_home, MODEL_HOME_PATH)
    joblib.dump(model_away, MODEL_AWAY_PATH)
    joblib.dump(lgb_home,   LGB_HOME_PATH)
    joblib.dump(lgb_away,   LGB_AWAY_PATH)
    print(f"[train] 4 model kaydedildi: {MODELS_DIR}")


if __name__ == "__main__":
    train()
