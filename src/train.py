"""
train.py
--------
ELO feature'larını kullanarak iki paralel XGBoost Regresyon modeli eğitir.
Ev sahibi ve deplasman gol sayılarını ayrı ayrı tahmin eder.
Modelleri models/ klasörüne kaydeder.
"""

import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import root_mean_squared_error

from features import build_features, FEATURE_COLS
from poisson_model import estimate_rho, save_rho, time_decay_weights

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_HOME_PATH = os.path.join(MODELS_DIR, "model_home.pkl")
MODEL_AWAY_PATH = os.path.join(MODELS_DIR, "model_away.pkl")


def make_model():
    # RandomizedSearchCV (TimeSeriesSplit, 5 fold) ile bulunan parametreler
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


def train():
    # 1. Feature'ları oluştur
    print("[train] Feature'lar hesaplanıyor...")
    train_df, _ = build_features(
        kaggle_results_path=os.path.join(DATA_DIR, "results.csv"),
        wc2026_path=os.path.join(DATA_DIR, "matches_2026.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )

    train_df = train_df[train_df["date"] >= "1990-01-01"].copy()
    print(f"[train] Eğitim seti: {len(train_df)} maç (1990 sonrası)")

    X      = train_df[FEATURE_COLS].values
    y_home = train_df["home_score"].values
    y_away = train_df["away_score"].values

    # 2. CV (gerçek hata oranı — model kendi verisini görmez)
    # Veri kronolojik sıralı olduğundan zaman bazlı (walk-forward) CV
    # kullanılıyor: her fold yalnızca kendinden önceki maçlarla eğitilip
    # sonraki bir bloğu test eder. Rastgele KFold, geleceği görerek
    # geçmişi test etmeye izin verip iyimser bir RMSE verebilirdi.
    print("[train] 5-fold zaman bazlı (TimeSeriesSplit) çapraz doğrulama çalışıyor...")
    cv = TimeSeriesSplit(n_splits=5)

    cv_home = -cross_val_score(make_model(), X, y_home, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_away = -cross_val_score(make_model(), X, y_away, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)

    print(f"[train] CV Ev Sahibi RMSE : {cv_home.mean():.3f} ± {cv_home.std():.3f} gol")
    print(f"[train] CV Deplasman RMSE : {cv_away.mean():.3f} ± {cv_away.std():.3f} gol")

    # 3. Final modelleri tüm veriyle eğit
    print("[train] Final modeller eğitiliyor...")
    model_home = make_model()
    model_away = make_model()
    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    # 4. Eğitim seti RMSE (referans — CV'den düşük olması normal)
    lambda_home_in = model_home.predict(X)
    lambda_away_in = model_away.predict(X)
    rmse_h = root_mean_squared_error(y_home, lambda_home_in)
    rmse_a = root_mean_squared_error(y_away, lambda_away_in)
    print(f"[train] Eğitim Ev Sahibi RMSE : {rmse_h:.3f} gol  (CV ile kıyasla)")
    print(f"[train] Eğitim Deplasman RMSE : {rmse_a:.3f} gol  (CV ile kıyasla)")

    # 4b. Dixon-Coles düşük skor düzeltmesi (rho), zaman ağırlıklı MLE ile
    #     tahmin edilir — yakın tarihli maçlar rho'ya daha fazla etki eder.
    days_ago = (train_df["date"].max() - train_df["date"]).dt.days.values
    weights = time_decay_weights(days_ago)
    rho = estimate_rho(lambda_home_in, lambda_away_in, y_home, y_away, weights=weights)
    print(f"[train] Dixon-Coles rho (zaman ağırlıklı): {rho:.4f}")
    save_rho(rho)

    # 5. Kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_home, MODEL_HOME_PATH)
    joblib.dump(model_away, MODEL_AWAY_PATH)
    print(f"[train] Modeller kaydedildi: {MODELS_DIR}")


if __name__ == "__main__":
    train()
