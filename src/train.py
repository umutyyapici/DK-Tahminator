"""
train.py
--------
ELO feature'larını kullanarak iki paralel XGBoost Regresyon modeli eğitir.
Ev sahibi ve Deplasman gol sayılarını ayrı ayrı tahmin etmeyi öğrenir.
Modelleri models/ klasörüne kaydeder.
"""
import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error
from features import build_features, FEATURE_COLS

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_HOME_PATH = os.path.join(MODELS_DIR, "model_home.pkl")
MODEL_AWAY_PATH = os.path.join(MODELS_DIR, "model_away.pkl")


def train():
    # 1. Feature'ları oluştur
    print("[train] Feature'lar hesaplanıyor...")
    train_df, _ = build_features(
        kaggle_results_path=os.path.join(DATA_DIR, "results.csv"),
        wc2026_path=os.path.join(DATA_DIR, "matches_2026.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )

    # 2. Yalnızca 1990 sonrası modern maçları kullan
    train_df = train_df[train_df["date"] >= "1990-01-01"].copy()
    print(f"[train] Regresyon Eğitim seti: {len(train_df)} maç (1990 sonrası)")

    X      = train_df[FEATURE_COLS].values
    y_home = train_df["home_score"].values  # Hedef: Ev sahibinin attığı gol
    y_away = train_df["away_score"].values  # Hedef: Deplasmanın attığı gol

    # 3. Model tanımları
    model_home = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model_away = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # 4. Çapraz Doğrulama (model kendi görmediği veriyi test eder)
    print("[train] 5-fold çapraz doğrulama ile hata oranları ölçülüyor...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_home = -cross_val_score(model_home, X, y_home, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_away = -cross_val_score(model_away, X, y_away, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)

    print(f"[train] CV Ev Sahibi RMSE : {cv_home.mean():.3f} ± {cv_home.std():.3f} gol")
    print(f"[train] CV Deplasman RMSE : {cv_away.mean():.3f} ± {cv_away.std():.3f} gol")

    # 5. Modelleri tüm veriyle eğit
    print("[train] Ev sahibi gol modeli eğitiliyor...")
    model_home.fit(X, y_home)

    print("[train] Deplasman gol modeli eğitiliyor...")
    model_away.fit(X, y_away)

    # 6. Eğitim seti RMSE (CV'den düşük olması normal — referans için)
    pred_home = model_home.predict(X)
    pred_away = model_away.predict(X)
    rmse_h = root_mean_squared_error(y_home, pred_home)
    rmse_a = root_mean_squared_error(y_away, pred_away)
    print(f"[train] Eğitim Ev Sahibi RMSE: {rmse_h:.3f} gol  (CV ile kıyasla)")
    print(f"[train] Eğitim Deplasman RMSE: {rmse_a:.3f} gol  (CV ile kıyasla)")

    # 7. Kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_home, MODEL_HOME_PATH)
    joblib.dump(model_away, MODEL_AWAY_PATH)
    print(f"[train] İki adet gol modeli başarıyla kaydedildi: {MODELS_DIR}")


if __name__ == "__main__":
    train()