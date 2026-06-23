"""
predict.py
----------
E itilmiş regresyon modellerini kullanarak oynanmamış 2026 WC maçlarının
beklenen gollerini tahmin eder, Poisson dağılımıyla olasılıkları hesaplar.
data/predictions.csv olarak kaydeder.
"""
import os
import joblib
import pandas as pd
from datetime import datetime

from features import build_features, FEATURE_COLS
from poisson_model import match_probabilities, load_rho, score_matrix, top_n_scores

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_HOME_PATH = os.path.join(MODELS_DIR, "model_home.pkl")
MODEL_AWAY_PATH = os.path.join(MODELS_DIR, "model_away.pkl")
LGB_HOME_PATH   = os.path.join(MODELS_DIR, "model_home_lgb.pkl")
LGB_AWAY_PATH   = os.path.join(MODELS_DIR, "model_away_lgb.pkl")
PREDICTIONS_OUT = os.path.join(DATA_DIR, "predictions.csv")


def predict():
    # 1. Modelleri yukle
    for path in [MODEL_HOME_PATH, MODEL_AWAY_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model bulunamadi: {path}\n"
                "Once `python train.py` calistirin."
            )
    model_home = joblib.load(MODEL_HOME_PATH)
    model_away = joblib.load(MODEL_AWAY_PATH)

    # LightGBM modelleri (varsa ensemble, yoksa sadece XGBoost)
    lgb_home = joblib.load(LGB_HOME_PATH) if os.path.exists(LGB_HOME_PATH) else None
    lgb_away = joblib.load(LGB_AWAY_PATH) if os.path.exists(LGB_AWAY_PATH) else None

    if lgb_home:
        print(f"[predict] 4 model yueklendi (XGB + LGB ensemble): {MODELS_DIR}")
    else:
        print(f"[predict] 2 model yueklendi (sadece XGB): {MODELS_DIR}")

    # 2. Feature'ları oluştur (ELO cache varsa kaggle döngüsü atlanır)
    _, pred_df = build_features(
        kaggle_results_path=os.path.join(DATA_DIR, "results.csv"),
        wc2026_path=os.path.join(DATA_DIR, "matches_2026.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
        pred_only=True,
    )

    if pred_df.empty:
        print("[predict] Tahmin edilecek maç yok (tüm maçlar oynanmış olabilir).")
        return

    # Takımı belli olmayan maçları atla (eleme turu — henüz netleşmemiş)
    pred_df = pred_df[pred_df["home_team"].notna() & pred_df["away_team"].notna()].copy()
    print(f"[predict] {len(pred_df)} maç için tahmin üretiliyor...")

    # 3. Beklenen gol sayilari (lambda) — XGB + LGB ensemble
    X = pred_df[FEATURE_COLS]
    xgb_lh = model_home.predict(X)
    xgb_la = model_away.predict(X)

    if lgb_home:
        lgb_lh = lgb_home.predict(X)
        lgb_la = lgb_away.predict(X)
        lambda_home = (xgb_lh + lgb_lh) / 2
        lambda_away = (xgb_la + lgb_la) / 2
    else:
        lambda_home = xgb_lh
        lambda_away = xgb_la

    # 4. Poisson olasılıkları (Dixon-Coles düşük skor düzeltmesi ile)
    rho = load_rho()
    rows = []
    for i, (_, match) in enumerate(pred_df.iterrows()):
        ph, pd_, pa, score = match_probabilities(lambda_home[i], lambda_away[i], rho)

        # En olası 3 skor (Dixon-Coles düzeltmeli matristen) — dashboard'da
        # client-side Poisson hesabıyla tutarsızlık olmaması için burada üretilir.
        matrix = score_matrix(lambda_home[i], lambda_away[i], rho)
        top3 = top_n_scores(matrix, n=3)

        row = {
            "date":              str(match["date"])[:10],
            "kickoff_utc":       match.get("kickoff_utc", ""),
            "home_team":         match["home_team"],
            "away_team":         match["away_team"],
            "stage":             match.get("stage", ""),
            "expected_home":     round(float(lambda_home[i]), 2),
            "expected_away":     round(float(lambda_away[i]), 2),
            "most_likely_score": score,
            "prob_home":         round(ph * 100, 1),
            "prob_draw":         round(pd_ * 100, 1),
            "prob_away":         round(pa * 100, 1),
            "predicted":         "Ev Sahibi" if ph > pd_ and ph > pa
                                 else ("Beraberlik" if pd_ > pa else "Deplasman"),
            "elo_home":          round(float(match["elo_home"]), 1),
            "elo_away":          round(float(match["elo_away"]), 1),
            "updated_at":        datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }
        for rank, (hg, ag) in enumerate(top3, start=1):
            row[f"top{rank}_score"] = f"{hg}-{ag}"
            row[f"top{rank}_prob"]  = round(float(matrix[hg, ag]) * 100, 1)

        rows.append(row)

    results = pd.DataFrame(rows)
    results.sort_values("date", inplace=True)
    results.reset_index(drop=True, inplace=True)

    # 5. Kaydet (CSV + JSON)
    results.to_csv(PREDICTIONS_OUT, index=False)
    json_out = PREDICTIONS_OUT.replace(".csv", ".json")
    results.to_json(json_out, orient="records", force_ascii=False)
    print(f"[predict] Tahminler kaydedildi: {PREDICTIONS_OUT}")

    # 6. Özet yazdır
    print("\n" + "=" * 90)
    print(f"{'Tarih':<12} {'Ev Sahibi':<22} {'Deplasman':<22} {'Skor':>5} {'Tahmin':<12} {'%H':>5} {'%B':>5} {'%D':>5}")
    print("=" * 90)
    for _, row in results.iterrows():
        print(
            f"{row['date']:<12} {row['home_team']:<22} {row['away_team']:<22} "
            f"{row['most_likely_score']:>5} {row['predicted']:<12} "
            f"{row['prob_home']:>5} {row['prob_draw']:>5} {row['prob_away']:>5}"
        )


if __name__ == "__main__":
    predict()
