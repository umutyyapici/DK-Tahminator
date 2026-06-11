"""
auto_predict.py
---------------
Otomatik siteye giriş YAPMAZ. Bunun yerine her gün için en yüksek beklenen
puanlı maçı "joker" olarak işaretler ve predictions.csv'ye `is_joker`
sütunu olarak yazar. Tahminleri ve jokeri siteye girmek kullanıcının
elindedir.

Mantık:
  1. predictions.csv'den bugün ve sonraki oynanmamış maçları al
  2. Her gün için beklenen puanı hesapla, en yükseğini joker olarak işaretle
  3. predictions.csv'yi `is_joker` sütunuyla birlikte yeniden kaydet
"""

import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from poisson_model import score_matrix, load_rho, MAX_GOALS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")


def calc_expected_points(pred_h: int, pred_a: int,
                         lambda_h: float, lambda_a: float,
                         rho: float = 0.0,
                         max_g: int = MAX_GOALS) -> float:
    """
    Poisson dağılımıyla (Dixon-Coles düşük skor düzeltmesi ile) beklenen puan hesabı.
    Puanlama: tam isabet=6, kıl payı=3, stratejist=2, bilge=1, teselli=1
    """
    matrix = score_matrix(lambda_h, lambda_a, rho)

    expected = 0.0
    pred_diff = pred_h - pred_a
    pred_result = "home" if pred_h > pred_a else ("away" if pred_h < pred_a else "draw")

    for ah in range(max_g):
        for aa in range(max_g):
            p = matrix[ah, aa]
            if p < 1e-6:
                continue

            actual_result = "home" if ah > aa else ("away" if ah < aa else "draw")
            result_correct = (pred_result == actual_result)
            actual_diff = ah - aa

            if not result_correct:
                # Teselli: yanlış sonuç ama bir takımın golü doğru
                if pred_h == ah or pred_a == aa:
                    expected += p * 1
                continue

            # Sonuç doğru — hangi kategori?
            if pred_h == ah and pred_a == aa:
                pts = 6  # tam isabet
            elif pred_h == ah or pred_a == aa:
                pts = 3  # kıl payı
            elif pred_diff == actual_diff:
                pts = 2  # stratejist
            else:
                pts = 1  # bilge

            expected += p * pts

    return expected


def auto_predict():
    if not os.path.exists(PREDICTIONS_PATH):
        print("[auto_predict] predictions.csv bulunamadı.")
        return

    preds_df = pd.read_csv(PREDICTIONS_PATH)
    preds_df["date"] = pd.to_datetime(preds_df["date"]).dt.date

    # Bugün ve sonrasındaki maçları al (oynanmamışlar)
    today = datetime.now(timezone.utc).date()
    upcoming = preds_df[preds_df["date"] >= today]

    if upcoming.empty:
        print("[auto_predict] Joker belirlenecek maç yok.")
        preds_df["is_joker"] = False
        preds_df.to_csv(PREDICTIONS_PATH, index=False)
        return

    rho = load_rho()

    # Her gün için beklenen puanları hesapla
    by_date = defaultdict(list)
    for idx, row in upcoming.iterrows():
        pred_h = int(round(float(row["most_likely_score"].split("-")[0])))
        pred_a = int(round(float(row["most_likely_score"].split("-")[1])))
        lambda_h = float(row.get("expected_home", 1.2))
        lambda_a = float(row.get("expected_away", 1.0))
        exp_pts = calc_expected_points(pred_h, pred_a, lambda_h, lambda_a, rho)
        by_date[str(row["date"])].append((idx, exp_pts))

    # Her gün için en yüksek beklenen puanlı maçı joker yap
    preds_df["is_joker"] = False
    for date_str, items in sorted(by_date.items()):
        joker_idx, joker_pts = max(items, key=lambda x: x[1])
        preds_df.loc[joker_idx, "is_joker"] = True
        joker_match = preds_df.loc[joker_idx]
        print(f"[auto_predict] {date_str} joker → {joker_match['home_team']} vs {joker_match['away_team']} (E[puan]={joker_pts:.2f})")

    preds_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"[auto_predict] {len(by_date)} gün için joker işaretlendi, predictions.csv güncellendi.")


if __name__ == "__main__":
    auto_predict()
