"""
update.py
---------
GitHub Action tarafından günlük çalıştırılan ana script.
Sırayla:
  1. football-data.org'dan yeni maç sonuçlarını çeker
  2. Modeli yeniden eğitir (ELO ve feature'lar güncellenir)
  3. Tahminleri yeniden üretir
  4. Oynanan maçlardaki tahmin başarısını hesaplar
  5. Backtest metriklerini günceller
  6. Her gün için joker maçı belirler ve predictions.csv'ye işaretler
"""

import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from fetch_matches import fetch_wc2026_matches, save_matches
from fetch_rankings import fetch_fifa_rankings, save_rankings
from train import train
from predict import predict
from evaluate import evaluate
from auto_predict import auto_predict

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
ACCURACY_OUT = os.path.join(DATA_DIR, "accuracy.json")
HISTORY_PATH = os.path.join(DATA_DIR, "predictions_history.csv")

HISTORY_COLS = ["date", "kickoff_utc", "home_team", "away_team", "stage", "predicted", "most_likely_score",
                "expected_home", "expected_away", "prob_home", "prob_draw", "prob_away", "elo_home", "elo_away",
                "top1_score", "top1_prob", "top2_score", "top2_prob", "top3_score", "top3_prob", "is_joker"]


def archive_predictions():
    """
    predict() her çalıştığında predictions.csv'yi sıfırdan üretir ve artık
    oynanmış maçları listeden düşürür. Bu yüzden bir maç bittiğinde tahmini
    kaybolmadan önce predictions_history.csv'ye kalıcı olarak kaydedilir —
    calc_accuracy() bu dosyayı kullanır.
    """
    preds_path = os.path.join(DATA_DIR, "predictions.csv")
    if not os.path.exists(preds_path):
        return

    preds = pd.read_csv(preds_path)
    preds = preds[[c for c in HISTORY_COLS if c in preds.columns]]

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        combined = pd.concat([history, preds], ignore_index=True)
    else:
        combined = preds

    # Bir maç için en son yapılan tahmin (maça en yakın olan) saklanır
    combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last", inplace=True)
    combined.to_csv(HISTORY_PATH, index=False)
    print(f"[archive] Tahmin geçmişi güncellendi: {HISTORY_PATH} ({len(combined)} maç)")


def calc_accuracy():
    matches_path = os.path.join(DATA_DIR, "matches_2026.csv")
    preds_path   = HISTORY_PATH

    if not os.path.exists(matches_path) or not os.path.exists(preds_path):
        print("[accuracy] Gerekli dosyalar bulunamadı, atlanıyor.")
        return

    matches = pd.read_csv(matches_path)
    preds   = pd.read_csv(preds_path)

    finished = matches[matches["status"] == "FINISHED"].copy()
    if finished.empty:
        print("[accuracy] Henüz oynanmış maç yok.")
        _save_accuracy(0, 0, 0, 0, 0, 0, 0)
        return

    merged = finished.merge(
        preds[["home_team", "away_team", "date", "predicted", "most_likely_score", "expected_home", "expected_away"]],
        on=["home_team", "away_team", "date"],
        how="inner"
    )
    merged = merged.dropna(subset=["home_score", "away_score"])

    if merged.empty:
        print("[accuracy] Eşleşen tahmin bulunamadı.")
        _save_accuracy(0, 0, 0, 0, 0, 0, 0)
        return

    total    = len(merged)
    correct  = 0
    exact    = 0
    top3_hit = 0

    def top3_scores(lh, la, n=9):
        from scipy.stats import poisson as sp
        hp = sp.pmf(range(n), max(0.01, lh))
        ap = sp.pmf(range(n), max(0.01, la))
        flat = [(i, j, hp[i]*ap[j]) for i in range(n) for j in range(n)]
        flat.sort(key=lambda x: -x[2])
        return [(i, j) for i, j, _ in flat[:3]]

    for _, row in merged.iterrows():
        hg = int(row["home_score"])
        ag = int(row["away_score"])

        if hg > ag:    actual = "Ev Sahibi"
        elif hg == ag: actual = "Beraberlik"
        else:          actual = "Deplasman"

        if row["predicted"] == actual:
            correct += 1

        # 1. skor
        if pd.notna(row.get("most_likely_score", None)):
            try:
                ph, pa = row["most_likely_score"].split("-")
                if int(ph) == hg and int(pa) == ag:
                    exact += 1
            except Exception:
                pass

        # Top 3
        try:
            lh = float(row.get("expected_home", 1.2))
            la = float(row.get("expected_away", 1.0))
            if any(h == hg and a == ag for h, a in top3_scores(lh, la)):
                top3_hit += 1
        except Exception:
            pass

    accuracy   = round(correct  / total * 100, 1) if total > 0 else 0
    exact_pct  = round(exact    / total * 100, 1) if total > 0 else 0
    top3_pct   = round(top3_hit / total * 100, 1) if total > 0 else 0

    _save_accuracy(accuracy, correct, total, exact, exact_pct, top3_hit, top3_pct)
    print(f"[accuracy] {correct}/{total} doğru → %{accuracy}  |  1. skor: {exact} (%{exact_pct})  |  Top3: {top3_hit} (%{top3_pct})")


def _save_accuracy(accuracy, correct, total, exact, exact_pct=0, top3=0, top3_pct=0):
    data = {
        "accuracy":   accuracy,
        "correct":    correct,
        "total":      total,
        "exact":      exact,
        "exact_pct":  exact_pct,
        "top3":       top3,
        "top3_pct":   top3_pct,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    os.makedirs(os.path.dirname(ACCURACY_OUT), exist_ok=True)
    with open(ACCURACY_OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[accuracy] Kaydedildi: {ACCURACY_OUT}")


def main(light: bool = False):
    total = 6 if light else 8
    step = 0

    def header(title):
        nonlocal step
        step += 1
        print("\n" + "=" * 50)
        print(f"ADIM {step}/{total}: {title}")
        print("=" * 50)

    header("FIFA siralamasi guncelleniyor...")
    try:
        rankings = fetch_fifa_rankings()
        save_rankings(rankings)
    except Exception as e:
        print(f"[rankings] UYARI: FIFA siralamasi cekilemedi: {e}")

    header("Mac sonuclari cekiliyor...")
    df = fetch_wc2026_matches()
    save_matches(df)

    header("Önceki tahminler arşivleniyor...")
    archive_predictions()

    if not light:
        header("Model yeniden eğitiliyor...")
        train()

    header("Tahminler üretiliyor...")
    predict()

    header("Başarı oranı hesaplanıyor...")
    calc_accuracy()

    if not light:
        header("Backtest güncelleniyor...")
        try:
            evaluate()
        except Exception as e:
            print(f"[backtest] HATA: {e}")

    header("Joker maçlar belirleniyor...")
    try:
        auto_predict()
    except Exception as e:
        print(f"[auto_predict] HATA: {e}")

    print("\n✓ Güncelleme tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC2026 veri güncelleme")
    parser.add_argument(
        "--light", action="store_true",
        help="Sadece sonuçları/tahminleri günceller; model eğitimi (train) ve "
             "backtest (evaluate) adımlarını atlar."
    )
    args = parser.parse_args()
    main(light=args.light)
