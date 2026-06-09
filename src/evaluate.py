"""
evaluate.py
-----------
Modelin geçmiş Dünya Kupası maçlarındaki başarısını ölçer.

Strateji:
  - Train seti : 2022 WC öncesi tüm maçlar (1990+)
  - Test seti  : 2022 FIFA World Cup maçları (gerçek sonuçlar biliniyor)

Metrikler:
  - Outcome accuracy    : home/draw/away tahmin doğruluğu
  - Brier score         : olasılık kalibrasyonu (düşük = iyi)
  - Log loss            : olasılık kalitesi (düşük = iyi)
  - Exact score accuracy: en olası skor tahminin doğruluğu
  - Referans (baseline) : her zaman ev sahibi galip de demek
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import log_loss, brier_score_loss
from scipy.stats import poisson

BACKTEST_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "backtest.json")

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, FEATURE_COLS, EloFeatureBuilder, get_k, HOME_ADVANTAGE
from features import actual_score, goal_diff_multiplier, build_name_map, normalize_team, FORM_WINDOW

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
WC2022_START = "2022-11-20"
WC2022_END   = "2022-12-18"
MAX_GOALS    = 7


# ── Poisson olasılıkları ─────────────────────────────────────────────────

def poisson_probs(lh, la):
    lh = max(0.01, lh)
    la = max(0.01, la)
    hp = poisson.pmf(range(MAX_GOALS), lh)
    ap = poisson.pmf(range(MAX_GOALS), la)
    matrix = np.outer(hp, ap)
    ph = float(np.tril(matrix, -1).sum())
    pd_ = float(np.trace(matrix))
    pa = float(np.triu(matrix, 1).sum())
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return ph, pd_, pa, idx[0], idx[1]


# ── Train/test split için özel feature builder ───────────────────────────

def build_train_test(kaggle_path, former_names_path):
    """
    Tüm tarihi 1990'dan itibaren işler.
    2022 WC öncesi → train, 2022 WC maçları → test olarak ayırır.
    Her iki set için de feature'lar ELO sürekliliğiyle hesaplanır
    (test seti ELO'yu görmez, sadece train seti ELO'su kullanılır).
    """
    hist = pd.read_csv(kaggle_path, parse_dates=["date"])
    name_map = build_name_map(former_names_path)
    for col in ["home_team", "away_team"]:
        hist[col] = hist[col].map(lambda x: normalize_team(x, name_map))
    hist["neutral"] = hist["neutral"].map(
        lambda x: True if str(x).upper() == "TRUE" else False
    )
    hist = hist[hist["date"] >= "1990-01-01"].sort_values("date").reset_index(drop=True)

    builder = EloFeatureBuilder()
    h2h_records = {}
    train_rows = []
    test_rows  = []

    for _, row in hist.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        home  = row["home_team"]
        away  = row["away_team"]
        hg    = int(row["home_score"])
        ag    = int(row["away_score"])
        neut  = bool(row["neutral"])
        tourn = str(row["tournament"])
        date  = row["date"]

        elo_h  = builder.get_elo(home)
        elo_a  = builder.get_elo(away)
        form_h = builder.get_form(home)
        form_a = builder.get_form(away)
        h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records)

        feat = {
            "date":         date,
            "home_team":    home,
            "away_team":    away,
            "home_score":   hg,
            "away_score":   ag,
            "tournament":   tourn,
            "elo_diff":     elo_h - elo_a,
            "elo_home_adj": elo_h + (0 if neut else HOME_ADVANTAGE),
            "elo_away":     elo_a,
            "form_home":    form_h,
            "form_away":    form_a,
            "form_diff":    form_h - form_a,
            "h2h_home_win": h2h_w,
            "h2h_draw":     h2h_d,
            "h2h_away_win": h2h_l,
            "is_wc":        1 if "world cup" in tourn.lower() else 0,
            "is_neutral":   1 if neut else 0,
        }

        is_wc2022 = (
            "world cup" in tourn.lower() and
            WC2022_START <= str(date)[:10] <= WC2022_END
        )

        if is_wc2022:
            test_rows.append(feat)
        else:
            train_rows.append(feat)

        # ELO ve H2H güncelle (test maçları da günceller — gerçekçi simülasyon)
        builder.update_elo(home, away, hg, ag, neut, tourn)
        key = (home, away)
        if key not in h2h_records:
            h2h_records[key] = []
        h2h_records[key].append(actual_score(hg, ag))

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


# ── Metrik hesaplayıcılar ────────────────────────────────────────────────

def outcome_label(hg, ag):
    if hg > ag: return 2
    if hg == ag: return 1
    return 0


def evaluate_predictions(df, model_home, model_away, label=""):
    X   = df[FEATURE_COLS].values
    lh  = model_home.predict(X)
    la  = model_away.predict(X)

    y_true       = []
    y_pred_class = []
    y_proba      = []
    exact_correct = 0

    for i, (_, row) in enumerate(df.iterrows()):
        ph, pd_, pa, sh, sa = poisson_probs(lh[i], la[i])
        true_label = outcome_label(row["home_score"], row["away_score"])

        y_true.append(true_label)
        y_pred_class.append(np.argmax([pa, pd_, ph]))  # 0=away,1=draw,2=home
        y_proba.append([pa, pd_, ph])

        if sh == int(row["home_score"]) and sa == int(row["away_score"]):
            exact_correct += 1

    y_true       = np.array(y_true)
    y_pred_class = np.array(y_pred_class)
    y_proba      = np.array(y_proba)

    accuracy      = (y_true == y_pred_class).mean()
    exact_acc     = exact_correct / len(df)
    ll            = log_loss(y_true, y_proba, labels=[0, 1, 2])

    # Brier score: her sınıf için ayrı hesapla, ortalamasını al
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(3)
    ])

    # Baseline: her zaman ev sahibi galip de
    baseline_acc = (y_true == 2).mean()

    n = len(df)
    print(f"\n{'='*50}")
    print(f"  {label}  ({n} maç)")
    print(f"{'='*50}")
    print(f"  Outcome Accuracy   : {accuracy:.3f}   (baseline: {baseline_acc:.3f})")
    print(f"  Exact Score Acc    : {exact_acc:.3f}")
    print(f"  Log Loss           : {ll:.3f}   (düşük = iyi)")
    print(f"  Brier Score (avg)  : {brier:.3f}  (düşük = iyi)")

    # Sınıf bazlı dağılım
    print(f"\n  Tahmin dağılımı:")
    for c, lbl in [(2,"Ev Sahibi"), (1,"Beraberlik"), (0,"Deplasman")]:
        pred_n = (y_pred_class == c).sum()
        true_n = (y_true == c).sum()
        print(f"    {lbl:<12}: tahmin={pred_n:>3}  gerçek={true_n:>3}")

    return {
        "accuracy": accuracy, "exact_acc": exact_acc,
        "log_loss": ll, "brier": brier, "baseline": baseline_acc,
    }


# ── Ana fonksiyon ────────────────────────────────────────────────────────

def evaluate():
    print("[eval] Veri yükleniyor ve train/test ayrılıyor...")
    train_df, test_df = build_train_test(
        kaggle_path=os.path.join(DATA_DIR, "results.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )
    print(f"[eval] Train: {len(train_df)} maç  |  Test (2022 WC): {len(test_df)} maç")

    if test_df.empty:
        print("[eval] HATA: 2022 WC maçı bulunamadı. results.csv'yi kontrol edin.")
        return

    # Modelleri train seti üzerinde eğit
    print("\n[eval] Modeller eğitiliyor (train seti)...")
    model_home = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model_away = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

    X_train = train_df[FEATURE_COLS].values
    model_home.fit(X_train, train_df["home_score"].values)
    model_away.fit(X_train, train_df["away_score"].values)

    # Train seti metrikleri (overfitting referansı)
    evaluate_predictions(train_df, model_home, model_away, "TRAIN SETİ")

    # Train seti metrikleri (overfitting referansı)
    train_metrics = evaluate_predictions(train_df, model_home, model_away, "TRAIN SETİ")

    # Test seti metrikleri (gerçek performans)
    test_metrics = evaluate_predictions(test_df, model_home, model_away, "TEST SETİ — 2022 FIFA World Cup")

    print("\n[eval] Not: Test accuracy > baseline ise model rastgele tahminden iyidir.")
    print("[eval] Futbol tahmininde %50+ accuracy iyi kabul edilir.")

    # backtest.json kaydet
    backtest = {
        "test_tournament":  "2022 FIFA World Cup",
        "test_matches":     len(test_df),
        "train_matches":    len(train_df),
        "accuracy":         round(float(test_metrics["accuracy"]) * 100, 1),
        "exact_acc":        round(float(test_metrics["exact_acc"]) * 100, 1),
        "log_loss":         round(float(test_metrics["log_loss"]), 3),
        "brier":            round(float(test_metrics["brier"]), 3),
        "baseline":         round(float(test_metrics["baseline"]) * 100, 1),
        "train_accuracy":   round(float(train_metrics["accuracy"]) * 100, 1),
        "updated_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    os.makedirs(os.path.dirname(BACKTEST_OUT), exist_ok=True)
    with open(BACKTEST_OUT, "w", encoding="utf-8") as f:
        json.dump(backtest, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Backtest sonuçları kaydedildi: {BACKTEST_OUT}")


if __name__ == "__main__":
    evaluate()
