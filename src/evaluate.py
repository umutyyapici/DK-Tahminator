"""
evaluate.py
-----------
Modelin geçmiş büyük turnuvalardaki başarısını ölçer.

Strateji:
  - Train seti : Test setindeki en erken maçtan önce oynanan tüm maçlar (1990+)
  - Test seti  : 2018-06-01 sonrası ana turnuvalar + FIFA WC qualification
      * FIFA World Cup + qualification (2018 sonrası)
      * UEFA Euro, UEFA Nations League
      * Copa América, African Cup of Nations
      * AFC Asian Cup, Gold Cup, CONCACAF Nations League
      * Oceania Nations Cup
      Toplam ~3.700 maç, train/test oranı ~%12

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
from sklearn.metrics import log_loss, brier_score_loss

BACKTEST_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "backtest.json")

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLS, EloFeatureBuilder, HOME_ADVANTAGE, FORM_WINDOW_SHORT
from features import actual_score, build_name_map, normalize_team, expected_score
from poisson_model import score_matrix, top_n_scores, estimate_rho, time_decay_weights
from train import make_xgb_model

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Test setine alınacak turnuva + tarih aralıkları ──────────────────────
# Kaggle results.csv'deki gerçek turnuva adları kullanılıyor.
# "Africa Cup of Nations" Kaggle'da yok — çıkarıldı.
# Test seti: 2018-06-01 sonrası ana turnuvalar + FIFA WC qualification
# Kaggle'daki exact turnuva adları kullanılıyor
TEST_CUTOFF = "2018-06-01"
TEST_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Nations League",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Nations League",
    "Oceania Nations Cup",
]

def is_test_match(tourn: str, date_str: str) -> bool:
    return tourn in TEST_TOURNAMENTS and date_str >= TEST_CUTOFF


# ── Poisson olasılıkları ─────────────────────────────────────────────────

def poisson_probs(lh, la, rho=0.0):
    matrix = score_matrix(lh, la, rho)
    ph  = float(np.tril(matrix, -1).sum())
    pd_ = float(np.trace(matrix))
    pa  = float(np.triu(matrix, 1).sum())
    top3 = top_n_scores(matrix, 3)
    sh, sa = top3[0]
    return ph, pd_, pa, sh, sa, top3


# ── Train/test split ─────────────────────────────────────────────────────

def build_train_test(kaggle_path, former_names_path):
    hist = pd.read_csv(kaggle_path, parse_dates=["date"])
    name_map = build_name_map(former_names_path)
    for col in ["home_team", "away_team"]:
        hist[col] = hist[col].map(lambda x: normalize_team(x, name_map))
    hist["neutral"] = hist["neutral"].map(
        lambda x: True if str(x).upper() == "TRUE" else False
    )
    hist = hist[hist["date"] >= "1990-01-01"].sort_values("date").reset_index(drop=True)

    builder     = EloFeatureBuilder()
    h2h_records = {}
    train_rows  = []
    test_rows   = []

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
        ds    = str(date)[:10]

        elo_h   = builder.get_elo(home)
        elo_a   = builder.get_elo(away)
        form_h  = builder.get_form(home)
        form_a  = builder.get_form(away)
        form5_h = builder.get_form(home, window=FORM_WINDOW_SHORT)
        form5_a = builder.get_form(away, window=FORM_WINDOW_SHORT)
        mom_h   = builder.get_elo_momentum(home)
        mom_a   = builder.get_elo_momentum(away)
        atk_h   = builder.get_avg_scored(home)
        def_h   = builder.get_avg_conceded(home)
        atk_a   = builder.get_avg_scored(away)
        def_a   = builder.get_avg_conceded(away)
        h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records,
                                                current_date=date)
        conf_h  = builder.get_conf_strength(home)
        conf_a  = builder.get_conf_strength(away)
        elo_win_prob_home = expected_score(elo_h + (0 if neut else HOME_ADVANTAGE), elo_a)

        feat = {
            "date":               date,
            "home_team":          home,
            "away_team":          away,
            "home_score":         hg,
            "away_score":         ag,
            "tournament":         tourn,
            "elo_diff":           elo_h - elo_a,
            "elo_home_adj":       elo_h + (0 if neut else HOME_ADVANTAGE),
            "elo_away":           elo_a,
            "form_home":          form_h,
            "form_away":          form_a,
            "form_diff":          form_h - form_a,
            "form5_home":         form5_h,
            "form5_away":         form5_a,
            "form5_diff":         form5_h - form5_a,
            "h2h_home_win":       h2h_w,
            "h2h_draw":           h2h_d,
            "h2h_away_win":       h2h_l,
            "is_wc":              1 if "world cup" in tourn.lower() else 0,
            "is_neutral":         1 if neut else 0,
            "elo_momentum_home":  mom_h,
            "elo_momentum_away":  mom_a,
            "elo_momentum_diff":  mom_h - mom_a,
            "atk_home":           atk_h,
            "def_home":           def_h,
            "atk_away":           atk_a,
            "def_away":           def_a,
            "atk_vs_def_home":    atk_h - def_a,
            "atk_vs_def_away":    atk_a - def_h,
            "elo_win_prob_home":  elo_win_prob_home,
            "conf_strength_home": conf_h,
            "conf_strength_away": conf_a,
            "conf_strength_diff": conf_h - conf_a,
            # FIFA rankings — use defaults for historical backtest
            "fifa_rank_home":     212,
            "fifa_rank_away":     212,
            "fifa_points_home":   700.0,
            "fifa_points_away":   700.0,
            "fifa_points_diff":   0.0,
            "fifa_movement_home": 0,
            "fifa_movement_away": 0,
        }

        if is_test_match(tourn, ds):
            test_rows.append(feat)
        else:
            train_rows.append(feat)

        builder.update_elo(home, away, hg, ag, neut, tourn)
        key = (home, away)
        if key not in h2h_records:
            h2h_records[key] = []
        h2h_records[key].append((actual_score(hg, ag), date))

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


# ── Metrik hesaplayıcılar ────────────────────────────────────────────────

def outcome_label(hg, ag):
    if hg > ag: return 2
    if hg == ag: return 1
    return 0


def evaluate_predictions(df, model_home, model_away, label="", rho=0.0):
    X  = df[FEATURE_COLS].values
    lh = model_home.predict(X)
    la = model_away.predict(X)

    y_true        = []
    y_pred_class  = []
    y_proba       = []
    exact_correct = 0
    top3_correct  = 0

    for i, (_, row) in enumerate(df.iterrows()):
        ph, pd_, pa, sh, sa, top3 = poisson_probs(lh[i], la[i], rho)
        true_label = outcome_label(row["home_score"], row["away_score"])
        hg = int(row["home_score"])
        ag = int(row["away_score"])

        y_true.append(true_label)
        y_pred_class.append(np.argmax([pa, pd_, ph]))
        y_proba.append([pa, pd_, ph])

        if sh == hg and sa == ag:
            exact_correct += 1
        if any(h == hg and a == ag for h, a in top3):
            top3_correct += 1

    y_true       = np.array(y_true)
    y_pred_class = np.array(y_pred_class)
    y_proba      = np.array(y_proba)

    accuracy     = (y_true == y_pred_class).mean()
    exact_acc    = exact_correct / len(df)
    top3_acc     = top3_correct  / len(df)
    ll           = log_loss(y_true, y_proba, labels=[0, 1, 2])
    brier        = np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, c])
        for c in range(3)
    ])
    baseline_acc = (y_true == 2).mean()

    n = len(df)
    print(f"\n{'='*55}")
    print(f"  {label}  ({n} maç)")
    print(f"{'='*55}")
    print(f"  Outcome Accuracy   : {accuracy:.3f}   (baseline: {baseline_acc:.3f})")
    print(f"  Exact Score (1.)   : {exact_acc:.3f}")
    print(f"  Exact Score (Top3) : {top3_acc:.3f}")
    print(f"  Log Loss           : {ll:.3f}   (düşük = iyi)")
    print(f"  Brier Score (avg)  : {brier:.3f}  (düşük = iyi)")
    print(f"\n  Tahmin dağılımı:")
    for c, lbl in [(2,"Ev Sahibi"), (1,"Beraberlik"), (0,"Deplasman")]:
        pred_n = (y_pred_class == c).sum()
        true_n = (y_true == c).sum()
        print(f"    {lbl:<12}: tahmin={pred_n:>3}  gerçek={true_n:>3}")

    # Turnuva bazlı özet
    if "tournament" in df.columns:
        print(f"\n  Turnuva bazlı dağılım:")
        for t, grp in df.groupby("tournament"):
            idx = df.index.get_indexer(grp.index)
            acc = (y_true[idx] == y_pred_class[idx]).mean()
            print(f"    {t:<35}: {len(grp):>3} maç  acc={acc:.3f}")

    return {
        "accuracy": accuracy, "exact_acc": exact_acc, "top3_acc": top3_acc,
        "log_loss": ll, "brier": brier, "baseline": baseline_acc,
    }


# ── Ana fonksiyon ────────────────────────────────────────────────────────

def evaluate():
    print("[eval] Veri yükleniyor ve train/test ayrılıyor...")
    train_df, test_df = build_train_test(
        kaggle_path=os.path.join(DATA_DIR, "results.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )
    print(f"[eval] Train: {len(train_df)} maç  |  Test: {len(test_df)} maç")

    if test_df.empty:
        print("[eval] HATA: Test maçı bulunamadı.")
        return

    # Test setindeki turnuva dağılımını göster
    print("\n[eval] Test setindeki turnuvalar:")
    for t, grp in test_df.groupby("tournament"):
        print(f"  {t:<35}: {len(grp)} maç")

    # Modelleri eğit (train.py ile aynı, ayarlanmış hiperparametreler)
    print("\n[eval] Modeller eğitiliyor...")
    model_home = make_xgb_model()
    model_away = make_xgb_model()

    X_train = train_df[FEATURE_COLS].values
    model_home.fit(X_train, train_df["home_score"].values)
    model_away.fit(X_train, train_df["away_score"].values)

    # Dixon-Coles rho'yu sadece train seti üzerinden, zaman ağırlıklı tahmin et
    # (test'e sızıntı olmasın)
    lh_train = model_home.predict(X_train)
    la_train = model_away.predict(X_train)
    days_ago = (train_df["date"].max() - train_df["date"]).dt.days.values
    weights  = time_decay_weights(days_ago)
    rho = estimate_rho(lh_train, la_train, train_df["home_score"].values,
                        train_df["away_score"].values, weights=weights)
    print(f"\n[eval] Dixon-Coles rho (train'den, zaman ağırlıklı tahmin edildi): {rho:.4f}")

    train_metrics = evaluate_predictions(train_df, model_home, model_away, "TRAIN SETİ", rho)
    test_metrics  = evaluate_predictions(test_df,  model_home, model_away, "TEST SETİ — Büyük Turnuvalar (2021-2024)", rho)

    print("\n[eval] Not: Test accuracy > baseline ise model rastgele tahminden iyidir.")
    print("[eval] Futbol tahmininde %50+ accuracy iyi kabul edilir.")

    # backtest.json kaydet
    backtest = {
        "test_tournaments": "2018+ WC+Qual · UEFA Euro · Nations Leagues · Copa América · AFCON · AFC Asian Cup · Gold Cup · CONCACAF NL · OFC",
        "test_matches":     len(test_df),
        "train_matches":    len(train_df),
        "accuracy":         round(float(test_metrics["accuracy"]) * 100, 1),
        "exact_acc":        round(float(test_metrics["exact_acc"]) * 100, 1),
        "top3_acc":         round(float(test_metrics["top3_acc"]) * 100, 1),
        "log_loss":         round(float(test_metrics["log_loss"]), 3),
        "brier":            round(float(test_metrics["brier"]), 3),
        "baseline":         round(float(test_metrics["baseline"]) * 100, 1),
        "train_accuracy":   round(float(train_metrics["accuracy"]) * 100, 1),
        "rho":              round(rho, 4),
        "updated_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    os.makedirs(os.path.dirname(BACKTEST_OUT), exist_ok=True)
    with open(BACKTEST_OUT, "w", encoding="utf-8") as f:
        json.dump(backtest, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Backtest sonuçları kaydedildi: {BACKTEST_OUT}")


if __name__ == "__main__":
    evaluate()
