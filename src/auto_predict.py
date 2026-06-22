"""
auto_predict.py
---------------
Her gün için en yüksek beklenen puanlı maçı "joker" olarak işaretler ve
predictions.csv'ye `is_joker` sütunu olarak yazar.

Joker kilitleme kuralı:
  - Bir günün joker maçı FINISHED statüsüne geçtikten sonra o gün için
    joker yeniden atanmaz. Seçim joker_locks.json'da kalıcı olarak saklanır.
  - Joker maçı henüz oynanmamışsa, model güncellemelerinde joker
    yeniden hesaplanabilir.
"""

import os
import json
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

_TR = ZoneInfo("Europe/Istanbul")

from poisson_model import score_matrix, load_rho, MAX_GOALS

DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
MATCHES_PATH     = os.path.join(DATA_DIR, "matches_2026.csv")
JOKER_LOCKS_PATH = os.path.join(DATA_DIR, "joker_locks.json")


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
    pred_diff   = pred_h - pred_a
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
                if pred_h == ah or pred_a == aa:
                    expected += p * 1
                continue

            if pred_h == ah and pred_a == aa:
                pts = 6
            elif pred_h == ah or pred_a == aa:
                pts = 3
            elif pred_diff == actual_diff:
                pts = 2
            else:
                pts = 1

            expected += p * pts

    return expected


def _load_joker_locks() -> dict:
    if os.path.exists(JOKER_LOCKS_PATH):
        with open(JOKER_LOCKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_joker_locks(locks: dict) -> None:
    with open(JOKER_LOCKS_PATH, "w", encoding="utf-8") as f:
        json.dump(locks, f, ensure_ascii=False, indent=2)


def _finished_pairs() -> set:
    """matches_2026.csv'den FINISHED maç (home_team, away_team) çiftlerini döndürür."""
    if not os.path.exists(MATCHES_PATH):
        return set()
    df = pd.read_csv(MATCHES_PATH)
    finished = df[df["status"] == "FINISHED"]
    return set(zip(finished["home_team"], finished["away_team"]))


def _tr_date(row) -> str:
    kickoff = str(row.get("kickoff_utc", ""))
    if kickoff and kickoff != "nan":
        try:
            dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            return str(dt.astimezone(_TR).date())
        except ValueError:
            pass
    return str(row["date"])


def auto_predict():
    if not os.path.exists(PREDICTIONS_PATH):
        print("[auto_predict] predictions.csv bulunamadı.")
        return

    preds_df = pd.read_csv(PREDICTIONS_PATH)
    preds_df["date"] = pd.to_datetime(preds_df["date"]).dt.date

    today    = datetime.now(timezone.utc).date()
    upcoming = preds_df[preds_df["date"] >= today]

    if upcoming.empty:
        print("[auto_predict] Joker belirlenecek maç yok.")
        preds_df["is_joker"] = False
        preds_df.to_csv(PREDICTIONS_PATH, index=False)
        return

    rho          = load_rho()
    joker_locks  = _load_joker_locks()
    finished     = _finished_pairs()

    # Oynanmış joker maçlarını kilitle
    for date_str, lock in joker_locks.items():
        if not lock.get("played", False):
            pair = (lock["home_team"], lock["away_team"])
            if pair in finished:
                lock["played"] = True
                print(f"[auto_predict] {date_str} joker kilitlendi (oynanmış): "
                      f"{lock['home_team']} vs {lock['away_team']}")

    # Gün bazında beklenen puan hesapla
    by_date = defaultdict(list)
    for idx, row in upcoming.iterrows():
        pred_h   = int(round(float(row["most_likely_score"].split("-")[0])))
        pred_a   = int(round(float(row["most_likely_score"].split("-")[1])))
        lambda_h = float(row.get("expected_home", 1.2))
        lambda_a = float(row.get("expected_away", 1.0))
        exp_pts  = calc_expected_points(pred_h, pred_a, lambda_h, lambda_a, rho)
        by_date[_tr_date(row)].append((idx, exp_pts))

    preds_df["is_joker"] = False

    for date_str, items in sorted(by_date.items()):
        lock = joker_locks.get(date_str)

        # Joker maçı oynanmışsa bu gün için yeni joker atama
        if lock and lock.get("played", False):
            print(f"[auto_predict] {date_str} joker kilitli (oynanmış), "
                  f"değiştirilmiyor: {lock['home_team']} vs {lock['away_team']}")
            continue

        # Joker maçı henüz oynanmamışsa en yüksek beklenen puanlıyı seç
        joker_idx, joker_pts = max(items, key=lambda x: x[1])
        joker_match = preds_df.loc[joker_idx]
        joker_locks[date_str] = {
            "home_team": joker_match["home_team"],
            "away_team": joker_match["away_team"],
            "played":    False,
        }
        preds_df.loc[joker_idx, "is_joker"] = True
        print(f"[auto_predict] {date_str} joker -> "
              f"{joker_match['home_team']} vs {joker_match['away_team']} "
              f"(E[puan]={joker_pts:.2f})")

    _save_joker_locks(joker_locks)
    preds_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"[auto_predict] {len(by_date)} gün için joker işaretlendi, "
          f"predictions.csv ve joker_locks.json güncellendi.")


if __name__ == "__main__":
    auto_predict()
