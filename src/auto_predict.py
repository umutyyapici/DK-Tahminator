"""
auto_predict.py
---------------
DK-Tahminator Supabase oyununa otomatik tahmin girer.
update.py tarafından çağrılır.

Mantık:
  1. predictions.csv'den bugün ve sonraki oynanmamış maçları al
  2. Supabase matches tablosundan match_id'leri eşleştir
  3. Beklenen puanı hesapla, en yükseğine joker oyna
  4. predictions tablosuna upsert et
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")  # service_role key
BOT_USER_ID  = os.environ.get("BOT_USER_ID", "")   # senin user_id'n

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")


def supabase_get(table: str, params: dict = {}) -> list:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params={"select": "*", **params},
        timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def supabase_upsert(table: str, rows: list) -> None:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        json=rows,
        timeout=15
    )
    resp.raise_for_status()


def calc_expected_points(pred_h: int, pred_a: int,
                         lambda_h: float, lambda_a: float,
                         max_g: int = 9) -> float:
    """
    Poisson dağılımıyla beklenen puan hesabı.
    Puanlama: tam isabet=6, kıl payı=3, stratejist=2, bilge=1, teselli=1
    """
    import math

    def pmf(k, lam):
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    expected = 0.0
    pred_diff = pred_h - pred_a
    pred_result = "home" if pred_h > pred_a else ("away" if pred_h < pred_a else "draw")

    for ah in range(max_g):
        for aa in range(max_g):
            p = pmf(ah, max(0.01, lambda_h)) * pmf(aa, max(0.01, lambda_a))
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


def normalize_name(name: str) -> str:
    """Basit normalizasyon — büyük/küçük harf ve boşluk."""
    return name.strip().lower() if name else ""


def auto_predict():
    if not SUPABASE_URL or not SUPABASE_KEY or not BOT_USER_ID:
        print("[auto_predict] SUPABASE_URL, SUPABASE_KEY veya BOT_USER_ID eksik, atlanıyor.")
        return

    # 1. predictions.csv'yi yükle
    preds_path = os.path.join(DATA_DIR, "predictions.csv")
    if not os.path.exists(preds_path):
        print("[auto_predict] predictions.csv bulunamadı.")
        return

    preds_df = pd.read_csv(preds_path)
    preds_df["date"] = pd.to_datetime(preds_df["date"]).dt.date

    # Bugün ve sonrasındaki maçları al (oynanmamışlar)
    today = datetime.now(timezone.utc).date()
    upcoming = preds_df[preds_df["date"] >= today].copy()

    if upcoming.empty:
        print("[auto_predict] Girilecek tahmin yok.")
        return

    print(f"[auto_predict] {len(upcoming)} maç için tahmin girilecek.")

    # 2. Supabase'den tüm maçları çek
    # locked=false filtresi yerine Python'da kontrol ediyoruz
    # çünkü bahis 1 saat önce kapanıyor ama locked henüz false olabilir
    all_matches = supabase_get("matches", {"order": "match_datetime.asc"})
    if not all_matches:
        print("[auto_predict] Supabase'de maç bulunamadı.")
        return

    # Bahis hâlâ açık maçları filtrele (maçtan 1 saat öncesine kadar açık)
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(hours=1)  # 1 saat sonrasına kadar olan maçlara gir
    sb_matches = []
    for m in all_matches:
        if m.get("locked"):
            continue
        dt_str = m.get("match_datetime", "")
        if not dt_str:
            continue
        try:
            from datetime import datetime as dt
            match_dt = dt.fromisoformat(dt_str.replace("Z", "+00:00"))
            if match_dt > cutoff:
                sb_matches.append(m)
        except Exception:
            sb_matches.append(m)  # parse edilemezse dahil et

    if not sb_matches:
        print("[auto_predict] Bahis açık maç yok.")
        return

    # 3. predictions.csv ile Supabase maçlarını eşleştir
    to_upsert = []
    joker_candidate = None  # (match_id, expected_pts)

    for _, row in upcoming.iterrows():
        pred_h = int(round(float(row["most_likely_score"].split("-")[0])))
        pred_a = int(round(float(row["most_likely_score"].split("-")[1])))
        lambda_h = float(row.get("expected_home", 1.2))
        lambda_a = float(row.get("expected_away", 1.0))
        date_str = str(row["date"])

        # Supabase'de eşleşen maçı bul
        match_id = None
        for m in sb_matches:
            m_home = normalize_name(m.get("home_team", ""))
            m_away = normalize_name(m.get("away_team", ""))
            p_home = normalize_name(row["home_team"])
            p_away = normalize_name(row["away_team"])
            m_date = str(m.get("match_datetime", ""))[:10]

            if m_home == p_home and m_away == p_away and m_date == date_str:
                match_id = m["id"]
                break

        if match_id is None:
            print(f"[auto_predict] Eşleşme bulunamadı: {row['home_team']} vs {row['away_team']} ({date_str})")
            continue

        # Beklenen puanı hesapla
        exp_pts = calc_expected_points(pred_h, pred_a, lambda_h, lambda_a)

        to_upsert.append({
            "match_id": match_id,
            "pred_h":   pred_h,
            "pred_a":   pred_a,
            "exp_pts":  exp_pts,
        })

        # Joker adayı — en yüksek beklenen puan
        if joker_candidate is None or exp_pts > joker_candidate[1]:
            joker_candidate = (match_id, exp_pts)

    if not to_upsert:
        print("[auto_predict] Eşleşen maç bulunamadı.")
        return

    # 4. Upsert
    rows = []
    for item in to_upsert:
        is_joker = (joker_candidate and item["match_id"] == joker_candidate[0])
        rows.append({
            "user_id":   BOT_USER_ID,
            "match_id":  item["match_id"],
            "pred_home": item["pred_h"],
            "pred_away": item["pred_a"],
            "is_joker":  is_joker,
        })

    supabase_upsert("predictions", rows)

    joker_match = next((r for r in rows if r["is_joker"]), None)
    print(f"[auto_predict] {len(rows)} tahmin girildi.")
    if joker_match:
        m = next((m for m in sb_matches if m["id"] == joker_match["match_id"]), {})
        print(f"[auto_predict] Joker → {m.get('home_team','')} vs {m.get('away_team','')} "
              f"({joker_match['pred_home']}-{joker_match['pred_away']}, "
              f"E[puan]={joker_candidate[1]:.2f})")


if __name__ == "__main__":
    auto_predict()
