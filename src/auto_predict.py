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

from poisson_model import score_matrix, load_rho, MAX_GOALS

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
        f"{SUPABASE_URL}/rest/v1/{table}?on_conflict=user_id,match_id",
        headers=headers,
        json=rows,
        timeout=15
    )
    if not resp.ok:
        print(f"[auto_predict] Supabase HATA detayı: {resp.status_code} {resp.text}")
    resp.raise_for_status()


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


def normalize_name(name: str) -> str:
    """Basit normalizasyon — büyük/küçük harf ve boşluk."""
    return name.strip().lower() if name else ""


def auto_predict():
    print("[auto_predict] VERSION: daily_joker_v2")
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

    rho = load_rho()

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
    # Günlük gruplama: her gün için ayrı joker
    from collections import defaultdict
    by_date = defaultdict(list)

    for _, row in upcoming.iterrows():
        pred_h = int(round(float(row["most_likely_score"].split("-")[0])))
        pred_a = int(round(float(row["most_likely_score"].split("-")[1])))
        lambda_h = float(row.get("expected_home", 1.2))
        lambda_a = float(row.get("expected_away", 1.0))
        date_str = str(row["date"])

        # Supabase'de eşleşen maçı bul
        # Not: tarihler farklı zaman dilimlerinden gelebilir (UTC vs TR),
        # bu yüzden takım isimleri eşleşirse ±1 günlük farkı tolere ediyoruz.
        match_id = None
        p_home = normalize_name(row["home_team"])
        p_away = normalize_name(row["away_team"])
        try:
            p_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            p_date = None

        for m in sb_matches:
            m_home = normalize_name(m.get("home_team", ""))
            m_away = normalize_name(m.get("away_team", ""))
            if m_home != p_home or m_away != p_away:
                continue

            m_date_str = str(m.get("match_datetime", ""))[:10]
            if p_date is None:
                match_id = m["id"]
                break
            try:
                m_date = datetime.strptime(m_date_str, "%Y-%m-%d").date()
                if abs((m_date - p_date).days) <= 1:
                    match_id = m["id"]
                    break
            except ValueError:
                match_id = m["id"]
                break

        if match_id is None:
            print(f"[auto_predict] Eşleşme bulunamadı: {row['home_team']} vs {row['away_team']} ({date_str})")
            continue

        exp_pts = calc_expected_points(pred_h, pred_a, lambda_h, lambda_a, rho)
        by_date[date_str].append({
            "match_id": match_id,
            "pred_h":   pred_h,
            "pred_a":   pred_a,
            "exp_pts":  exp_pts,
            "date":     date_str,
        })

    if not by_date:
        print("[auto_predict] Eşleşen maç bulunamadı.")
        return

    # 4. Her gün için joker seç ve upsert listesi oluştur
    rows = []
    for date_str, items in sorted(by_date.items()):
        # O günün en yüksek beklenen puanlı maçına joker
        joker_id = max(items, key=lambda x: x["exp_pts"])["match_id"]
        for item in items:
            rows.append({
                "user_id":   BOT_USER_ID,
                "match_id":  item["match_id"],
                "pred_home": item["pred_h"],
                "pred_away": item["pred_a"],
                "is_joker":  item["match_id"] == joker_id,
            })
        joker_match = next(m for m in sb_matches if m["id"] == joker_id)
        joker_item  = next(i for i in items if i["match_id"] == joker_id)
        print(f"[auto_predict] {date_str} joker → {joker_match.get('home_team','')} vs {joker_match.get('away_team','')} (E[puan]={joker_item['exp_pts']:.2f})")

    supabase_upsert("predictions", rows)

    print(f"[auto_predict] {len(rows)} tahmin girildi, {sum(1 for r in rows if r['is_joker'])} joker.")


if __name__ == "__main__":
    auto_predict()
