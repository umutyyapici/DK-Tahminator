"""
fetch_matches.py
----------------
football-data.org API'sinden 2026 Dünya Kupası maçlarını çeker.
Hem takvimi (gelecek maçlar) hem de sonuçları (oynanan maçlar) alır.
Kaggle results.csv formatına dönüştürüp data/matches_2026.csv'ye yazar.
"""

import os
import requests
import pandas as pd
from datetime import datetime

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
WC_CODE = "WC"  # football-data.org'daki Dünya Kupası kodu
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "matches_2026.csv")


def fetch_wc2026_matches() -> pd.DataFrame:
    """
    Tüm 2026 WC maçlarını çeker (oynanmış + oynanacak).
    Döndürülen DataFrame Kaggle results.csv formatındadır:
    date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
    """
    url = f"{BASE_URL}/competitions/{WC_CODE}/matches"
    
    # Sezon filtresini 2026 olarak netleştiriyoruz
    params = {"season": 2026}

    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for m in data.get("matches", []):
        status = m.get("status", "")  # SCHEDULED, IN_PLAY, FINISHED vs.

        home_score = None
        away_score = None
        if status == "FINISHED":
            ft = m.get("score", {}).get("fullTime", {})
            home_score = ft.get("home")
            away_score = ft.get("away")

        # API'den bazen boş takım gelebilir (üst turlar henüz netleşmediyse).
        # homeTeam nesnesi var ama name null olabilir → or {} ile her iki formatı kapat.
        home_name = (m.get("homeTeam") or {}).get("name") or "Belli Değil"
        away_name = (m.get("awayTeam") or {}).get("name") or "Belli Değil"

        rows.append({
            "date":       m["utcDate"][:10],           # "2026-06-11"
            "kickoff_utc": m["utcDate"],                # "2026-06-11T22:00:00Z"
            "home_team":  home_name,
            "away_team":  away_name,
            "home_score": home_score,                  # None = oynanmadı
            "away_score": away_score,
            "tournament": "FIFA World Cup",
            "city":       m.get("venue", ""),
            "country":    "USA/Mexico/Canada",
            "neutral":    True,                        # WC nötr sahada oynanır
            "status":     status,
            "stage":      m.get("stage", ""),          # GROUP_STAGE, ROUND_OF_16 vs.
            "matchday":   m.get("matchday"),
            "match_id":   m["id"],                     # football-data.org ID'si
        })

    if not rows:
        raise ValueError("API'den maç verisi çekilemedi. Sezon parametresini veya API iznini kontrol edin.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_matches(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(OUTPUT_PATH, index=False)
    json_path = OUTPUT_PATH.replace(".csv", ".json")
    df.to_json(json_path, orient="records", force_ascii=False)
    finished = df[df["status"] == "FINISHED"].shape[0]
    scheduled = df[df["status"] == "SCHEDULED"].shape[0]
    print(f"[fetch] {len(df)} maç kaydedildi → {finished} oynanmış, {scheduled} planlanmış")
    print(f"[fetch] Dosya: {OUTPUT_PATH}")


if __name__ == "__main__":
    # Sadece API_KEY tamamen boş kalmışsa hata versin
    if not API_KEY:
        raise EnvironmentError(
            "FOOTBALL_DATA_API_KEY tanımlı değil veya kodun içine yazılmadı.\n"
            "Lütfen kodu düzenleyin veya ortam değişkenini tanımlayın."
        )
    df = fetch_wc2026_matches()
    save_matches(df)
    print(df[["date", "home_team", "away_team", "home_score", "away_score", "status"]].to_string())
