"""
fetch_rankings.py
-----------------
FIFA Men's World Ranking'i api.fifa.com uzerinden ceker.
Public endpoint, kimlik dogrulamasi gerekmez.
data/fifa_rankings.json olarak kaydeder.
"""

import os
import json
import requests

FIFA_RANKING_URL = (
    "https://api.fifa.com/api/v3/fifarankings/rankings/live"
    "?gender=1&sportType=0&language=en"
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fifa_rankings.json")

# FIFA resmi adi --> projedeki ic takim adi
FIFA_NAME_MAP = {
    "Korea Republic":          "South Korea",
    "IR Iran":                 "Iran",
    "USA":                     "United States",
    "Turkiye":                 "Turkey",
    "Ürkiye":             "Turkey",
    "Türkiye":            "Turkey",
    "DPR Korea":               "North Korea",
    "Congo DR":                "DR Congo",
    "Côte d'Ivoire":      "Ivory Coast",
    "Cabo Verde":              "Cape Verde",
    "The Gambia":              "Gambia",
    "Czechia":                 "Czech Republic",
    "Palestine":               "Palestine",
    "Republic of Ireland":     "Republic of Ireland",
    "Bosnia and Herzegovina":  "Bosnia and Herzegovina",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://inside.fifa.com/",
    "Origin":  "https://inside.fifa.com",
}


def fetch_fifa_rankings() -> dict:
    """
    Guncel FIFA Men's siralamasi.
    Doendueruer: {takim_adi: {rank, points, prev_rank, prev_points, movement}}
    """
    resp = requests.get(FIFA_RANKING_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rankings = {}
    for entry in data.get("Results", []):
        name = None
        for n in entry.get("TeamName", []):
            if "en" in n.get("Locale", "").lower():
                name = n.get("Description")
                break
        if not name:
            names = entry.get("TeamName", [])
            if names:
                name = names[0].get("Description", "")
        if not name:
            continue

        internal = FIFA_NAME_MAP.get(name, name)
        rankings[internal] = {
            "rank":        int(entry.get("Rank", 999)),
            "points":      round(float(entry.get("TotalPoints", 0)), 4),
            "prev_rank":   int(entry.get("PrevRank", 999)),
            "prev_points": round(float(entry.get("PrevPoints", 0)), 4),
            "movement":    int(entry.get("RankingMovement", 0)),
        }

    return rankings


def save_rankings(rankings: dict) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False, indent=2)
    print(f"[rankings] {len(rankings)} takim siralama kaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    r = fetch_fifa_rankings()
    save_rankings(r)
    print("\nIlk 15 siralama:")
    for team, info in list(r.items())[:15]:
        print(f"  {info['rank']:>3}. {team:<30} {info['points']:.1f} puan  ({info['movement']:+d})")
