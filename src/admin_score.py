"""
admin_score.py
--------------
Admin'in maç skorlarını manuel olarak girmesi için CLI aracı.
Girilen skor data/admin_scores.json'a kaydedilir; bir sonraki
pipeline çalışmasında API verisi yerine bu skor kullanılır.

Kullanım:
    python src/admin_score.py \\
        --date 2026-07-01 \\
        --home "South Africa" \\
        --away "Canada" \\
        --home-score 2 \\
        --away-score 1 \\
        --note "API gecikmesi"

Silmek için --delete flag'ini kullan:
    python src/admin_score.py --date 2026-07-01 --home "South Africa" --away "Canada" --delete
"""

import json
import os
import argparse
from datetime import datetime, timezone

DATA_DIR          = os.path.join(os.path.dirname(__file__), "..", "data")
ADMIN_SCORES_PATH = os.path.join(DATA_DIR, "admin_scores.json")


def _load() -> dict:
    if os.path.exists(ADMIN_SCORES_PATH):
        with open(ADMIN_SCORES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(overrides: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ADMIN_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Admin manuel skor girişi")
    p.add_argument("--date",       required=True,  help="Maç tarihi (YYYY-MM-DD)")
    p.add_argument("--home",       required=True,  help="Ev sahibi takım adı")
    p.add_argument("--away",       required=True,  help="Deplasman takımı adı")
    p.add_argument("--home-score", type=int,       dest="home_score", help="Ev sahibi gol sayısı")
    p.add_argument("--away-score", type=int,       dest="away_score", help="Deplasman gol sayısı")
    p.add_argument("--note",       default="",     help="Not (opsiyonel)")
    p.add_argument("--delete",     action="store_true", help="Bu maçın override'ını sil")
    args = p.parse_args()

    key = f"{args.date}|{args.home}|{args.away}"
    overrides = _load()

    if args.delete:
        if key in overrides:
            del overrides[key]
            _save(overrides)
            print(f"[admin] Override silindi: {key}")
        else:
            print(f"[admin] Override bulunamadı: {key}")
        return

    if args.home_score is None or args.away_score is None:
        p.error("--home-score ve --away-score zorunludur (--delete kullanılmıyorsa)")

    overrides[key] = {
        "home_score": args.home_score,
        "away_score": args.away_score,
        "status":     "FINISHED",
        "note":       args.note,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _save(overrides)
    print(f"[admin] Kaydedildi: {key} → {args.home_score}-{args.away_score}")
    print(f"[admin] Not: Değişikliğin etkili olması için pipeline'ın çalışması gerekir.")


if __name__ == "__main__":
    main()
