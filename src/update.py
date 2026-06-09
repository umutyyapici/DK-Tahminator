"""
update.py
---------
GitHub Action tarafından günlük çalıştırılan ana script.
Sırayla:
  1. football-data.org'dan yeni maç sonuçlarını çeker
  2. Modeli yeniden eğitir (ELO ve feature'lar güncellenir)
  3. Tahminleri yeniden üretir
"""

import sys
import os

# src/ altındaki modülleri import edebilmek için
sys.path.insert(0, os.path.dirname(__file__))

from fetch_matches import fetch_wc2026_matches, save_matches
from train import train
from predict import predict


def main():
    print("=" * 50)
    print("ADIM 1/3: Maç sonuçları çekiliyor...")
    print("=" * 50)
    df = fetch_wc2026_matches()
    save_matches(df)

    print("\n" + "=" * 50)
    print("ADIM 2/3: Model yeniden eğitiliyor...")
    print("=" * 50)
    train()

    print("\n" + "=" * 50)
    print("ADIM 3/3: Tahminler üretiliyor...")
    print("=" * 50)
    predict()

    print("\n✓ Güncelleme tamamlandı.")


if __name__ == "__main__":
    main()
