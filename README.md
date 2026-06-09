# WC 2026 Match Predictor

ELO tabanlı 2026 FIFA Dünya Kupası maç tahmin projesi.

## Veri Kaynakları

| Kaynak | İçerik |
|--------|--------|
| [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 1872'den günümüze tarihsel maç sonuçları |
| [football-data.org](https://www.football-data.org) | 2026 WC takvimi ve canlı sonuçlar |

## Kurulum

```bash
pip install -r requirements.txt
```

Kaggle'dan indirilen CSV'leri `data/` klasörüne koy:
```
data/
├── results.csv
├── shootouts.csv
├── goalscorers.csv
└── former_names.csv
```

API key'ini ortam değişkeni olarak tanımla:
```bash
export FOOTBALL_DATA_API_KEY=your_key_here
```

## Kullanım

### İlk kurulum (tek seferlik)

```bash
# 1. 2026 WC maçlarını çek
python src/fetch_matches.py

# 2. Modeli eğit
python src/train.py

# 3. Tahminleri üret
python src/predict.py
```

### Günlük güncelleme (elle)

```bash
python src/update.py
```

### GitHub Actions (otomatik)

`Settings → Secrets → Actions` altına `FOOTBALL_DATA_API_KEY` secret'ını ekle.
Her gün 08:00 UTC'de otomatik çalışır.

## Model

- **Algoritma:** XGBoost + Platt Scaling (olasılık kalibrasyonu)
- **Hedef:** 3 sınıf — Ev Sahibi Galip / Beraberlik / Deplasman Galip
- **Feature'lar:**

| Feature | Açıklama |
|---------|----------|
| `elo_diff` | İki takım arasındaki ELO farkı |
| `elo_home_adj` | Ev sahibi ELO'su (nötr sahada avantaj eklenmez) |
| `form_home/away` | Son 10 maçtaki ortalama puan |
| `form_diff` | Form farkı |
| `h2h_*` | Kafa kafaya tarihsel win/draw/loss oranları |
| `is_wc` | Dünya Kupası maçı mı? |
| `is_neutral` | Nötr sahada mı? |

## Çıktılar

- `data/matches_2026.csv` — API'den çekilen maç takvimi + sonuçlar
- `data/predictions.csv` — Güncel tahminler (olasılıklar dahil)
- `models/model.pkl` — Eğitilmiş model
