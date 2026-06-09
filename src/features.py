"""
features.py
-----------
1. Kaggle results.csv + data/matches_2026.csv'yi birleştirir
2. former_names.csv ile takım adı normalizasyonu yapar
3. Her maç için ELO tabanlı feature'lar üretir
4. Eğitim dataseti ve 2026 tahmin dataseti döndürür
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# --- Sabitler -----------------------------------------------------------

INITIAL_ELO    = 1500.0
K_BASE         = 32        # standart maçlar
K_WC           = 60        # Dünya Kupası maçları (daha kritik)
K_FRIENDLY     = 20        # hazırlık maçları
HOME_ADVANTAGE = 100       # nötr sahada uygulanmaz

TOURNAMENT_K = {
    "FIFA World Cup":             K_WC,
    "Confederations Cup":         50,
    "UEFA Euro":                  50,
    "Copa América":               50,
    "Africa Cup of Nations":      50,
    "AFC Asian Cup":              50,
    "Friendly":                   K_FRIENDLY,
    "UEFA Nations League":        35,
}

FORM_WINDOW = 10           # son kaç maç forma hesabında kullanılır


# --- Yardımcı fonksiyonlar ----------------------------------------------

def get_k(tournament: str) -> float:
    for key, k in TOURNAMENT_K.items():
        if key.lower() in tournament.lower():
            return k
    return K_BASE


def expected_score(elo_a: float, elo_b: float) -> float:
    """A takımının beklenen skoru (0-1 arası)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def actual_score(home_goals: int, away_goals: int) -> float:
    """Ev sahibi açısından sonuç: 1=galibiyet, 0.5=beraberlik, 0=mağlubiyet."""
    if home_goals > away_goals:
        return 1.0
    elif home_goals == away_goals:
        return 0.5
    return 0.0


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Büyük farkla kazanmak daha fazla ELO değişimine yol açar.
    FIFA'nın kullandığı yaklaşıma benzer.
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return 1.75 + (gd - 3) * 0.1


# --- Takım adı normalizasyonu -------------------------------------------

def build_name_map(former_names_path: str) -> dict:
    """
    former_names.csv'den {eski_ad: güncel_ad} sözlüğü oluşturur.
    Tarih aralığı göz ardı edilir; güncel ada map'leme yeterli.
    """
    if not os.path.exists(former_names_path):
        return {}
    df = pd.read_csv(former_names_path)
    return dict(zip(df["former"], df["current"]))


def normalize_team(name: str, name_map: dict) -> str:
    return name_map.get(name, name)


# --- Ana sınıf ----------------------------------------------------------

class EloFeatureBuilder:
    """
    Tarihsel maçları kronolojik sırayla işleyerek:
    - Her takım için ELO rating tutar
    - Her takım için son N maç formunu tutar
    - Her maç için feature vektörü üretir
    """

    def __init__(self):
        self.elos: dict[str, float] = {}
        self.history: dict[str, list] = {}   # takım → [sonuçlar]
        self.elo_history: dict[str, list] = {}  # takım → [elo değerleri]

    def get_elo(self, team: str) -> float:
        return self.elos.get(team, INITIAL_ELO)

    def update_elo(self, home: str, away: str, home_goals: int,
                   away_goals: int, neutral: bool, tournament: str) -> Tuple[float, float]:
        """ELO'ları günceller, eski değerleri döndürür (feature için)."""
        k = get_k(tournament)
        ha = 0 if neutral else HOME_ADVANTAGE

        elo_h = self.get_elo(home)
        elo_a = self.get_elo(away)

        exp_h = expected_score(elo_h + ha, elo_a)
        act_h = actual_score(home_goals, away_goals)
        mult  = goal_diff_multiplier(home_goals - away_goals)

        delta = k * mult * (act_h - exp_h)

        self.elos[home] = elo_h + delta
        self.elos[away] = elo_a - delta

        # elo geçmişi güncelle
        self._update_elo_history(home, elo_h)
        self._update_elo_history(away, elo_a)

        # forma güncelle
        self._update_form(home, act_h)
        self._update_form(away, 1.0 - act_h)

        return elo_h, elo_a  # maç öncesi değerler

    def _update_form(self, team: str, result: float):
        if team not in self.history:
            self.history[team] = []
        self.history[team].append(result)

    def _update_elo_history(self, team: str, elo: float):
        if team not in self.elo_history:
            self.elo_history[team] = []
        self.elo_history[team].append(elo)

    def get_elo_momentum(self, team: str, window: int = 5) -> float:
        """Son N maçtaki ELO değişimi. Pozitif = yükselen, negatif = düşen."""
        hist = self.elo_history.get(team, [])
        if len(hist) < 2:
            return 0.0
        recent = hist[-window:]
        return float(recent[-1] - recent[0])

    def get_form(self, team: str, window: int = FORM_WINDOW) -> float:
        """Son N maçtaki ortalama puan (0-1 arası)."""
        hist = self.history.get(team, [])
        if not hist:
            return 0.5
        return float(np.mean(hist[-window:]))

    def get_h2h(self, home: str, away: str,
                h2h_records: dict) -> Tuple[float, float, float]:
        """
        İki takım arasındaki kafa kafaya win/draw/loss oranları.
        h2h_records: {(home, away): [sonuçlar]} sözlüğü
        """
        key = (home, away)
        key_rev = (away, home)

        results = []
        for r in h2h_records.get(key, []):
            results.append(r)          # 1=home win, 0.5=draw, 0=away win
        for r in h2h_records.get(key_rev, []):
            results.append(1.0 - r)   # ters çevir

        if not results:
            return 0.33, 0.33, 0.34   # no data → uniform

        wins   = sum(1 for r in results if r == 1.0) / len(results)
        draws  = sum(1 for r in results if r == 0.5) / len(results)
        losses = 1.0 - wins - draws
        return wins, draws, losses


# --- Pipeline -----------------------------------------------------------

def build_features(kaggle_results_path: str,
                   wc2026_path: str,
                   former_names_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Döndürür:
        train_df  — home_score ve away_score bilinen tüm tarihsel maçlar + feature'lar
        pred_df   — 2026 WC'de henüz oynanmamış maçlar + feature'lar
    """
    # 1. Veri yükle
    hist = pd.read_csv(kaggle_results_path, parse_dates=["date"])

    wc2026 = pd.DataFrame()
    if os.path.exists(wc2026_path):
        wc2026 = pd.read_csv(wc2026_path, parse_dates=["date"])

    name_map = build_name_map(former_names_path)

    # 2. Birleştir
    if not wc2026.empty:
        # Sadece oynanmış WC2026 maçlarını tarihsel veriye ekle
        played = wc2026[wc2026["status"] == "FINISHED"][
            ["date", "home_team", "away_team", "home_score",
             "away_score", "tournament", "neutral"]
        ].copy()
        hist = pd.concat([hist, played], ignore_index=True)

    hist.sort_values("date", inplace=True)
    hist.reset_index(drop=True, inplace=True)

    # 3. Takım adlarını normalize et
    for col in ["home_team", "away_team"]:
        hist[col] = hist[col].map(lambda x: normalize_team(x, name_map))
        if not wc2026.empty:
            wc2026[col] = wc2026[col].map(lambda x: normalize_team(x, name_map))

    # 4. neutral sütununu bool'a çevir
    hist["neutral"] = hist["neutral"].map(
        lambda x: True if str(x).upper() == "TRUE" else False
    )

    # 5. ELO + feature'ları hesapla
    builder = EloFeatureBuilder()
    h2h_records: dict = {}

    feature_rows = []

    for _, row in hist.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        home  = row["home_team"]
        away  = row["away_team"]
        hg    = int(row["home_score"])
        ag    = int(row["away_score"])
        neut  = bool(row["neutral"])
        tourn = str(row["tournament"])

        # Maç öncesi feature'lar (güncelleme öncesi alınır)
        elo_h = builder.get_elo(home)
        elo_a = builder.get_elo(away)
        form_h = builder.get_form(home)
        form_a = builder.get_form(away)
        mom_h = builder.get_elo_momentum(home)
        mom_a = builder.get_elo_momentum(away)
        h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records)

        # Hedef değişken: 2=ev galip, 1=beraberlik, 0=deplasman galip
        if hg > ag:
            outcome = 2
        elif hg == ag:
            outcome = 1
        else:
            outcome = 0

        feature_rows.append({
            "date":          row["date"],
            "home_team":     home,
            "away_team":     away,
            "home_score":    hg,
            "away_score":    ag,
            "tournament":    tourn,
            "neutral":       neut,
            # --- Feature'lar ---
            "elo_home":      elo_h,
            "elo_away":      elo_a,
            "elo_diff":      elo_h - elo_a,
            "elo_home_adj":  elo_h + (0 if neut else HOME_ADVANTAGE),
            "form_home":     form_h,
            "form_away":     form_a,
            "form_diff":     form_h - form_a,
            "h2h_home_win":  h2h_w,
            "h2h_draw":      h2h_d,
            "h2h_away_win":  h2h_l,
            "is_wc":         1 if "world cup" in tourn.lower() else 0,
            "is_neutral":    1 if neut else 0,
            "elo_momentum_home": mom_h,
            "elo_momentum_away": mom_a,
            "elo_momentum_diff": mom_h - mom_a,
            "outcome":       outcome,
        })

        # ELO + H2H güncelle
        builder.update_elo(home, away, hg, ag, neut, tourn)
        key = (home, away)
        if key not in h2h_records:
            h2h_records[key] = []
        h2h_records[key].append(actual_score(hg, ag))

    train_df = pd.DataFrame(feature_rows)

    # 6. 2026 tahmin feature'ları (oynanmamış maçlar)
    pred_rows = []
    if not wc2026.empty:
        unplayed = wc2026[wc2026["status"] != "FINISHED"].copy()
        for _, row in unplayed.iterrows():
            home  = row["home_team"]
            away  = row["away_team"]
            neut  = True  # WC nötr saha

            elo_h  = builder.get_elo(home)
            elo_a  = builder.get_elo(away)
            form_h = builder.get_form(home)
            form_a = builder.get_form(away)
            mom_h  = builder.get_elo_momentum(home)
            mom_a  = builder.get_elo_momentum(away)
            h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records)

            pred_rows.append({
                "date":         row["date"],
                "home_team":    home,
                "away_team":    away,
                "stage":        row.get("stage", ""),
                "match_id":     row.get("match_id", ""),
                "elo_home":     elo_h,
                "elo_away":     elo_a,
                "elo_diff":     elo_h - elo_a,
                "elo_home_adj": elo_h,   # nötr saha, avantaj yok
                "form_home":    form_h,
                "form_away":    form_a,
                "form_diff":    form_h - form_a,
                "h2h_home_win": h2h_w,
                "h2h_draw":     h2h_d,
                "h2h_away_win": h2h_l,
                "is_wc":        1,
                "is_neutral":   1,
                "elo_momentum_home": mom_h,
                "elo_momentum_away": mom_a,
                "elo_momentum_diff": mom_h - mom_a,
            })

    pred_df = pd.DataFrame(pred_rows)
    return train_df, pred_df


FEATURE_COLS = [
    "elo_diff", "elo_home_adj", "elo_away",
    "form_home", "form_away", "form_diff",
    "h2h_home_win", "h2h_draw", "h2h_away_win",
    "is_wc", "is_neutral",
    "elo_momentum_home", "elo_momentum_away", "elo_momentum_diff",
]


if __name__ == "__main__":
    train, pred = build_features(
        kaggle_results_path=os.path.join(DATA_DIR, "results.csv"),
        wc2026_path=os.path.join(DATA_DIR, "matches_2026.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )
    print(f"Eğitim seti: {len(train)} maç")
    print(f"Tahmin seti: {len(pred)} maç")
    print(train[["date", "home_team", "away_team", "elo_diff", "outcome"]].tail())
