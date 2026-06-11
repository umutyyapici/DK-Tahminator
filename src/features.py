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
K_BASE         = 32        # tanımlı olmayan turnuvalar / küçük davetiyeli kupalar
K_WC           = 60        # Dünya Kupası finalleri (en üst düzey)
K_FRIENDLY     = 20        # hazırlık maçları
HOME_ADVANTAGE = 100       # nötr sahada uygulanmaz

# Turnuva → K katsayısı, rekabet seviyesine göre kademeli (tam eşleşme).
# Önceki sürümde alt-dize eşleşmesi kullanılıyordu; bu da örn.
# "FIFA World Cup qualification"in "FIFA World Cup" ile eşleşip final
# maçlarıyla AYNI (en yüksek) katsayıyı almasına yol açıyordu. Artık her
# turnuva adı kendi katsayısını alıyor; eşleşmeyenler K_BASE'e düşer.
TOURNAMENT_K = {
    # --- Tier S: Dünya Kupası finalleri -------------------------------
    "FIFA World Cup":                       K_WC,

    # --- Tier A: Kıtasal şampiyonalar / Konfederasyonlar Kupası -------
    "UEFA Euro":                            50,
    "Copa América":                         50,
    "African Cup of Nations":               50,
    "AFC Asian Cup":                        50,
    "Confederations Cup":                   50,

    # --- Tier B: Dünya Kupası elemeleri (yüksek rekabet, final değil) -
    "FIFA World Cup qualification":         40,

    # --- Tier C: Avrupa elemeleri / UEFA Uluslar Ligi -----------------
    "UEFA Euro qualification":              35,
    "UEFA Nations League":                  35,
    "Copa América qualification":           35,

    # --- Tier D: Diğer kıtasal elemeler / Nations League'ler ----------
    "African Cup of Nations qualification": 28,
    "AFC Asian Cup qualification":          28,
    "CONCACAF Nations League":              28,
    "CONCACAF Nations League qualification": 28,
    "CONCACAF Championship":                28,   # Gold Cup öncesi adı
    "CONCACAF Championship qualification":  28,
    "Gold Cup":                             28,
    "Gold Cup qualification":               28,

    # --- Tier E: Bölgesel kupalar (düşük rekabet yoğunluğu) -----------
    "CECAFA Cup":                           22,
    "CFU Caribbean Cup":                    22,
    "CFU Caribbean Cup qualification":      22,
    "COSAFA Cup":                           22,
    "COSAFA Cup qualification":             22,
    "AFF Championship":                     22,
    "AFF Championship qualification":       22,
    "ASEAN Championship":                   22,
    "ASEAN Championship qualification":     22,
    "SAFF Cup":                             22,
    "Arab Cup":                             22,
    "Arab Cup qualification":               22,
    "EAFF Championship":                    22,
    "EAFF Championship qualification":      22,
    "WAFF Championship":                    22,
    "UNCAF Cup":                            22,
    "Baltic Cup":                           22,
    "Nordic Championship":                  22,
    "Gulf Cup":                             22,
    "AFC Challenge Cup":                    22,
    "AFC Challenge Cup qualification":      22,
    "Oceania Nations Cup":                  22,
    "Oceania Nations Cup qualification":    22,
    "Pan American Championship":            22,

    # --- Tier F: Hazırlık maçları / FIFA pencereleri ------------------
    "Friendly":                             K_FRIENDLY,
    "FIFA Series":                          K_FRIENDLY,

    # --- Tier G: Çok branşlı oyunlar / amatör / CONIFA ----------------
    "Olympic Games":                        14,
    "Island Games":                         14,
    "Asian Games":                          14,
    "Southeast Asian Games":                14,
    "Southeast Asian Peninsular Games":     14,
    "South Pacific Games":                  14,
    "South Pacific Mini Games":             14,
    "South Asian Games":                    14,
    "Indian Ocean Island Games":            14,
    "Pacific Games":                        14,
    "Pacific Mini Games":                   14,
    "Central American and Caribbean Games": 14,
    "All-African Games":                    14,
    "Bolivarian Games":                     14,
    "Far Eastern Championship Games":       14,
    "East Asian Games":                     14,
    "Inter-Allied Games":                   14,
    "Afro-Asian Games":                     14,
    "GaNEFo":                               14,
    "CONIFA World Football Cup":            14,
    "CONIFA World Football Cup qualification": 14,
    "CONIFA World Cup qualification":       14,
    "CONIFA European Football Cup":         14,
    "CONIFA Africa Football Cup":           14,
    "CONIFA South America Football Cup":    14,
    "CONIFA Asia Cup":                      14,
    "ConIFA Challenger Cup":                14,
    "Viva World Cup":                       14,
}

FORM_WINDOW = 10           # son kaç maç forma hesabında kullanılır


# --- Konfederasyon haritası ----------------------------------------------
# FIFA'nın 6 konfederasyonuna göre üye ülkeler. Eşleşmeyen (CONIFA üyeleri,
# tarihsel/bölgesel takımlar vb.) "OTHER" konfederasyonuna düşer ve kendi
# ortalamasını oluşturur. "conf_strength" feature'ı, bir takımın bağlı
# olduğu konfederasyondaki takımların o ana kadarki ortalama ELO'sudur —
# örn. CONMEBOL'un ortalama gücü ile UEFA'nın ortalama gücü arasındaki
# farkı modele açıkça gösterir.
CONFEDERATION = {
    # --- UEFA ---
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA", "Croatia": "UEFA",
    "Cyprus": "UEFA", "Czech Republic": "UEFA", "Denmark": "UEFA",
    "England": "UEFA", "Estonia": "UEFA", "Faroe Islands": "UEFA",
    "Finland": "UEFA", "France": "UEFA", "Georgia": "UEFA", "Germany": "UEFA",
    "Gibraltar": "UEFA", "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Israel": "UEFA", "Italy": "UEFA", "Kazakhstan": "UEFA", "Kosovo": "UEFA",
    "Latvia": "UEFA", "Liechtenstein": "UEFA", "Lithuania": "UEFA",
    "Luxembourg": "UEFA", "Malta": "UEFA", "Moldova": "UEFA", "Monaco": "UEFA",
    "Montenegro": "UEFA", "Netherlands": "UEFA", "North Macedonia": "UEFA",
    "Northern Ireland": "UEFA", "Norway": "UEFA", "Poland": "UEFA",
    "Portugal": "UEFA", "Republic of Ireland": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "San Marino": "UEFA", "Scotland": "UEFA",
    "Serbia": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA", "Spain": "UEFA",
    "Sweden": "UEFA", "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA",
    "Wales": "UEFA",
    # tarihsel UEFA üyeleri
    "Czechoslovakia": "UEFA", "Yugoslavia": "UEFA", "German DR": "UEFA",

    # --- CONMEBOL ---
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",

    # --- CONCACAF ---
    "Anguilla": "CONCACAF", "Antigua and Barbuda": "CONCACAF", "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF", "Barbados": "CONCACAF", "Belize": "CONCACAF",
    "Bermuda": "CONCACAF", "Bonaire": "CONCACAF", "British Virgin Islands": "CONCACAF",
    "Canada": "CONCACAF", "Cayman Islands": "CONCACAF", "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF", "Curaçao": "CONCACAF", "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF", "El Salvador": "CONCACAF",
    "French Guiana": "CONCACAF", "Grenada": "CONCACAF", "Guadeloupe": "CONCACAF",
    "Guatemala": "CONCACAF", "Guyana": "CONCACAF", "Haiti": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF", "Martinique": "CONCACAF",
    "Mexico": "CONCACAF", "Montserrat": "CONCACAF", "Nicaragua": "CONCACAF",
    "Panama": "CONCACAF", "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Martin": "CONCACAF", "Saint Vincent and the Grenadines": "CONCACAF",
    "Sint Maarten": "CONCACAF", "Suriname": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "Turks and Caicos Islands": "CONCACAF",
    "United States": "CONCACAF", "United States Virgin Islands": "CONCACAF",

    # --- CAF ---
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Botswana": "CAF",
    "Burkina Faso": "CAF", "Burundi": "CAF", "Cameroon": "CAF", "Cape Verde": "CAF",
    "Central African Republic": "CAF", "Chad": "CAF", "Comoros": "CAF",
    "Congo": "CAF", "DR Congo": "CAF", "Djibouti": "CAF", "Egypt": "CAF",
    "Equatorial Guinea": "CAF", "Eritrea": "CAF", "Eswatini": "CAF",
    "Ethiopia": "CAF", "Gabon": "CAF", "Gambia": "CAF", "Ghana": "CAF",
    "Guinea": "CAF", "Guinea-Bissau": "CAF", "Ivory Coast": "CAF", "Kenya": "CAF",
    "Lesotho": "CAF", "Liberia": "CAF", "Libya": "CAF", "Madagascar": "CAF",
    "Malawi": "CAF", "Mali": "CAF", "Mauritania": "CAF", "Mauritius": "CAF",
    "Mayotte": "CAF", "Morocco": "CAF", "Mozambique": "CAF", "Namibia": "CAF",
    "Niger": "CAF", "Nigeria": "CAF", "Rwanda": "CAF",
    "São Tomé and Príncipe": "CAF", "Senegal": "CAF", "Seychelles": "CAF",
    "Sierra Leone": "CAF", "Somalia": "CAF", "South Africa": "CAF",
    "South Sudan": "CAF", "Sudan": "CAF", "Tanzania": "CAF", "Togo": "CAF",
    "Tunisia": "CAF", "Uganda": "CAF", "Zambia": "CAF", "Zanzibar": "CAF",
    "Zimbabwe": "CAF",

    # --- AFC ---
    "Afghanistan": "AFC", "Australia": "AFC", "Bahrain": "AFC",
    "Bangladesh": "AFC", "Bhutan": "AFC", "Brunei": "AFC", "Cambodia": "AFC",
    "China": "AFC", "Taiwan": "AFC", "Guam": "AFC", "Hong Kong": "AFC",
    "India": "AFC", "Indonesia": "AFC", "Iran": "AFC", "Iraq": "AFC",
    "Japan": "AFC", "Jordan": "AFC", "Kuwait": "AFC", "Kyrgyzstan": "AFC",
    "Laos": "AFC", "Lebanon": "AFC", "Macau": "AFC", "Malaysia": "AFC",
    "Maldives": "AFC", "Mongolia": "AFC", "Myanmar": "AFC", "Nepal": "AFC",
    "North Korea": "AFC", "Oman": "AFC", "Pakistan": "AFC", "Palestine": "AFC",
    "Philippines": "AFC", "Qatar": "AFC", "Saudi Arabia": "AFC",
    "Singapore": "AFC", "South Korea": "AFC", "Sri Lanka": "AFC", "Syria": "AFC",
    "Tajikistan": "AFC", "Thailand": "AFC", "Timor-Leste": "AFC",
    "Turkmenistan": "AFC", "United Arab Emirates": "AFC", "Uzbekistan": "AFC",
    "Vietnam": "AFC", "Yemen": "AFC",
    # tarihsel AFC üyeleri
    "North Vietnam": "AFC", "Vietnam Republic": "AFC", "South Yemen": "AFC",
    "Yemen DPR": "AFC",

    # --- OFC ---
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "Kiribati": "OFC", "Marshall Islands": "OFC", "Micronesia": "OFC",
    "New Caledonia": "OFC", "New Zealand": "OFC", "Niue": "OFC",
    "Northern Mariana Islands": "OFC", "Palau": "OFC",
    "Papua New Guinea": "OFC", "Samoa": "OFC", "Solomon Islands": "OFC",
    "Tahiti": "OFC", "Tonga": "OFC", "Tuvalu": "OFC", "Vanuatu": "OFC",

    # --- matches_2026.csv'de results.csv'den farklı yazılan adlar ---
    "Bosnia-Herzegovina": "UEFA",
    "Cape Verde Islands":  "CAF",
    "Congo DR":            "CAF",
    "Czechia":             "UEFA",
}


# --- Yardımcı fonksiyonlar ----------------------------------------------

def get_k(tournament: str) -> float:
    return TOURNAMENT_K.get(tournament, K_BASE)


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
        self.history: dict[str, list] = {}        # takım → [sonuçlar]
        self.elo_history: dict[str, list] = {}    # takım → [elo değerleri]
        self.goals_scored: dict[str, list] = {}   # takım → [attığı goller]
        self.goals_conceded: dict[str, list] = {} # takım → [yediği goller]
        self.conf_sum: dict[str, float] = {}      # konfederasyon → ELO toplamı
        self.conf_count: dict[str, int] = {}      # konfederasyon → takım sayısı

    def _register(self, team: str):
        """Takımı ilk karşılaşmasında ELO=INITIAL_ELO ile konfederasyonuna kaydeder."""
        if team not in self.elos:
            self.elos[team] = INITIAL_ELO
            conf = CONFEDERATION.get(team, "OTHER")
            self.conf_sum[conf]   = self.conf_sum.get(conf, 0.0) + INITIAL_ELO
            self.conf_count[conf] = self.conf_count.get(conf, 0) + 1

    def get_elo(self, team: str) -> float:
        self._register(team)
        return self.elos[team]

    def get_conf_strength(self, team: str) -> float:
        """Takımın bağlı olduğu konfederasyondaki o ana kadarki ortalama ELO."""
        self._register(team)
        conf = CONFEDERATION.get(team, "OTHER")
        return self.conf_sum[conf] / self.conf_count[conf]

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

        # konfederasyon ortalamalarını güncelle
        conf_h = CONFEDERATION.get(home, "OTHER")
        conf_a = CONFEDERATION.get(away, "OTHER")
        self.conf_sum[conf_h] = self.conf_sum.get(conf_h, 0.0) + delta
        self.conf_sum[conf_a] = self.conf_sum.get(conf_a, 0.0) - delta

        # elo geçmişi güncelle
        self._update_elo_history(home, elo_h)
        self._update_elo_history(away, elo_a)

        # forma güncelle
        self._update_form(home, act_h)
        self._update_form(away, 1.0 - act_h)

        # gol geçmişi güncelle
        self._update_goals(home, home_goals, away_goals)
        self._update_goals(away, away_goals, home_goals)

        return elo_h, elo_a  # maç öncesi değerler

    def _update_form(self, team: str, result: float):
        if team not in self.history:
            self.history[team] = []
        self.history[team].append(result)

    def _update_goals(self, team: str, scored: int, conceded: int):
        if team not in self.goals_scored:
            self.goals_scored[team] = []
            self.goals_conceded[team] = []
        self.goals_scored[team].append(scored)
        self.goals_conceded[team].append(conceded)

    def get_avg_scored(self, team: str, window: int = FORM_WINDOW) -> float:
        """Son N maçtaki ortalama atılan gol."""
        hist = self.goals_scored.get(team, [])
        if not hist:
            return 1.2  # genel ortalama
        return float(np.mean(hist[-window:]))

    def get_avg_conceded(self, team: str, window: int = FORM_WINDOW) -> float:
        """Son N maçtaki ortalama yenilen gol."""
        hist = self.goals_conceded.get(team, [])
        if not hist:
            return 1.0  # genel ortalama
        return float(np.mean(hist[-window:]))

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
        atk_h = builder.get_avg_scored(home)    # ev hücum gücü
        def_h = builder.get_avg_conceded(home)  # ev defans zayıflığı
        atk_a = builder.get_avg_scored(away)    # dep hücum gücü
        def_a = builder.get_avg_conceded(away)  # dep defans zayıflığı
        h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records)
        conf_h = builder.get_conf_strength(home)
        conf_a = builder.get_conf_strength(away)

        # ELO tabanlı beklenen sonuç (ev sahibi avantajı dahil)
        elo_win_prob_home = expected_score(
            elo_h + (0 if neut else HOME_ADVANTAGE), elo_a
        )

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
            "atk_home":      atk_h,
            "def_home":      def_h,
            "atk_away":      atk_a,
            "def_away":      def_a,
            "atk_vs_def_home": atk_h - def_a,  # ev hücum - dep defans
            "atk_vs_def_away": atk_a - def_h,  # dep hücum - ev defans
            "elo_win_prob_home": elo_win_prob_home,
            "conf_strength_home": conf_h,
            "conf_strength_away": conf_a,
            "conf_strength_diff": conf_h - conf_a,
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
            atk_h  = builder.get_avg_scored(home)
            def_h  = builder.get_avg_conceded(home)
            atk_a  = builder.get_avg_scored(away)
            def_a  = builder.get_avg_conceded(away)
            h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records)
            conf_h = builder.get_conf_strength(home)
            conf_a = builder.get_conf_strength(away)
            elo_win_prob_home = expected_score(elo_h, elo_a)  # nötr saha

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
                "atk_home":      atk_h,
                "def_home":      def_h,
                "atk_away":      atk_a,
                "def_away":      def_a,
                "atk_vs_def_home": atk_h - def_a,
                "atk_vs_def_away": atk_a - def_h,
                "elo_win_prob_home": elo_win_prob_home,
                "conf_strength_home": conf_h,
                "conf_strength_away": conf_a,
                "conf_strength_diff": conf_h - conf_a,
            })

    pred_df = pd.DataFrame(pred_rows)
    return train_df, pred_df


FEATURE_COLS = [
    "elo_diff", "elo_home_adj", "elo_away",
    "form_home", "form_away", "form_diff",
    "h2h_home_win", "h2h_draw", "h2h_away_win",
    "is_wc", "is_neutral",
    "elo_momentum_home", "elo_momentum_away", "elo_momentum_diff",
    "atk_home", "def_home", "atk_away", "def_away",
    "atk_vs_def_home", "atk_vs_def_away",
    "elo_win_prob_home",
    "conf_strength_home", "conf_strength_away", "conf_strength_diff",
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
