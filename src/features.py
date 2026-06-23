"""
features.py
-----------
1. Kaggle results.csv + data/matches_2026.csv birlestirir
2. former_names.csv ile takim adi normalizasyonu yapar
3. data/fifa_rankings.json ile FIFA siralamasi ekler (varsa)
4. Her mac icin ELO tabanli + FIFA + form feature'lar ueretir
5. Egitim dataseti ve 2026 tahmin dataseti doendueruer
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# --- Sabitler -----------------------------------------------------------

INITIAL_ELO    = 1500.0
K_BASE         = 32
K_WC           = 60
K_FRIENDLY     = 20
HOME_ADVANTAGE = 100

TOURNAMENT_K = {
    # --- Tier S ---
    "FIFA World Cup":                       K_WC,

    # --- Tier A ---
    "UEFA Euro":                            50,
    "Copa America":                         50,
    "Copa América":                         50,
    "African Cup of Nations":               50,
    "AFC Asian Cup":                        50,
    "Confederations Cup":                   50,

    # --- Tier B ---
    "FIFA World Cup qualification":         40,

    # --- Tier C ---
    "UEFA Euro qualification":              35,
    "UEFA Nations League":                  35,
    "Copa America qualification":           35,
    "Copa América qualification":           35,

    # --- Tier D ---
    "African Cup of Nations qualification": 28,
    "AFC Asian Cup qualification":          28,
    "CONCACAF Nations League":              28,
    "CONCACAF Nations League qualification": 28,
    "CONCACAF Championship":                28,
    "CONCACAF Championship qualification":  28,
    "Gold Cup":                             28,
    "Gold Cup qualification":               28,

    # --- Tier E ---
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

    # --- Tier F ---
    "Friendly":                             K_FRIENDLY,
    "FIFA Series":                          K_FRIENDLY,

    # --- Tier G ---
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

FORM_WINDOW       = 10   # uzun form penceresi
FORM_WINDOW_SHORT = 5    # kisa form penceresi

# FIFA siralamasi bilinmeyen takim icin varsayilan degerler
FIFA_DEFAULT_RANK     = 212    # son sirali (211) + 1
FIFA_DEFAULT_POINTS   = 700.0  # alt limitin altinda
FIFA_DEFAULT_MOVEMENT = 0


# --- Konfederasyon haritasi ----------------------------------------------
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
    "Cuba": "CONCACAF", "Curacao": "CONCACAF", "Curaçao": "CONCACAF",
    "Dominica": "CONCACAF", "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF", "French Guiana": "CONCACAF",
    "Grenada": "CONCACAF", "Guadeloupe": "CONCACAF", "Guatemala": "CONCACAF",
    "Guyana": "CONCACAF", "Haiti": "CONCACAF", "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF", "Martinique": "CONCACAF", "Mexico": "CONCACAF",
    "Montserrat": "CONCACAF", "Nicaragua": "CONCACAF", "Panama": "CONCACAF",
    "Puerto Rico": "CONCACAF", "Saint Kitts and Nevis": "CONCACAF",
    "Saint Lucia": "CONCACAF", "Saint Martin": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF", "Sint Maarten": "CONCACAF",
    "Suriname": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF", "United States": "CONCACAF",
    "United States Virgin Islands": "CONCACAF",

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
    "Sao Tome and Principe": "CAF", "São Tomé e Príncipe": "CAF",
    "Senegal": "CAF", "Seychelles": "CAF", "Sierra Leone": "CAF",
    "Somalia": "CAF", "South Africa": "CAF", "South Sudan": "CAF",
    "Sudan": "CAF", "Tanzania": "CAF", "Togo": "CAF", "Tunisia": "CAF",
    "Uganda": "CAF", "Zambia": "CAF", "Zanzibar": "CAF", "Zimbabwe": "CAF",

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
    "North Vietnam": "AFC", "Vietnam Republic": "AFC",
    "South Yemen": "AFC", "Yemen DPR": "AFC",

    # --- OFC ---
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "Kiribati": "OFC", "Marshall Islands": "OFC", "Micronesia": "OFC",
    "New Caledonia": "OFC", "New Zealand": "OFC", "Niue": "OFC",
    "Northern Mariana Islands": "OFC", "Palau": "OFC",
    "Papua New Guinea": "OFC", "Samoa": "OFC", "Solomon Islands": "OFC",
    "Tahiti": "OFC", "Tonga": "OFC", "Tuvalu": "OFC", "Vanuatu": "OFC",

    # --- matches_2026.csv ve diger alternatif adlar ---
    "Bosnia-Herzegovina": "UEFA",
    "Cape Verde Islands":  "CAF",
    "Congo DR":            "CAF",
    "Czechia":             "UEFA",
}


# --- Yardimci fonksiyonlar ----------------------------------------------

def get_k(tournament: str) -> float:
    return TOURNAMENT_K.get(tournament, K_BASE)


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def actual_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    elif home_goals == away_goals:
        return 0.5
    return 0.0


def goal_diff_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return 1.75 + (gd - 3) * 0.1


# --- Takim adi normalizasyonu -------------------------------------------

def build_name_map(former_names_path: str) -> dict:
    if not os.path.exists(former_names_path):
        return {}
    df = pd.read_csv(former_names_path)
    return dict(zip(df["former"], df["current"]))


def normalize_team(name: str, name_map: dict) -> str:
    return name_map.get(name, name)


# --- Ana sinif ----------------------------------------------------------

class EloFeatureBuilder:

    def __init__(self):
        self.elos: dict[str, float] = {}
        self.history: dict[str, list] = {}
        self.elo_history: dict[str, list] = {}
        self.goals_scored: dict[str, list] = {}
        self.goals_conceded: dict[str, list] = {}
        self.conf_sum: dict[str, float] = {}
        self.conf_count: dict[str, int] = {}

    def _register(self, team: str):
        if team not in self.elos:
            self.elos[team] = INITIAL_ELO
            conf = CONFEDERATION.get(team, "OTHER")
            self.conf_sum[conf]   = self.conf_sum.get(conf, 0.0) + INITIAL_ELO
            self.conf_count[conf] = self.conf_count.get(conf, 0) + 1

    def get_elo(self, team: str) -> float:
        self._register(team)
        return self.elos[team]

    def get_conf_strength(self, team: str) -> float:
        self._register(team)
        conf = CONFEDERATION.get(team, "OTHER")
        return self.conf_sum[conf] / self.conf_count[conf]

    def update_elo(self, home: str, away: str, home_goals: int,
                   away_goals: int, neutral: bool, tournament: str) -> Tuple[float, float]:
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

        conf_h = CONFEDERATION.get(home, "OTHER")
        conf_a = CONFEDERATION.get(away, "OTHER")
        self.conf_sum[conf_h] = self.conf_sum.get(conf_h, 0.0) + delta
        self.conf_sum[conf_a] = self.conf_sum.get(conf_a, 0.0) - delta

        self._update_elo_history(home, elo_h)
        self._update_elo_history(away, elo_a)
        self._update_form(home, act_h)
        self._update_form(away, 1.0 - act_h)
        self._update_goals(home, home_goals, away_goals)
        self._update_goals(away, away_goals, home_goals)

        return elo_h, elo_a

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
        hist = self.goals_scored.get(team, [])
        if not hist:
            return 1.2
        return float(np.mean(hist[-window:]))

    def get_avg_conceded(self, team: str, window: int = FORM_WINDOW) -> float:
        hist = self.goals_conceded.get(team, [])
        if not hist:
            return 1.0
        return float(np.mean(hist[-window:]))

    def _update_elo_history(self, team: str, elo: float):
        if team not in self.elo_history:
            self.elo_history[team] = []
        self.elo_history[team].append(elo)

    def get_elo_momentum(self, team: str, window: int = 5) -> float:
        hist = self.elo_history.get(team, [])
        if len(hist) < 2:
            return 0.0
        recent = hist[-window:]
        return float(recent[-1] - recent[0])

    def get_form(self, team: str, window: int = FORM_WINDOW) -> float:
        hist = self.history.get(team, [])
        if not hist:
            return 0.5
        return float(np.mean(hist[-window:]))

    def get_h2h(self, home: str, away: str,
                h2h_records: dict,
                current_date=None) -> Tuple[float, float, float]:
        """
        Kafa kafaya istatistikler. Yakin tarihli maclara daha yuksek agirlik
        verilir (ustel azalim, yari omur ~7 yil). h2h_records yapisi:
        {(home, away): [(result, date)]}
        """
        if current_date is None:
            current_date = pd.Timestamp.now()

        key     = (home, away)
        key_rev = (away, home)

        all_entries = []
        for r, date in h2h_records.get(key, []):
            years_ago = max(0.0, (current_date - date).days / 365.25)
            w = np.exp(-0.1 * years_ago)
            all_entries.append((r, w))
        for r, date in h2h_records.get(key_rev, []):
            years_ago = max(0.0, (current_date - date).days / 365.25)
            w = np.exp(-0.1 * years_ago)
            all_entries.append((1.0 - r, w))

        if not all_entries:
            return 0.33, 0.33, 0.34

        total_w = sum(w for _, w in all_entries)
        wins    = sum(w for r, w in all_entries if r == 1.0) / total_w
        draws   = sum(w for r, w in all_entries if r == 0.5) / total_w
        losses  = 1.0 - wins - draws
        return wins, draws, losses


# --- ELO durum önbelleği ------------------------------------------------

def _csv_fingerprint(path: str) -> str:
    """Dosya boyutu + değiştirilme zamanı parmak izi (tek syscall)."""
    s = os.stat(path)
    return f"{s.st_size}:{int(s.st_mtime)}"


def _save_elo_cache(builder: EloFeatureBuilder, h2h_records: dict,
                    fingerprint: str) -> None:
    path = os.path.join(DATA_DIR, "elo_cache.json")
    state = {
        "fingerprint":    fingerprint,
        "elos":           builder.elos,
        "conf_sum":       builder.conf_sum,
        "conf_count":     builder.conf_count,
        # Form/momentum pencereleri için son 20 kayıt yeterli
        "history":        {k: v[-20:] for k, v in builder.history.items()},
        "elo_history":    {k: v[-20:] for k, v in builder.elo_history.items()},
        "goals_scored":   {k: v[-20:] for k, v in builder.goals_scored.items()},
        "goals_conceded": {k: v[-20:] for k, v in builder.goals_conceded.items()},
        "h2h":            {
            f"{h}|||{a}": [[r, d.strftime('%Y-%m-%d')] for r, d in v]
            for (h, a), v in h2h_records.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    print(f"[features] ELO cache kaydedildi ({len(builder.elos)} takım, {len(h2h_records)} h2h çifti)")


def _load_elo_cache(fingerprint: str) -> "tuple | None":
    path = os.path.join(DATA_DIR, "elo_cache.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("fingerprint") != fingerprint:
        return None
    b = EloFeatureBuilder()
    b.elos          = {k: float(v)               for k, v in s["elos"].items()}
    b.conf_sum      = {k: float(v)               for k, v in s["conf_sum"].items()}
    b.conf_count    = {k: int(v)                 for k, v in s["conf_count"].items()}
    b.history       = {k: [float(x) for x in v] for k, v in s["history"].items()}
    b.elo_history   = {k: [float(x) for x in v] for k, v in s["elo_history"].items()}
    b.goals_scored  = {k: [int(x)   for x in v] for k, v in s["goals_scored"].items()}
    b.goals_conceded= {k: [int(x)   for x in v] for k, v in s["goals_conceded"].items()}
    h2h: dict = {}
    for ks, entries in s["h2h"].items():
        parts = ks.split("|||", 1)
        if len(parts) == 2:
            h2h[(parts[0], parts[1])] = [(float(r), pd.Timestamp(d)) for r, d in entries]
    print(f"[features] ELO cache yüklendi ({len(b.elos)} takım, {len(h2h)} h2h çifti)")
    return b, h2h


def _make_pred_df(builder: EloFeatureBuilder, h2h_records: dict,
                  unplayed: pd.DataFrame, rankings: dict) -> pd.DataFrame:
    """Oynanmamış maçlar için tahmin feature'larını oluşturur."""
    def _fifa(team: str) -> dict:
        d = rankings.get(team, {})
        return {
            "rank":     d.get("rank",     FIFA_DEFAULT_RANK),
            "points":   d.get("points",   FIFA_DEFAULT_POINTS),
            "movement": d.get("movement", FIFA_DEFAULT_MOVEMENT),
        }

    pred_rows = []
    for _, row in unplayed.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        date  = row["date"]
        elo_h = builder.get_elo(home);  elo_a = builder.get_elo(away)
        form_h = builder.get_form(home); form_a = builder.get_form(away)
        form5_h = builder.get_form(home, window=FORM_WINDOW_SHORT)
        form5_a = builder.get_form(away, window=FORM_WINDOW_SHORT)
        mom_h  = builder.get_elo_momentum(home)
        mom_a  = builder.get_elo_momentum(away)
        atk_h  = builder.get_avg_scored(home);  def_h = builder.get_avg_conceded(home)
        atk_a  = builder.get_avg_scored(away);  def_a = builder.get_avg_conceded(away)
        h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records, current_date=date)
        conf_h = builder.get_conf_strength(home)
        conf_a = builder.get_conf_strength(away)
        elo_win_prob_home = expected_score(elo_h, elo_a)
        fifa_h = _fifa(home); fifa_a = _fifa(away)

        pred_rows.append({
            "date":         date,
            "home_team":    home,
            "away_team":    away,
            "stage":        row.get("stage", ""),
            "match_id":     row.get("match_id", ""),
            "kickoff_utc":  row.get("kickoff_utc", ""),
            "elo_home":     elo_h,  "elo_away":     elo_a,
            "elo_diff":     elo_h - elo_a,  "elo_home_adj": elo_h,
            "form_home":    form_h, "form_away":    form_a,
            "form_diff":    form_h - form_a,
            "form5_home":   form5_h,"form5_away":   form5_a,
            "form5_diff":   form5_h - form5_a,
            "h2h_home_win": h2h_w,  "h2h_draw":     h2h_d,  "h2h_away_win": h2h_l,
            "is_wc": 1,  "is_neutral": 1,
            "elo_momentum_home": mom_h,  "elo_momentum_away": mom_a,
            "elo_momentum_diff": mom_h - mom_a,
            "atk_home": atk_h,  "def_home": def_h,
            "atk_away": atk_a,  "def_away": def_a,
            "atk_vs_def_home": atk_h - def_a,  "atk_vs_def_away": atk_a - def_h,
            "elo_win_prob_home": elo_win_prob_home,
            "conf_strength_home": conf_h,  "conf_strength_away": conf_a,
            "conf_strength_diff": conf_h - conf_a,
            "fifa_rank_home":     fifa_h["rank"],   "fifa_rank_away":     fifa_a["rank"],
            "fifa_points_home":   fifa_h["points"], "fifa_points_away":   fifa_a["points"],
            "fifa_points_diff":   fifa_h["points"] - fifa_a["points"],
            "fifa_movement_home": fifa_h["movement"],"fifa_movement_away": fifa_a["movement"],
        })
    return pd.DataFrame(pred_rows)


# --- Pipeline -----------------------------------------------------------

def build_features(kaggle_results_path: str,
                   wc2026_path: str,
                   former_names_path: str,
                   pred_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Doendueruer:
        train_df  -- home/away score bilinen tum tarihsel maclar + feature'lar
        pred_df   -- 2026 WC'de oynanmamis maclar + feature'lar

    pred_only=True: ELO cache varsa kaggle döngüsü atlanır (predict.py için hızlı yol).
    """
    # 0. FIFA siralamalarini yukle (varsa)
    ranking_path = os.path.join(DATA_DIR, "fifa_rankings.json")
    _rankings: dict = {}
    if os.path.exists(ranking_path):
        with open(ranking_path, "r", encoding="utf-8") as _f:
            _rankings = json.load(_f)
        print(f"[features] FIFA siralamasi yueklendi: {len(_rankings)} takim")

    def _fifa(team: str) -> dict:
        d = _rankings.get(team, {})
        return {
            "rank":     d.get("rank",     FIFA_DEFAULT_RANK),
            "points":   d.get("points",   FIFA_DEFAULT_POINTS),
            "movement": d.get("movement", FIFA_DEFAULT_MOVEMENT),
        }

    # ── pred_only hızlı yolu: ELO cache varsa kaggle döngüsünü atla ──────
    if pred_only and os.path.exists(wc2026_path):
        wc2026_q = pd.read_csv(wc2026_path, parse_dates=["date"])
        played_cnt = int((wc2026_q["status"] == "FINISHED").sum())
        nm_q = build_name_map(former_names_path)
        for col in ["home_team", "away_team"]:
            wc2026_q[col] = wc2026_q[col].map(lambda x: normalize_team(x, nm_q))
        combined_fp = _csv_fingerprint(kaggle_results_path) + f"|{played_cnt}"
        cached = _load_elo_cache(combined_fp)
        if cached is not None:
            builder_c, h2h_c = cached
            unplayed_c = wc2026_q[wc2026_q["status"] != "FINISHED"].copy()
            return pd.DataFrame(), _make_pred_df(builder_c, h2h_c, unplayed_c, _rankings)
        print("[features] ELO cache miss — tam hesaplama yapılıyor")

    # 1. Veri yukle
    hist = pd.read_csv(kaggle_results_path, parse_dates=["date"])

    wc2026 = pd.DataFrame()
    if os.path.exists(wc2026_path):
        wc2026 = pd.read_csv(wc2026_path, parse_dates=["date"])

    name_map = build_name_map(former_names_path)

    # 2. Birlestir
    if not wc2026.empty:
        played = wc2026[wc2026["status"] == "FINISHED"][
            ["date", "home_team", "away_team", "home_score",
             "away_score", "tournament", "neutral"]
        ].copy()
        hist = pd.concat([hist, played], ignore_index=True)

    hist.sort_values("date", inplace=True)
    hist.reset_index(drop=True, inplace=True)

    # 3. Takim adlarini normalize et
    for col in ["home_team", "away_team"]:
        hist[col] = hist[col].map(lambda x: normalize_team(x, name_map))
        if not wc2026.empty:
            wc2026[col] = wc2026[col].map(lambda x: normalize_team(x, name_map))

    # 4. neutral bool'a cevir
    hist["neutral"] = hist["neutral"].map(
        lambda x: True if str(x).upper() == "TRUE" else False
    )

    # 5. ELO + feature hesapla
    builder = EloFeatureBuilder()
    h2h_records: dict = {}  # {(home, away): [(result, date)]}
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
        date  = row["date"]

        # Mac oncesi feature'lar
        elo_h  = builder.get_elo(home)
        elo_a  = builder.get_elo(away)
        form_h = builder.get_form(home)
        form_a = builder.get_form(away)
        form5_h = builder.get_form(home, window=FORM_WINDOW_SHORT)
        form5_a = builder.get_form(away, window=FORM_WINDOW_SHORT)
        mom_h  = builder.get_elo_momentum(home)
        mom_a  = builder.get_elo_momentum(away)
        atk_h  = builder.get_avg_scored(home)
        def_h  = builder.get_avg_conceded(home)
        atk_a  = builder.get_avg_scored(away)
        def_a  = builder.get_avg_conceded(away)
        h2h_w, h2h_d, h2h_l = builder.get_h2h(home, away, h2h_records,
                                                current_date=date)
        conf_h = builder.get_conf_strength(home)
        conf_a = builder.get_conf_strength(away)
        elo_win_prob_home = expected_score(
            elo_h + (0 if neut else HOME_ADVANTAGE), elo_a
        )

        fifa_h = _fifa(home)
        fifa_a = _fifa(away)

        if hg > ag:
            outcome = 2
        elif hg == ag:
            outcome = 1
        else:
            outcome = 0

        feature_rows.append({
            "date":          date,
            "home_team":     home,
            "away_team":     away,
            "home_score":    hg,
            "away_score":    ag,
            "tournament":    tourn,
            "neutral":       neut,
            # ELO
            "elo_home":      elo_h,
            "elo_away":      elo_a,
            "elo_diff":      elo_h - elo_a,
            "elo_home_adj":  elo_h + (0 if neut else HOME_ADVANTAGE),
            # Form (uzun + kisa pencere)
            "form_home":     form_h,
            "form_away":     form_a,
            "form_diff":     form_h - form_a,
            "form5_home":    form5_h,
            "form5_away":    form5_a,
            "form5_diff":    form5_h - form5_a,
            # H2H
            "h2h_home_win":  h2h_w,
            "h2h_draw":      h2h_d,
            "h2h_away_win":  h2h_l,
            # Baglam
            "is_wc":         1 if "world cup" in tourn.lower() else 0,
            "is_neutral":    1 if neut else 0,
            # ELO momentum
            "elo_momentum_home": mom_h,
            "elo_momentum_away": mom_a,
            "elo_momentum_diff": mom_h - mom_a,
            # Hucum / Defans
            "atk_home":      atk_h,
            "def_home":      def_h,
            "atk_away":      atk_a,
            "def_away":      def_a,
            "atk_vs_def_home": atk_h - def_a,
            "atk_vs_def_away": atk_a - def_h,
            "elo_win_prob_home": elo_win_prob_home,
            # Konfederasyon
            "conf_strength_home": conf_h,
            "conf_strength_away": conf_a,
            "conf_strength_diff": conf_h - conf_a,
            # FIFA siralama
            "fifa_rank_home":     fifa_h["rank"],
            "fifa_rank_away":     fifa_a["rank"],
            "fifa_points_home":   fifa_h["points"],
            "fifa_points_away":   fifa_a["points"],
            "fifa_points_diff":   fifa_h["points"] - fifa_a["points"],
            "fifa_movement_home": fifa_h["movement"],
            "fifa_movement_away": fifa_a["movement"],
            # Hedef
            "outcome":       outcome,
        })

        # ELO guncelle
        builder.update_elo(home, away, hg, ag, neut, tourn)

        # H2H guncelle — (sonuc, tarih) olarak sakla
        key = (home, away)
        if key not in h2h_records:
            h2h_records[key] = []
        h2h_records[key].append((actual_score(hg, ag), date))

    train_df = pd.DataFrame(feature_rows)

    # ELO cache kaydet (kaggle + oynanmış 2026 dahil) — predict.py hızlı yolu için
    played_cnt = int((wc2026["status"] == "FINISHED").sum()) if not wc2026.empty else 0
    combined_fp = _csv_fingerprint(kaggle_results_path) + f"|{played_cnt}"
    _save_elo_cache(builder, h2h_records, combined_fp)

    # 6. 2026 tahmin feature'lari (oynanmamis maclar)
    if not wc2026.empty:
        unplayed = wc2026[wc2026["status"] != "FINISHED"].copy()
        pred_df = _make_pred_df(builder, h2h_records, unplayed, _rankings)
    else:
        pred_df = pd.DataFrame()
    return train_df, pred_df


FEATURE_COLS = [
    # ELO
    "elo_diff", "elo_home_adj", "elo_away",
    # Form (uzun pencere)
    "form_home", "form_away", "form_diff",
    # Form (kisa pencere — guncel form)
    "form5_home", "form5_away", "form5_diff",
    # H2H
    "h2h_home_win", "h2h_draw", "h2h_away_win",
    # Baglam
    "is_wc", "is_neutral",
    # ELO momentum
    "elo_momentum_home", "elo_momentum_away", "elo_momentum_diff",
    # Hucum / Defans
    "atk_home", "def_home", "atk_away", "def_away",
    "atk_vs_def_home", "atk_vs_def_away",
    "elo_win_prob_home",
    # Konfederasyon
    "conf_strength_home", "conf_strength_away", "conf_strength_diff",
    # FIFA siralama
    "fifa_rank_home", "fifa_rank_away",
    "fifa_points_home", "fifa_points_away", "fifa_points_diff",
    "fifa_movement_home", "fifa_movement_away",
]


if __name__ == "__main__":
    train, pred = build_features(
        kaggle_results_path=os.path.join(DATA_DIR, "results.csv"),
        wc2026_path=os.path.join(DATA_DIR, "matches_2026.csv"),
        former_names_path=os.path.join(DATA_DIR, "former_names.csv"),
    )
    print(f"Egitim seti: {len(train)} mac, {len(FEATURE_COLS)} feature")
    print(f"Tahmin seti: {len(pred)} mac")
    print(train[["date", "home_team", "away_team", "elo_diff", "outcome"]].tail())
