"""
poisson_model.py
-----------------
Paylaşılan Poisson skor matrisi ve Dixon-Coles düşük skor düzeltmesi.
ρ (rho), 0-0/1-0/0-1/1-1 hücrelerini düzeltir; negatif ρ beraberlik
olasılığını (özellikle düşük skorlu maçlarda) artırır.
Referans: Dixon, M.J. & Coles, S.G. (1997), "Modelling Association
Football Scores and Inefficiencies in the Betting Market".
"""
import os
import json
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

MAX_GOALS  = 9  # Olasılık matrisinde 0-8 gol arası hesaplanır
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RHO_PATH   = os.path.join(MODELS_DIR, "rho.json")
RHO_BOUNDS = (-0.2, 0.2)

# Zaman ağırlıklı rho tahmini için varsayılan azalma katsayısı (xi).
# Maçlar arası gün farkına göre üstel ağırlıklandırma: w = exp(-xi * gün).
# xi = ln(2) / 1095 → ~3 yıl yarı ömür (milli takım maçları seyrek
# oynandığı için kulüp futbolundaki xi ~ 0.0065/gün çok agresif kalır).
DEFAULT_XI = 0.00065


def dixon_coles_tau(x, y, lambda_x, lambda_y, rho):
    """0-0, 1-0, 0-1, 1-1 hücreleri için düzeltme çarpanı."""
    if x == 0 and y == 0:
        return 1 - lambda_x * lambda_y * rho
    if x == 0 and y == 1:
        return 1 + lambda_x * rho
    if x == 1 and y == 0:
        return 1 + lambda_y * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lambda_home, lambda_away, rho=0.0):
    """
    MAX_GOALS x MAX_GOALS olasılık matrisi.
    matrix[i][j] = ev sahibi i gol, deplasman j gol atar olasılığı
    """
    lambda_home = max(0.01, lambda_home)
    lambda_away = max(0.01, lambda_away)
    home_probs = poisson.pmf(range(MAX_GOALS), lambda_home)
    away_probs = poisson.pmf(range(MAX_GOALS), lambda_away)
    matrix = np.outer(home_probs, away_probs)

    if rho != 0.0:
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= dixon_coles_tau(x, y, lambda_home, lambda_away, rho)
        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()

    return matrix


def top_n_scores(matrix, n=3):
    """Olasılığa göre en yüksek n skoru (i, j) çiftleri olarak döndürür."""
    flat = [(i, j, matrix[i, j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    flat.sort(key=lambda t: -t[2])
    return [(i, j) for i, j, _ in flat[:n]]


def match_probabilities(lambda_home, lambda_away, rho=0.0):
    """
    İki Poisson parametresinden maç sonucu olasılıklarını hesaplar.
    Döndürür: (prob_home_win, prob_draw, prob_away_win, most_likely_score)
    """
    matrix = score_matrix(lambda_home, lambda_away, rho)

    prob_home_win = float(np.tril(matrix, -1).sum())   # i > j
    prob_draw     = float(np.trace(matrix))             # i == j
    prob_away_win = float(np.triu(matrix, 1).sum())    # i < j

    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely_score = f"{idx[0]}-{idx[1]}"

    return prob_home_win, prob_draw, prob_away_win, most_likely_score


def _tau_vec(hg, ag, lh, la, rho):
    tau = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    tau[m01] = 1 + lh[m01] * rho
    tau[m10] = 1 + la[m10] * rho
    tau[m11] = 1 - rho
    return tau


def time_decay_weights(days_ago, xi=DEFAULT_XI):
    """
    Maçın "bugün"e göre kaç gün önce oynandığına bakarak üstel ağırlık
    üretir (Dixon-Coles, 1997). xi=0 → tüm maçlar eşit ağırlıklı.
    """
    days_ago = np.maximum(0.0, np.asarray(days_ago, dtype=float))
    return np.exp(-xi * days_ago)


def estimate_rho(lambda_home, lambda_away, home_goals, away_goals,
                 weights=None, bounds=RHO_BOUNDS):
    """
    Gerçekleşen skorların (ağırlıklı) log-olabilirliğini maksimize eden
    rho'yu bulur. `weights` verilirse (örn. time_decay_weights ile)
    yakın tarihli maçlar rho tahminine daha fazla katkı yapar.

    Tahmin edilen rho, sınırlardan birine çok yaklaşırsa (sınırın
    %5'inden az kala) sınırlar genişletilip yeniden tahmin edilir —
    böylece gerçek optimum yapay olarak kırpılmamış olur.
    """
    lh = np.maximum(0.01, np.asarray(lambda_home, dtype=float))
    la = np.maximum(0.01, np.asarray(lambda_away, dtype=float))
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)

    if weights is None:
        weights = np.ones(len(hg))
    else:
        weights = np.asarray(weights, dtype=float)

    base_p = poisson.pmf(hg, lh) * poisson.pmf(ag, la)

    def neg_log_likelihood(rho):
        p = np.clip(base_p * _tau_vec(hg, ag, lh, la, rho), 1e-10, None)
        return -(weights * np.log(p)).sum()

    result = minimize_scalar(neg_log_likelihood, bounds=bounds, method="bounded")
    rho = float(result.x)

    span = bounds[1] - bounds[0]
    near_lower = rho - bounds[0] < 0.05 * span
    near_upper = bounds[1] - rho < 0.05 * span
    if near_lower or near_upper:
        wider = (bounds[0] * 2, bounds[1] * 2)
        result = minimize_scalar(neg_log_likelihood, bounds=wider, method="bounded")
        rho = float(result.x)

    return rho


def load_rho(default=0.0):
    if os.path.exists(RHO_PATH):
        try:
            with open(RHO_PATH) as f:
                return float(json.load(f).get("rho", default))
        except Exception:
            return default
    return default


def save_rho(rho):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(RHO_PATH, "w") as f:
        json.dump({"rho": round(float(rho), 4)}, f, indent=2)
