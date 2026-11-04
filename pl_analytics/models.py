"""Models and optimisers: scikit-learn pipelines, XI and FPL-squad selection, captain planning and calibration.

All randomness uses RANDOM_STATE, so every run gives identical results.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import model_feature_cols, points_feature_cols

RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["prev_position"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]


# ── Player models (Experiments 1 and 2) ──────────────────────────────────────
def make_player_pipeline(feature_cols, model):
    """Median-impute + scale numeric features, one-hot encode position, then fit `model`."""
    numeric_features = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numeric_features),
                                      ("cat", categorical_pipe, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def make_points_pipeline(quantile=None):
    """Experiment 1: gradient boosting on player history.

    quantile=None → absolute-error loss (predicts the median outcome, robust to outliers).
    quantile=0.1 / 0.9 → quantile loss, giving the low / high end of an 80% range.
    """
    loss = dict(loss="absolute_error") if quantile is None else dict(loss="quantile", quantile=quantile)
    return make_player_pipeline(points_feature_cols, HistGradientBoostingRegressor(
        **loss, learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=30, random_state=RANDOM_STATE))


def make_regression_pipeline():
    """Experiment 2: random forest on last season's stats (prices move smoothly, so a forest suits them)."""
    return make_player_pipeline(model_feature_cols, RandomForestRegressor(
        n_estimators=350, min_samples_leaf=4, max_features=0.9, random_state=RANDOM_STATE, n_jobs=-1))


def regression_metrics(name, y_true, y_pred):
    """MAE, RMSE and R² in one dictionary row."""
    return {"Model": name, "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5, "R²": r2_score(y_true, y_pred)}


def interval_coverage(y_true, low, high):
    """Share of actual values inside [low, high], and the average width of the range."""
    y_true, low, high = map(np.asarray, (y_true, low, high))
    return {"Coverage": float(np.mean((y_true >= low) & (y_true <= high))), "Average width": float(np.mean(high - low))}


# ── Match models (Experiments 3 and 4) ───────────────────────────────────────
def candidate_outcome_models():
    """Candidate result classifiers (imputation first: a club's first matches have no form history yet)."""
    return {
        "Logistic regression": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                                                 ("model", LogisticRegression(C=0.5, max_iter=2000))]),
        "Random forest": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")),
                                           ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                                                            random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient boosting": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")),
                                               ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200,
                                                                                        max_depth=3, random_state=RANDOM_STATE))]),
    }


def make_goal_model():
    """Experiment 4: one Poisson regression per target (home goals, away goals)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(PoissonRegressor(alpha=0.1, max_iter=1000))),
    ])


def brier_score(y_true, proba, classes):
    """Multi-class Brier score: mean squared gap between predicted probabilities and what happened (lower is better)."""
    onehot = np.column_stack([(np.asarray(y_true) == c).astype(float) for c in classes])
    return float(np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1)))


def reliability_table(y_true, proba, classes, n_bins=8):
    """Pool every (outcome, probability) pair and bin them: mean predicted chance vs how often it happened."""
    outcome = np.concatenate([(np.asarray(y_true) == c).astype(int) for c in classes])
    predicted = np.concatenate([np.asarray(proba)[:, i] for i in range(len(classes))])
    observed_rate, mean_predicted = calibration_curve(outcome, predicted, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"Predicted chance": mean_predicted, "Observed frequency": observed_rate})


# ── Team selection ───────────────────────────────────────────────────────────
def select_best_xi(players, score_col="predicted_next_points"):
    """Try every legal formation (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD) and return the highest-scoring XI.

    Within a formation the best choice is simply the top-n players per position, so checking
    all 3 × 4 formation counts finds the true optimum.
    """
    by_pos = {pos: players[players["position"] == pos].sort_values(score_col, ascending=False) for pos in POSITIONS}
    best_score, best_xi = -np.inf, pd.DataFrame()
    for n_def in range(3, 6):
        for n_mid in range(2, 6):
            n_fwd = 10 - n_def - n_mid                                # 10 outfield places after the goalkeeper
            if not 1 <= n_fwd <= 3:
                continue
            counts = {"GK": 1, "DEF": n_def, "MID": n_mid, "FWD": n_fwd}
            xi = pd.concat([by_pos[pos].head(n) for pos, n in counts.items()])
            if len(xi) == 11 and xi[score_col].sum() > best_score:
                best_score, best_xi = xi[score_col].sum(), xi
    return best_xi


FPL_SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}          # 15 players
FPL_XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def pick_fpl_squad(players, score_col, price_col="start_price", budget=100.0, max_per_club=3, bench_weight=0.1):
    """Pick a legal Fantasy Premier League squad with an integer program (scipy `milp`).

    Rules: 15 players (2 GK, 5 DEF, 5 MID, 3 FWD), total price ≤ budget, at most `max_per_club` per club,
    a starting XI in a legal formation, and one captain (whose points count twice).
    Objective: maximise predicted points of the XI, plus the captain's points again, plus a small weight on
    the bench (substitutes only score when a starter misses out).

    Returns the 15 players with `starter` and `captain` flags.
    """
    pool = players.dropna(subset=[score_col, price_col]).reset_index(drop=True)
    n = len(pool)
    points = pool[score_col].to_numpy(float)
    # Variable blocks: x = in squad, s = starts, c = captain (each n binary variables)
    objective = -np.concatenate([bench_weight * points, (1 - bench_weight) * points, points])   # milp minimises
    rows, lower, upper = [], [], []

    def add(coeffs_x=None, coeffs_s=None, coeffs_c=None, lo=-np.inf, hi=np.inf):
        row = np.zeros(3 * n)
        for block, coeffs in enumerate([coeffs_x, coeffs_s, coeffs_c]):
            if coeffs is not None:
                row[block * n:(block + 1) * n] = coeffs
        rows.append(row), lower.append(lo), upper.append(hi)

    add(coeffs_x=pool[price_col].to_numpy(float), hi=budget)                       # budget
    for pos in POSITIONS:
        mask = (pool["position"] == pos).to_numpy(float)
        add(coeffs_x=mask, lo=FPL_SQUAD_SIZE[pos], hi=FPL_SQUAD_SIZE[pos])         # squad shape
        add(coeffs_s=mask, lo=FPL_XI_LIMITS[pos][0], hi=FPL_XI_LIMITS[pos][1])      # legal formation
    for club in pool["team"].dropna().unique():
        add(coeffs_x=(pool["team"] == club).to_numpy(float), hi=max_per_club)      # ≤ 3 per club
    add(coeffs_s=np.ones(n), lo=11, hi=11)                                          # 11 starters
    add(coeffs_c=np.ones(n), lo=1, hi=1)                                            # 1 captain
    eye = np.eye(n)
    for i in range(n):                                                              # start ⇒ in squad; captain ⇒ starts
        add(coeffs_x=-eye[i], coeffs_s=eye[i], hi=0)
        add(coeffs_s=-eye[i], coeffs_c=eye[i], hi=0)

    result = milp(objective, constraints=LinearConstraint(np.array(rows), lower, upper),
                  integrality=np.ones(3 * n), bounds=Bounds(0, 1))
    if not result.success:
        raise RuntimeError(f"No legal FPL squad found: {result.message}")
    x, s, c = (np.round(result.x[k * n:(k + 1) * n]).astype(bool) for k in range(3))
    squad = pool[x].copy()
    squad["starter"], squad["captain"] = s[x], c[x]
    return squad.sort_values(["starter", "position", score_col], ascending=[False, True, False],
                             key=lambda col: col.map({p: i for i, p in enumerate(POSITIONS)}) if col.name == "position" else col)


def captain_plan(starters, fixtures, club_strength, score_col, n_gameweeks=38, home_boost=0.1):
    """Suggest a captain for every gameweek from the starting XI, using the published fixture list.

    Each starter's weekly expectation = season prediction / 38 × fixture factor, where the factor rewards
    weak opponents (opponent's points per match last season vs the league average) and home matches.
    A club's n-th fixture of the season is treated as gameweek n.
    """
    fixtures = fixtures.sort_values("match_date")
    long = pd.concat([
        fixtures.assign(team=fixtures["HomeTeam"], opponent=fixtures["AwayTeam"], home=True),
        fixtures.assign(team=fixtures["AwayTeam"], opponent=fixtures["HomeTeam"], home=False),
    ]).sort_values("match_date")
    long["gameweek"] = long.groupby("team").cumcount() + 1
    long = long[long["gameweek"] <= n_gameweeks]
    average = np.mean(list(club_strength.values()))
    weakest = min(club_strength.values())
    long["opp_strength"] = long["opponent"].map(club_strength).fillna(weakest)     # promoted clubs → weakest
    long["fixture_factor"] = (average / long["opp_strength"]) ** 0.5 * np.where(long["home"], 1 + home_boost, 1 - home_boost)

    weekly = starters[["player", "web_name", "team", score_col]].merge(
        long[["team", "gameweek", "opponent", "home", "fixture_factor"]], on="team")
    weekly["expected_points"] = weekly[score_col] / 38 * weekly["fixture_factor"]
    best = weekly.sort_values("expected_points", ascending=False).drop_duplicates("gameweek").sort_values("gameweek")
    return best[["gameweek", "player", "web_name", "team", "opponent", "home", "expected_points"]].reset_index(drop=True)
