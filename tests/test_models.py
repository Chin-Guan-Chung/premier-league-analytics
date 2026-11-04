"""Team selection must obey the rules: legal formations, FPL budget, squad shape and the 3-per-club limit."""
import numpy as np
import pandas as pd
import pytest

from pl_analytics.models import (
    FPL_SQUAD_SIZE, FPL_XI_LIMITS, brier_score, captain_plan, interval_coverage, pick_fpl_squad, select_best_xi,
)


@pytest.fixture
def player_pool():
    """120 synthetic players across 10 clubs, with prices loosely linked to predicted points."""
    rng = np.random.default_rng(0)
    positions = np.repeat(["GK", "DEF", "MID", "FWD"], [20, 40, 40, 20])
    points = rng.uniform(20, 220, len(positions))
    return pd.DataFrame({
        "player": [f"P{i}" for i in range(len(positions))],
        "web_name": [f"P{i}" for i in range(len(positions))],
        "position": positions,
        "team": [f"Club {i % 10}" for i in range(len(positions))],
        "pred": points,
        "price": np.round(4 + points / 25 + rng.uniform(-0.5, 0.5, len(positions)), 1),
    })


def test_best_xi_is_a_legal_formation(player_pool):
    xi = select_best_xi(player_pool, score_col="pred")
    counts = xi["position"].value_counts()
    assert len(xi) == 11
    for pos, (low, high) in FPL_XI_LIMITS.items():
        assert low <= counts.get(pos, 0) <= high


def test_fpl_squad_obeys_every_rule(player_pool):
    squad = pick_fpl_squad(player_pool, "pred", price_col="price", budget=100.0)
    starters = squad[squad["starter"]]
    assert len(squad) == 15
    assert squad["price"].sum() <= 100.0 + 1e-9
    assert squad["position"].value_counts().to_dict() == FPL_SQUAD_SIZE
    assert squad["team"].value_counts().max() <= 3
    assert len(starters) == 11
    counts = starters["position"].value_counts()
    for pos, (low, high) in FPL_XI_LIMITS.items():
        assert low <= counts.get(pos, 0) <= high
    assert squad["captain"].sum() == 1
    assert squad.loc[squad["captain"], "starter"].all()             # the captain must start
    # The captain should be the highest-scoring starter (that is what doubling rewards)
    assert squad.loc[squad["captain"], "pred"].iloc[0] == starters["pred"].max()


def test_tighter_budget_never_scores_more(player_pool):
    rich = pick_fpl_squad(player_pool, "pred", price_col="price", budget=100.0)
    poor = pick_fpl_squad(player_pool, "pred", price_col="price", budget=80.0)
    assert poor["price"].sum() <= 80.0 + 1e-9
    assert poor.loc[poor["starter"], "pred"].sum() <= rich.loc[rich["starter"], "pred"].sum() + 1e-9


def test_captain_plan_picks_a_starter_every_gameweek(player_pool):
    squad = pick_fpl_squad(player_pool, "pred", price_col="price")
    starters = squad[squad["starter"]]
    clubs = [f"Club {i}" for i in range(10)]
    fixtures = pd.DataFrame([(pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * gw), clubs[i], clubs[(i + gw + 1) % 10])
                             for gw in range(4) for i in range(0, 10, 2)], columns=["match_date", "HomeTeam", "AwayTeam"])
    strength = {club: 1.0 + 0.1 * i for i, club in enumerate(clubs)}
    plan = captain_plan(starters, fixtures, strength, "pred", n_gameweeks=4)
    assert plan["gameweek"].tolist() == sorted(plan["gameweek"].unique().tolist())
    assert set(plan["player"]) <= set(starters["player"])


def test_interval_coverage_and_brier_score():
    cov = interval_coverage([1, 5, 10], [0, 6, 9], [2, 7, 11])
    assert cov["Coverage"] == pytest.approx(2 / 3)
    assert cov["Average width"] == pytest.approx(5 / 3)
    classes = ["A", "D", "H"]
    perfect = np.array([[0, 0, 1], [1, 0, 0]])
    assert brier_score(["H", "A"], perfect, classes) == 0
    assert brier_score(["H", "A"], np.full((2, 3), 1 / 3), classes) == pytest.approx(2 / 3)
