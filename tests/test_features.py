"""Pre-match features must only use information from before kick-off (no target leakage)."""
import numpy as np
import pytest
import pandas as pd

from pl_analytics.features import ELO_START, build_match_features


def make_matches(rows, season="2020-21"):
    """rows = [(home, away, home_goals, away_goals), ...] played on consecutive days."""
    df = pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["match_id"] = np.arange(len(df))
    df["match_date"] = pd.Timestamp("2020-09-01") + pd.to_timedelta(df["match_id"], unit="D")
    df["season"] = season
    df["season_start_year"] = int(season[:4])
    df["FTR"] = np.select([df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], ["H", "D"], default="A")
    df["B365H"], df["B365D"], df["B365A"] = 2.0, 3.5, 4.0
    return df


ROWS = [("A", "B", 3, 0), ("C", "D", 1, 1), ("A", "C", 0, 2), ("B", "D", 2, 1), ("D", "A", 1, 0)]


def test_first_match_has_no_form_and_starting_elo():
    data, _ = build_match_features(make_matches(ROWS))
    first = data.iloc[0]
    assert np.isnan(first["home_form_points_5"]) and np.isnan(first["away_form_points_5"])
    assert first["home_matches_played_before"] == 0
    assert first["home_elo"] == ELO_START and first["away_elo"] == ELO_START


def test_form_uses_only_earlier_matches():
    data, _ = build_match_features(make_matches(ROWS))
    # Match 2 (A vs C): A won its only earlier match 3-0, C drew 1-1
    row = data.iloc[2]
    assert row["home_form_points_5"] == 3 and row["home_form_goals_for_5"] == 3
    assert row["away_form_points_5"] == 1


def test_changing_a_result_does_not_change_its_own_features():
    base, _ = build_match_features(make_matches(ROWS))
    changed_rows = list(ROWS)
    changed_rows[2] = ("A", "C", 5, 0)                     # flip match 2's result
    changed, _ = build_match_features(make_matches(changed_rows))
    feature_cols = [c for c in base.columns if c.startswith(("home_", "away_", "elo_", "p_"))]
    pd.testing.assert_series_equal(base.iloc[2][feature_cols], changed.iloc[2][feature_cols], check_names=False)
    assert not base.iloc[4][["away_elo"]].equals(changed.iloc[4][["away_elo"]])   # ...but later matches do change


def test_elo_rises_after_a_win():
    data, ratings = build_match_features(make_matches(ROWS))
    assert data.iloc[2]["home_elo"] > ELO_START            # A beat B in match 0
    assert ratings["B"] < ELO_START


def test_promoted_club_starts_at_lowest_rating():
    season1 = make_matches(ROWS, "2020-21")
    season2 = make_matches([("E", "A", 1, 1)], "2021-22")  # E is promoted
    season2["match_id"] += len(season1)
    data, _ = build_match_features(pd.concat([season1, season2], ignore_index=True))
    _, end_of_season1 = build_match_features(season1)
    # Rule: every summer ratings move 25% back towards the start value; a promoted club gets the lowest of them.
    expected = min(0.75 * r + 0.25 * ELO_START for r in end_of_season1.values())
    first_new = data[data["season"] == "2021-22"].iloc[0]
    assert first_new["home_elo"] == pytest.approx(expected)
    assert first_new["away_elo"] == pytest.approx(0.75 * end_of_season1["A"] + 0.25 * ELO_START)


def test_odds_become_probabilities_that_sum_to_one():
    data, _ = build_match_features(make_matches(ROWS))
    total = data[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(total, 1.0)
