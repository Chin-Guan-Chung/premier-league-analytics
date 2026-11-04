"""The league tables rebuilt from match results must match the official Premier League tables.

These tests read the cached football-data.co.uk files in data_cache/ and are skipped if the notebook
has not been run yet (no network access is needed).
"""
import pytest

from pl_analytics.data import CACHE_DIR, SEASON_LABELS, load_match_season, season_code
from pl_analytics.features import build_season_standings, league_table

# Official champions and runners-up, with points (source: premierleague.com final tables)
OFFICIAL_TOP_TWO = {
    "2015-16": (("Leicester", 81), ("Arsenal", 71)),
    "2016-17": (("Chelsea", 93), ("Tottenham", 86)),
    "2017-18": (("Man City", 100), ("Man United", 81)),
    "2018-19": (("Man City", 98), ("Liverpool", 97)),
    "2019-20": (("Liverpool", 99), ("Man City", 81)),
    "2020-21": (("Man City", 86), ("Man United", 74)),
    "2021-22": (("Man City", 93), ("Liverpool", 92)),
    "2022-23": (("Man City", 89), ("Arsenal", 84)),
    "2023-24": (("Man City", 91), ("Arsenal", 89)),
    "2024-25": (("Liverpool", 84), ("Arsenal", 74)),
}


def cached(season):
    return (CACHE_DIR / f"football_data_E0_{season_code(season)}.csv").exists()


@pytest.fixture(scope="module")
def standings():
    if not all(cached(s) for s in SEASON_LABELS):
        pytest.skip("match data not cached yet: run the notebook once first")
    import pandas as pd
    matches = pd.concat([load_match_season(s) for s in SEASON_LABELS], ignore_index=True)
    matches = matches.copy()
    matches["match_id"] = range(len(matches))
    return build_season_standings(matches)


@pytest.mark.parametrize("season", SEASON_LABELS)
def test_every_season_is_complete(standings, season):
    table = standings[standings["season"] == season]
    assert len(table) == 20
    assert (table["MP"] == 38).all()
    assert table["W"].sum() == table["L"].sum()                 # every win is someone's loss
    assert table["GF"].sum() == table["GA"].sum()


@pytest.mark.parametrize("season", SEASON_LABELS)
def test_champion_and_runner_up_match_official_table(standings, season):
    table = standings[standings["season"] == season].sort_values("Position")
    (champion, champion_pts), (runner_up, runner_up_pts) = OFFICIAL_TOP_TWO[season]
    assert tuple(table.iloc[0][["team", "Points"]]) == (champion, champion_pts)
    assert tuple(table.iloc[1][["team", "Points"]]) == (runner_up, runner_up_pts)


def test_points_deductions_are_not_applied(standings):
    """Documented limitation: tables are rebuilt from results, so 2023-24 deductions are absent.

    Official 2023-24: Everton 40 pts after an 8-point deduction, Nottingham Forest 32 after a 4-point deduction.
    """
    table = standings[standings["season"] == "2023-24"].set_index("team")["Points"]
    assert table["Everton"] == 40 + 8
    assert table["Nott'm Forest"] == 32 + 4


def test_league_table_agrees_with_season_standings(standings):
    if not cached("2024-25"):
        pytest.skip("2024-25 not cached")
    one = league_table(load_match_season("2024-25"))
    rebuilt = standings[standings["season"] == "2024-25"].sort_values("Position")
    assert one["team"].tolist() == rebuilt["team"].tolist()
    assert one["Points"].tolist() == rebuilt["Points"].tolist()
