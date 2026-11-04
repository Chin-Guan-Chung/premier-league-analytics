"""Feature engineering: league tables, the Club Score, pre-match features (form, Elo, odds) and player history.

Every feature here uses only information available *before* the thing being predicted:
match features look at earlier matches only, and player features look at earlier seasons
plus the pre-season (Gameweek 1) squad list.
"""
import numpy as np
import pandas as pd

from .data import PLAYER_BASE_FEATURES, season_label

# ── League tables and the Club Score ─────────────────────────────────────────
def team_rows(matches, side):
    """One row per club per match from the home or away perspective (goals for/against, points, W/D/L)."""
    is_home = side == "home"
    team_col, gf_col, ga_col, win_code = ("HomeTeam", "FTHG", "FTAG", "H") if is_home else ("AwayTeam", "FTAG", "FTHG", "A")
    rows = matches[["match_id", "season", team_col, gf_col, ga_col, "FTR"]].copy()
    rows.columns = ["match_id", "season", "team", "gf", "ga", "result"]
    rows["win"] = (rows["result"] == win_code).astype(int)
    rows["draw"] = (rows["result"] == "D").astype(int)
    rows["loss"] = 1 - rows["win"] - rows["draw"]
    rows["points"] = 3 * rows["win"] + rows["draw"]
    return rows


def build_season_standings(matches):
    """Every season's final table rebuilt from the results (Premier League tie-breakers: points, GD, goals scored)."""
    results = pd.concat([team_rows(matches, "home"), team_rows(matches, "away")], ignore_index=True)
    results["gd"] = results["gf"] - results["ga"]
    standings = (
        results.groupby(["season", "team"], as_index=False)
        .agg(MP=("match_id", "count"), W=("win", "sum"), D=("draw", "sum"), L=("loss", "sum"),
             GF=("gf", "sum"), GA=("ga", "sum"), GD=("gd", "sum"), Points=("points", "sum"))
    )
    standings["Points_per_match"] = standings["Points"] / standings["MP"]
    standings["GD_per_match"] = standings["GD"] / standings["MP"]
    standings["Win_rate"] = standings["W"] / standings["MP"]
    standings = standings.sort_values(["season", "Points", "GD", "GF"], ascending=[True, False, False, False])
    standings["Position"] = standings.groupby("season").cumcount() + 1
    return standings


def league_table(match_df):
    """Final league table from one set of results (same rules as build_season_standings)."""
    sides = []
    for team_col, gf, ga, win_code in [("HomeTeam", "FTHG", "FTAG", "H"), ("AwayTeam", "FTAG", "FTHG", "A")]:
        s = match_df[[team_col, gf, ga, "FTR"]].set_axis(["team", "GF", "GA", "FTR"], axis=1)
        s["W"], s["D"] = (s["FTR"] == win_code).astype(int), (s["FTR"] == "D").astype(int)
        sides.append(s)
    table = pd.concat(sides).groupby("team", as_index=False).agg(
        MP=("GF", "size"), W=("W", "sum"), D=("D", "sum"), GF=("GF", "sum"), GA=("GA", "sum"))
    table["L"] = table["MP"] - table["W"] - table["D"]
    table["GD"] = table["GF"] - table["GA"]
    table["Points"] = 3 * table["W"] + table["D"]
    table = table.sort_values(["Points", "GD", "GF"], ascending=False).reset_index(drop=True)
    table.insert(0, "Position", np.arange(1, len(table) + 1))
    return table


def compute_club_score(summary, weights):
    """Min-max scale each metric to 0–1, take the weighted sum and rescale to 0–100."""
    total = 0
    for col, weight in weights.items():
        spread = summary[col].max() - summary[col].min()
        total = total + weight * ((summary[col] - summary[col].min()) / spread if spread else 0.0)
    return 100 * total / sum(weights.values())


# ── Pre-match features: form, Elo, odds ──────────────────────────────────────
ELO_START, ELO_K, ELO_HOME_ADV = 1500, 20, 60       # start rating, update speed, home-advantage bonus (rating points)
ROLLING_SOURCES = {"points": "points", "gf": "goals_for", "ga": "goals_against", "win_flag": "win_rate", "gd": "goal_difference"}
FORM_COLS = [f"form_{label}_5" for label in ROLLING_SOURCES.values()] + ["matches_played_before"]
FORM_FEATURES = [f"{side}_{c}" for side in ("home", "away") for c in FORM_COLS]
ELO_FEATURES = ["home_elo", "away_elo", "elo_diff"]
ODDS_FEATURES = ["p_home", "p_draw", "p_away", "p_over25"]
match_feature_cols = FORM_FEATURES + ELO_FEATURES + ODDS_FEATURES


def build_match_features(match_df):
    """Add form, Elo and odds features to every match. Every feature uses only information from before kick-off.

    Returns (match data with features, final Elo ratings). Written as a function so §8 can rebuild the
    features for the 2025/26 season with exactly the same logic.
    """
    match_df = match_df.sort_values("match_id").reset_index(drop=True)

    # ── Form: one row per club per match (home and away perspectives), in date order ──
    rows = []
    for side, team_col, gf, ga, win_code in [("home", "HomeTeam", "FTHG", "FTAG", "H"), ("away", "AwayTeam", "FTAG", "FTHG", "A")]:
        r = match_df[["match_id", "match_date", team_col, gf, ga, "FTR"]].rename(columns={team_col: "team", gf: "gf", ga: "ga"})
        r["points"] = np.select([r["FTR"] == win_code, r["FTR"] == "D"], [3, 1], default=0)
        r["win_flag"] = (r["FTR"] == win_code).astype(int)
        r["side"] = side
        rows.append(r)
    history = pd.concat(rows, ignore_index=True)
    history["gd"] = history["gf"] - history["ga"]
    history = history.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)
    # Rolling 5-match means of PAST matches only (shift(1) excludes the current match)
    for raw_col, label in ROLLING_SOURCES.items():
        history[f"form_{label}_5"] = history.groupby("team")[raw_col].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    history["matches_played_before"] = history.groupby("team").cumcount()
    # Pivot back to one row per match with home_* and away_* feature columns
    home_form = history[history["side"] == "home"].set_index("match_id")[FORM_COLS].add_prefix("home_")
    away_form = history[history["side"] == "away"].set_index("match_id")[FORM_COLS].add_prefix("away_")
    data = match_df.join(home_form, on="match_id").join(away_form, on="match_id")

    # ── Elo ratings, updated match by match in date order ──
    ratings, home_elo, away_elo, current_season, season_clubs = {}, [], [], None, set()
    for row in data.itertuples():
        if row.season != current_season:
            # Each summer: keep only last season's clubs, pull their ratings 25% back towards average, and give every
            # promoted club (even one returning after years away) the lowest of those ratings, usually a relegated side's.
            ratings = {team: 0.75 * rating + 0.25 * ELO_START for team, rating in ratings.items() if team in season_clubs}
            promoted_rating = min(ratings.values()) if ratings else ELO_START
            current_season, season_clubs = row.season, set()
        season_clubs.update([row.HomeTeam, row.AwayTeam])
        h = ratings.setdefault(row.HomeTeam, promoted_rating)
        a = ratings.setdefault(row.AwayTeam, promoted_rating)
        home_elo.append(h)                                # record the PRE-match ratings as features
        away_elo.append(a)
        expected_home = 1 / (1 + 10 ** ((a - h - ELO_HOME_ADV) / 400))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[row.FTR]
        margin = np.log(abs(row.FTHG - row.FTAG) + 1) + 1 if row.FTHG != row.FTAG else 1.0   # bigger wins move ratings more
        change = ELO_K * margin * (actual_home - expected_home)
        ratings[row.HomeTeam], ratings[row.AwayTeam] = h + change, a - change
    data["home_elo"], data["away_elo"] = home_elo, away_elo
    data["elo_diff"] = data["home_elo"] - data["away_elo"]

    # ── Market-implied probabilities from pre-match odds ──
    implied = 1 / data[["B365H", "B365D", "B365A"]]
    implied = implied.div(implied.sum(axis=1), axis=0)                # remove the bookmaker margin
    data["p_home"], data["p_draw"], data["p_away"] = implied.to_numpy().T
    # Over/under 2.5 goals odds: column names changed over time, so take whichever exists for each season
    def first_available(*cols):
        out = pd.Series(np.nan, index=data.index)
        for col in cols:
            if col in data:
                out = out.combine_first(data[col])
        return out
    over, under = first_available("Avg>2.5", "BbAv>2.5", "B365>2.5"), first_available("Avg<2.5", "BbAv<2.5", "B365<2.5")
    data["p_over25"] = (1 / over) / (1 / over + 1 / under)
    return data, ratings


# ── Player history and pre-season features ───────────────────────────────────
model_feature_cols = [f"prev_{c}" for c in PLAYER_BASE_FEATURES] + ["prev_position"]        # used by the price model
HISTORY_FEATURES = ["prev2_total_points", "prev2_minutes", "prev_points_per90", "prev_team_ppm", "prev_seasons_played"]
PRESEASON_FEATURES = ["start_team_ppm", "changed_club", "start_price"]                       # from the GW1 squad list
points_feature_cols = model_feature_cols + HISTORY_FEATURES + PRESEASON_FEATURES             # used by the points model


def build_player_panel(fpl_players, season_standings, gw1_squads):
    """Lagged player features for every season, plus projection rows for the season after the data ends.

    Returns a dict with:
      player_lagged    – training rows: season t stats as targets, season t-1 (and t-2) stats as features
      projection_all   – every final-season player, projected one season ahead (transfer-blind)
      projection_rows  – the same, but only players in the next season's GW1 squad list (transfer-aware)
      latest_year      – first year of the final season in the data (e.g. 2024 for 2024-25)
    """
    # Add an empty placeholder row for every player in the final season, labelled as the NEXT season.
    # The same shift() logic then builds the projection features with no special-case code.
    latest_year = fpl_players["season_start_year"].max()
    future = fpl_players[fpl_players["season_start_year"] == latest_year].copy()
    future[PLAYER_BASE_FEATURES] = np.nan
    future["season_start_year"], future["season"] = latest_year + 1, season_label(latest_year + 1)
    panel = pd.concat([fpl_players, future]).sort_values(["player_key", "season_start_year"]).reset_index(drop=True)

    grouped = panel.groupby("player_key")
    for col in PLAYER_BASE_FEATURES + ["position", "team", "season_start_year"]:
        panel[f"prev_{col}"] = grouped[col].shift(1)                  # the player's previous season
    panel["prev2_total_points"] = grouped["total_points"].shift(2)     # two seasons back
    panel["prev2_minutes"] = grouped["minutes"].shift(2)
    two_back = grouped["season_start_year"].shift(2)
    panel.loc[panel["season_start_year"] - two_back != 2, ["prev2_total_points", "prev2_minutes"]] = np.nan
    panel["prev_points_per90"] = np.where(panel["prev_minutes"] > 0, panel["prev_total_points"] / panel["prev_minutes"] * 90, 0)
    panel["prev_seasons_played"] = grouped.cumcount()                  # earlier FPL seasons for this player

    # Strength of the player's club last season (points per match from the rebuilt league tables)
    club_ppm = {(int(s[:4]), t): v for s, t, v in season_standings[["season", "team", "Points_per_match"]].itertuples(index=False)}
    panel["prev_team_ppm"] = [club_ppm.get((y, t), np.nan) for y, t in zip(panel["prev_season_start_year"], panel["prev_team"])]

    # Pre-season (GW1) information for the target season: club, starting price, and that club's strength last season.
    # Promoted clubs have no top-flight PPM, so they get the weakest PPM of the season before.
    panel = panel.merge(gw1_squads, on=["player_id", "season_start_year"], how="left")
    weakest_ppm = {int(s[:4]): v for s, v in season_standings.groupby("season")["Points_per_match"].min().items()}
    panel["start_team_ppm"] = [club_ppm.get((y - 1, t), weakest_ppm.get(y - 1, np.nan)) if isinstance(t, str) else np.nan
                               for y, t in zip(panel["season_start_year"], panel["start_team"])]
    panel["changed_club"] = np.where(panel["start_team"].isna() | panel["prev_team"].isna(), np.nan,
                                     (panel["start_team"] != panel["prev_team"]).astype(float))

    # Keep only true consecutive seasons (e.g. 2022-23 → 2023-24), not a gap after relegation or a loan abroad.
    consecutive = (panel["season_start_year"] - panel["prev_season_start_year"]) == 1
    projection_all = panel[consecutive & (panel["season_start_year"] == latest_year + 1)].copy()
    projection_rows = projection_all[projection_all["start_team"].notna()].copy()
    player_lagged = panel[consecutive & (panel["season_start_year"] <= latest_year)].copy()
    player_lagged = player_lagged.dropna(subset=["prev_total_points", "total_points"])
    player_lagged["target_season_year"] = player_lagged["season_start_year"].astype(int)
    return {"player_lagged": player_lagged, "projection_all": projection_all,
            "projection_rows": projection_rows, "latest_year": int(latest_year)}


def as_projection_table(rows, transfer_aware=True):
    """Projection rows with the plain stat columns filled from last season (for "2024/25 points / price" tables).

    With transfer_aware=True the club is the GW1 club for the new season; otherwise last season's club.
    """
    rows = rows.copy()
    for col in PLAYER_BASE_FEATURES:
        rows[col] = rows[f"prev_{col}"]
    rows["team"] = rows["start_team"] if transfer_aware else rows["prev_team"]
    return rows
