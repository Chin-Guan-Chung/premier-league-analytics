"""Data loading: cached downloads, match results, FPL player-seasons, Gameweek-1 squads and Understat stats.

Every loader reads from ./data_cache when the file is already there, so after the first run the
project works offline. Loaders return tidy DataFrames plus, where useful, a load report.
"""
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Configuration ────────────────────────────────────────────────────────────
SEASON_LABELS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025)]     # 2015-16 … 2024-25
FPL_SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2016, 2025)]       # 2016-17 … 2024-25

MATCH_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
FPL_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
UNDERSTAT_URL = ("https://raw.githubusercontent.com/vibedatascience/understat_players_aggregated/"
                 "main/understat_players_aggregated_2014_2024.csv")
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"           # <project>/data_cache

FPL_CANDIDATE_FILES = ["players_raw.csv", "cleaned_players.csv"]      # full export first, cleaned file as fallback
# FPL spells some clubs differently from football-data.co.uk; map them onto the match-data names.
TEAM_ALIASES = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
                "Nottingham Forest": "Nott'm Forest"}
POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Numeric FPL stats used for rankings and as model features (missing in older seasons → NaN, imputed later)
PLAYER_BASE_FEATURES = [
    "total_points", "points_per_game", "minutes", "goals_scored", "assists", "clean_sheets",
    "bonus", "bps", "starts", "expected_goals", "expected_assists", "expected_goal_involvements",
    "saves", "now_cost", "selected_by_percent", "ict_index", "influence", "creativity", "threat",
]


# ── Small helpers ────────────────────────────────────────────────────────────
def read_csv_cached(url, cache_name, **kwargs):
    """Download a CSV once, cache it in ./data_cache, then always read from the cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        response = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0 PremierLeagueResearchNotebook/1.0"})
        response.raise_for_status()                       # fail loudly on 404 / 500
        cache_path.write_bytes(response.content)
    return pd.read_csv(cache_path, low_memory=False, **kwargs)


def season_code(label):
    """'2015-16' → '1516' (the football-data.co.uk URL format)."""
    return label[2:4] + label[-2:]


def season_label(start_year):
    """2024 → '2024-25'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def normalise_name(value):
    """Lower-case, accent-free, alphanumeric-only name used as a join key ('Martin Ødegaard' → 'martinodegaard')."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def safe_numeric(df, columns):
    """Convert the listed columns to numbers in place; anything non-numeric becomes NaN."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def first_existing(df, names):
    """First column name from `names` that exists in df (handles schema differences between seasons)."""
    return next((name for name in names if name in df.columns), None)


def map_position(raw):
    """Map numeric (1-4) or text ('GK', 'GKP', 'DEF', ...) FPL positions onto GK / DEF / MID / FWD.

    Each value is mapped individually, because concatenating seasons can mix numbers and strings in one column.
    """
    numeric = pd.to_numeric(raw, errors="coerce").map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    text = raw.astype(str).str.upper()
    from_text = pd.Series(np.select(
        [text.str.contains("GK|GOALKEEPER"), text.str.contains("DEF"), text.str.contains("MID"),
         text.str.contains("FWD|FORWARD|STRIKER")],
        ["GK", "DEF", "MID", "FWD"], default="UNK"), index=raw.index)
    return numeric.fillna(from_text)


# ── Match results ────────────────────────────────────────────────────────────
def load_match_season(season):
    """Download (or read from cache) one season of results and clean it into a consistent schema."""
    code = season_code(season)
    one = read_csv_cached(MATCH_URL_TEMPLATE.format(season_code=code), f"football_data_E0_{code}.csv", encoding="latin-1")
    one.columns = [str(c).strip() for c in one.columns]          # some files have stray spaces in headers
    missing = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"} - set(one.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")

    one = one.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()   # drop blank trailing rows
    one["season"] = season
    one["season_start_year"] = int(season[:4])
    # Dates are dd/mm/yy in early files and dd/mm/yyyy later, so parse in mixed mode with day first.
    one["match_date"] = pd.to_datetime(one["Date"], dayfirst=True, format="mixed", errors="coerce")
    one = safe_numeric(one, ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"])
    one[["FTHG", "FTAG"]] = one[["FTHG", "FTAG"]].astype(int)
    # Recompute the full-time result from the score so the label is always consistent: H / D / A.
    one["FTR"] = np.select([one["FTHG"] > one["FTAG"], one["FTHG"] == one["FTAG"]], ["H", "D"], default="A")
    return one


def load_matches(seasons=SEASON_LABELS):
    """Load and combine several seasons. Returns (matches, load_report); a failing season is skipped, not fatal."""
    frames, log = [], []
    for season in seasons:
        try:
            frames.append(load_match_season(season))
            log.append({"season": season, "rows": len(frames[-1]), "status": "OK"})
        except Exception as exc:
            log.append({"season": season, "rows": 0, "status": f"Skipped: {type(exc).__name__}: {exc}"})
    if not frames:
        raise RuntimeError("No match CSVs loaded. Check your internet connection and the football-data.co.uk URLs.")
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.drop_duplicates(subset=["season", "match_date", "HomeTeam", "AwayTeam"], keep="last")
    matches = matches.sort_values(["match_date", "HomeTeam"]).reset_index(drop=True)
    matches["match_id"] = np.arange(len(matches))                       # stable ID used to join features later
    return matches, pd.DataFrame(log)


# ── Fantasy Premier League ───────────────────────────────────────────────────
def fpl_team_lookup(season, matches):
    """Return {FPL team id → club name} for one season."""
    try:
        teams = read_csv_cached(f"{FPL_BASE_URL}/{season}/teams.csv", f"fpl_{season}_teams.csv")
        return {int(i): TEAM_ALIASES.get(n, n) for i, n in zip(teams["id"], teams["name"])}
    except Exception:
        # teams.csv does not exist before 2019-20; FPL IDs are alphabetical, so rebuild from the match data.
        clubs = sorted(matches.loc[matches["season"] == season, "HomeTeam"].unique())
        return {i + 1: name for i, name in enumerate(clubs)} if len(clubs) == 20 else {}


def load_fpl_raw(seasons, matches):
    """Load each season's FPL player export and attach club names. Returns (fpl_raw, load_report)."""
    frames, log = [], []
    for season in seasons:
        loaded, last_error = None, None
        for filename in FPL_CANDIDATE_FILES:                             # try each file until one loads
            try:
                loaded = read_csv_cached(f"{FPL_BASE_URL}/{season}/{filename}", f"fpl_{season}_{filename}")
                if loaded.empty:
                    raise ValueError("CSV is empty")
                log.append({"season": season, "file": filename, "rows": len(loaded), "status": "OK"})
                break
            except Exception as exc:
                loaded, last_error = None, f"{type(exc).__name__}: {exc}"
        if loaded is None:
            log.append({"season": season, "file": "—", "rows": 0, "status": f"Skipped: {last_error}"})
            continue
        loaded = loaded.copy()
        loaded["season"] = season
        loaded["season_start_year"] = int(season[:4])
        if "team" in loaded.columns and pd.to_numeric(loaded["team"], errors="coerce").notna().all():
            loaded["team_name"] = pd.to_numeric(loaded["team"]).map(fpl_team_lookup(season, matches))
        frames.append(loaded)
    fpl_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return fpl_raw, pd.DataFrame(log)


def clean_fpl_players(fpl_raw):
    """One tidy row per player per season: names, stable ID, position, club, numeric stats and £m price."""
    source = fpl_raw
    first_col = first_existing(source, ["first_name", "firstName"])
    second_col = first_existing(source, ["second_name", "secondName"])
    players = pd.DataFrame()

    # Full name ("Mohamed Salah") for joins and tables; web_name ("M.Salah") for compact chart labels.
    players["player"] = (source[first_col].fillna("").astype(str).str.strip() + " "
                         + source[second_col].fillna("").astype(str).str.strip()).str.strip()
    players["web_name"] = source["web_name"] if "web_name" in source else players["player"]
    players["player_key_base"] = players["player"].map(normalise_name)
    # FPL's `code` is a stable player ID that never changes across seasons, unlike names and positions.
    players["player_id"] = pd.to_numeric(source["code"], errors="coerce") if "code" in source else np.nan
    players["season"] = source["season"]
    players["season_start_year"] = source["season_start_year"].astype(int)
    players["position"] = map_position(source[first_existing(source, ["element_type", "position"])])
    players["team"] = source["team_name"].fillna("Unknown") if "team_name" in source else "Unknown"
    for column in PLAYER_BASE_FEATURES:
        players[column] = pd.to_numeric(source[column], errors="coerce") if column in source else np.nan
    players["now_cost"] = players["now_cost"] / 10.0          # tenths of £m → £m

    # Keep real players only (drops blank names and the 2024-25 'assistant manager' cards)
    players = players[players["player_key_base"].ne("") & players["position"].isin(POSITIONS)]
    # Link seasons by the stable FPL ID; fall back to name + position only if an ID is missing.
    players["player_key"] = np.where(players["player_id"].notna(),
                                     "fpl_" + players["player_id"].astype("Int64").astype(str),
                                     players["player_key_base"] + "_" + players["position"])
    # If a player appears twice in a season, keep the row with more minutes.
    return (players.sort_values(["season_start_year", "minutes"], ascending=[True, False])
            .drop_duplicates(["player_key", "season"], keep="first").reset_index(drop=True))


def load_gw1_squads(seasons, matches):
    """Pre-season squads: every player registered at Gameweek 1, with their club and starting price (£m)."""
    frames = []
    for season in seasons:
        try:
            gw1 = read_csv_cached(f"{FPL_BASE_URL}/{season}/gws/gw1.csv", f"fpl_{season}_gw1.csv", encoding="latin-1")
            # players_raw maps that season's player number (`element`) to the stable ID (`code`)
            ids = read_csv_cached(f"{FPL_BASE_URL}/{season}/players_raw.csv", f"fpl_{season}_players_raw.csv")
        except Exception as exc:
            print(f"GW1 squad for {season} skipped: {type(exc).__name__}: {exc}")
            continue
        gw1 = (gw1.drop_duplicates("element")                              # double gameweeks list a player twice
               .merge(ids[["id", "code", "team"]].rename(columns={"team": "team_id"}), left_on="element", right_on="id"))
        if "team" in gw1:                                                  # club name as listed at GW1 (2020-21 on)
            start_team = gw1["team"].map(lambda name: TEAM_ALIASES.get(name, name))
        else:                                                              # older files: that season's FPL club ID
            start_team = gw1["team_id"].map(fpl_team_lookup(season, matches))
        frames.append(pd.DataFrame({"player_id": gw1["code"].to_numpy(), "season_start_year": int(season[:4]),
                                    "start_team": start_team.to_numpy(), "start_price": gw1["value"].to_numpy() / 10}))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "season_start_year"])


def load_fpl_actual(season, matches):
    """One season's final FPL results (points, minutes, end price, defensive contributions), keyed by stable ID."""
    raw = read_csv_cached(f"{FPL_BASE_URL}/{season}/players_raw.csv", f"fpl_{season}_players_raw.csv")
    actual = pd.DataFrame({
        "player": (raw["first_name"].fillna("") + " " + raw["second_name"].fillna("")).str.strip(),
        "web_name": raw["web_name"],
        "position": map_position(raw["element_type"]),
        "team": pd.to_numeric(raw["team"]).map(fpl_team_lookup(season, matches)),
        "actual_points": raw["total_points"],
        "actual_minutes": raw["minutes"],
        "actual_price": raw["now_cost"] / 10,                            # end-of-season price, £m
        "defensive_contribution": raw.get("defensive_contribution", np.nan),
        "player_id": pd.to_numeric(raw["code"], errors="coerce"),        # stable FPL ID
    })
    actual = actual[actual["position"].isin(POSITIONS)].reset_index(drop=True)
    actual["team_2526"] = actual["team"]
    actual["actual_rank"] = actual["actual_points"].rank(ascending=False, method="min").astype(int)
    return actual


# ── Understat ────────────────────────────────────────────────────────────────
def load_understat_players(first_year=2015, last_year=2024):
    """EPL player-seasons with goals, xG, assists, xA and per-90 rates (mid-season transfers summed)."""
    try:
        understat_raw = read_csv_cached(UNDERSTAT_URL, "understat_players_aggregated_2014_2024.csv")
    except Exception as exc:
        warnings.warn(f"Understat data could not be loaded ({type(exc).__name__}: {exc}); attacking metrics skipped.")
        return pd.DataFrame()

    # Map each target column onto whichever source column name exists
    aliases = {
        "player_id": ["id", "player_id"], "player": ["player_name", "name"], "team": ["team_title", "team"],
        "league": ["league"], "year": ["year"], "season": ["season"], "goals": ["goals"], "xG": ["xG", "xg"],
        "assists": ["assists"], "xA": ["xA", "xa"], "shots": ["shots"], "key_passes": ["key_passes"],
        "games": ["games"], "minutes": ["time", "minutes"], "position": ["primary_position", "position"],
    }
    us = pd.DataFrame({target: understat_raw[first_existing(understat_raw, names)]
                       if first_existing(understat_raw, names) else np.nan
                       for target, names in aliases.items()})
    us = safe_numeric(us, ["year", "goals", "xG", "assists", "xA", "shots", "key_passes", "games", "minutes"])
    us = us[us["league"].astype(str).str.upper().isin(["EPL", "PREMIER LEAGUE", "ENGLISH PREMIER LEAGUE"])]
    us = us[us["year"].between(first_year, last_year)].dropna(subset=["player_id", "player", "year"])

    # Sum mid-season transfers into one row per player-season
    players = us.groupby(["player_id", "player", "year"], as_index=False).agg(
        season=("season", "first"), team=("team", lambda s: " / ".join(sorted(set(map(str, s))))),
        goals=("goals", "sum"), xG=("xG", "sum"), assists=("assists", "sum"), xA=("xA", "sum"),
        shots=("shots", "sum"), key_passes=("key_passes", "sum"), games=("games", "sum"),
        minutes=("minutes", "sum"), position=("position", "first"),
    )
    # Understat codes positions as GK / D / M / F / S (sub); align them with the FPL labels.
    players["position"] = players["position"].astype(str).str.upper().map(
        lambda p: "GK" if p.startswith("GK") else {"D": "DEF", "M": "MID", "F": "FWD"}.get(p[:1], "SUB"))
    # Per-90-minute rates make part-time and full-time players comparable
    for stat in ["goals", "xG", "assists", "xA"]:
        players[f"{stat}_per90"] = np.where(players["minutes"] > 0, players[stat] / players["minutes"] * 90, np.nan)
    return players
