# Premier League Analytics · 2015/16 – 2024/25

**Done by Chin Guan Chung**

A data-science project on ten Premier League seasons. It answers five questions with pandas, seaborn, Plotly and
scikit-learn, then checks every forecast against the 2025/26 season that none of the models saw.

| # | Question | How it is answered |
|:-:|---|---|
| Q1 | Which is the best club? | A weighted 10-season **Club Score**, with a check of whether the winner changes under other weightings |
| Q2 | Who are the best players? | Career **FPL points** and an **Attacking Score** from goals, xG, assists and xA per 90 minutes |
| Q3 | What is the best starting XI? | A gradient-boosting model predicts next-season FPL points **with 80% ranges**; optimisers pick the best XI and a **£100m FPL-legal squad** with a weekly captain plan |
| Q4 | What will FPL prices be next season? | A random forest projects end-of-season FPL prices |
| Q5 | How much does home advantage matter, and how predictable are matches? | Season trends, a **result classifier with a calibration check**, and a Poisson goals model, using form, Elo ratings and betting odds |

Every model is compared with a simple baseline, and all the results and "Key takeaways" are **computed in the
notebook**, so no number in the write-up is typed by hand. See §8 of the notebook for the final answers and
§7 for the 2025/26 reality check.

## Project layout

```
premier_league_analytics_2016_2025.ipynb   The analysis: story, charts and results (run top to bottom)
pl_analytics/                              Reusable code imported by the notebook
    theme.py        Premier League colours, Plotly template, Matplotlib style, tables, result cards
    data.py         Cached downloads and loaders (matches, FPL, Gameweek-1 squads, Understat)
    features.py     League tables, Club Score, match features (form, Elo, odds), player history
    models.py       ML pipelines, XI and FPL-squad optimisers, captain plan, calibration
    charts.py       Pitch chart shared by the notebook and the dashboard
    dashboard.py    Stand-alone HTML dashboard export
tests/                                     pytest checks (official tables, no leakage, FPL rules)
dashboard/index.html                       Interactive charts as one web page (written by the notebook)
data_cache/                                Downloaded CSVs (created on first run)
```

## How to run

Requires Python 3.10 or newer (tested on 3.14).

```bash
pip install -r requirements.txt
```

Then open `premier_league_analytics_2016_2025.ipynb` in Jupyter, VS Code or JupyterLab and run all cells. The first
run downloads about 40 public CSV files into `data_cache/`; later runs work offline. A full run takes about two minutes.

To run the tests (the table tests use the cached data, so run the notebook once first):

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

## The dashboard

The last notebook cell writes **`dashboard/index.html`**: a tabbed page with the interactive Plotly charts (club
trends, league tables, player leaderboards, the predicted XI and FPL squad, price movers, match calibration and the
2025/26 reality check). Open it in any browser; no Python is needed. Plotly.js loads from the jsDelivr CDN, so the
page needs an internet connection.

## Data sources

| Data | Source |
|---|---|
| Match results, stats and betting odds | [football-data.co.uk](https://www.football-data.co.uk/englandm.php) |
| Fantasy Premier League player data, Gameweek-1 squads | [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) |
| Understat goals, xG, assists and xA | [understat_players_aggregated](https://github.com/vibedatascience/understat_players_aggregated) |

The datasets are community-maintained; check each source's terms before redistributing the data.

## Method notes and limitations

* Every model is tested on **later seasons than it was trained on**, and every feature uses only information from
  before the thing being predicted (tests in `tests/test_features.py` check this for the match features).
* League tables are rebuilt from results, so points deductions (Everton and Nottingham Forest in 2023/24) are not
  applied; `tests/test_tables.py` documents this.
* "FPL price" is the Fantasy Premier League game price, not a transfer fee. FPL points are a game metric, not a
  complete measure of football ability.
* Player forecasts are transfer-aware up to Gameweek 1. Later moves, loans and injuries are not visible to them.
* This is an educational analysis, not betting advice.
