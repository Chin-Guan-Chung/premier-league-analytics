"""Static dashboard export: write the notebook's interactive Plotly charts into one self-contained HTML page.

The page needs no Python or Jupyter to view. Plotly.js is loaded from the jsDelivr CDN, so the file stays small.
"""
import html
from pathlib import Path

import plotly
import plotly.offline

from .theme import PL, PL_FONT

PLOTLY_CDN = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{plotly.offline.get_plotlyjs_version()}/plotly.min.js"


def write_dashboard(sections, path, title, subtitle="", author=""):
    """Write `sections` = {tab name: [(caption, plotly figure), ...]} as a tabbed HTML page. Returns the path."""
    tabs, panels = [], []
    for i, (name, figures) in enumerate(sections.items()):
        tab_id = f"tab{i}"
        tabs.append(f'<button class="tab{" active" if i == 0 else ""}" data-target="{tab_id}">{html.escape(name)}</button>')
        cards = "".join(
            f'<figure class="card"><figcaption>{html.escape(caption)}</figcaption>'
            f'{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})}</figure>'
            for caption, fig in figures if fig is not None)
        panels.append(f'<section id="{tab_id}" class="panel{" active" if i == 0 else ""}">{cards}</section>')

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{ --purple: {PL['purple']}; --pink: {PL['pink']}; --cyan: {PL['cyan']}; --green: {PL['green']}; --light: {PL['light']}; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: {PL_FONT}; background: var(--light); color: var(--purple); }}
  header {{ background: linear-gradient(100deg, var(--purple) 0%, #5A0F63 70%, var(--pink) 140%); color: white;
           padding: 28px 24px 0; border-bottom: 4px solid var(--green); }}
  header h1 {{ margin: 0 0 6px; font-size: clamp(22px, 4vw, 34px); }}
  header p {{ margin: 0 0 4px; opacity: .9; }}
  header .author {{ color: var(--cyan); font-weight: 700; letter-spacing: .04em; }}
  nav {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 18px; }}
  .tab {{ border: 0; border-radius: 8px 8px 0 0; padding: 10px 16px; font: inherit; font-weight: 700; cursor: pointer;
          background: rgba(255,255,255,.12); color: white; }}
  .tab.active {{ background: var(--light); color: var(--purple); }}
  main {{ max-width: 1240px; margin: 0 auto; padding: 20px 16px 40px; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .card {{ background: white; border-radius: 12px; margin: 0 0 20px; padding: 12px; overflow-x: auto;
           box-shadow: 0 2px 10px rgba(55,0,60,.08); }}
  figcaption {{ font-weight: 700; padding: 4px 8px 8px; border-left: 4px solid var(--pink); margin-bottom: 6px; }}
  footer {{ text-align: center; font-size: 13px; padding: 0 16px 30px; opacity: .75; }}
</style></head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>{html.escape(subtitle)}</p>
  {f'<p class="author">{html.escape(author)}</p>' if author else ''}
  <nav>{''.join(tabs)}</nav>
</header>
<main>{''.join(panels)}</main>
<footer>Data: football-data.co.uk · vaastav/Fantasy-Premier-League · Understat. Educational analysis, not betting advice.</footer>
<script>
  document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab, .panel').forEach(el => el.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(btn.dataset.target);
    panel.classList.add('active');
    // Charts drawn while hidden have no width; resize them now that the tab is visible.
    panel.querySelectorAll('.js-plotly-plot').forEach(plot => Plotly.Plots.resize(plot));
  }}));
</script>
</body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
