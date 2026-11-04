"""Premier League visual theme.

Brand colours, a Plotly template, matching Matplotlib/seaborn styling, and small HTML helpers
(KPI tiles, result cards, styled tables and generated "Key takeaways") used throughout the notebook.
Importing this module registers the `premier_league` Plotly template and applies the seaborn theme.
"""
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from IPython.display import HTML, Markdown, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ── Brand palette ────────────────────────────────────────────────────────────
PL = {
    "purple": "#37003C",       # primary brand colour: text, headers, default bars
    "purple_mid": "#5A0F63",   # lighter purple for gradients
    "pink": "#E90052",         # magenta accent: highlights and "the answer"
    "cyan": "#04F5FF",         # bright accent for dark backgrounds
    "green": "#00FF85",        # bright accent for dark backgrounds
    "cyan_dk": "#00B8C8",      # readable cyan on white
    "green_dk": "#00B860",     # readable green on white
    "violet": "#963CFF",
    "grey": "#76766F",
    "light": "#F4F0F5",        # pale lilac panel background
    "grid": "#E6DCE8",         # subtle grid lines
}
PL_FONT = "Premier Sans, Radikal, Helvetica Neue, Arial, sans-serif"
PL_COLORWAY = [PL["purple"], PL["pink"], PL["cyan_dk"], PL["violet"], PL["green_dk"], "#FF8A00", PL["grey"]]

# FPL-style position colours, used consistently in every player chart
POS_COLORS = {"GK": PL["violet"], "DEF": PL["green_dk"], "MID": PL["cyan_dk"], "FWD": PL["pink"],
              "SUB": PL["grey"], "UNK": PL["grey"]}
POS_ORDER = ["GK", "DEF", "MID", "FWD"]

# Primary kit colours keyed by football-data.co.uk club names (used for club-level charts)
CLUB_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#670E36", "Bournemouth": "#B50E12", "Brentford": "#E30613",
    "Brighton": "#0057B8", "Burnley": "#6C1D45", "Cardiff": "#0070B5", "Chelsea": "#034694",
    "Crystal Palace": "#1B458F", "Everton": "#003399", "Fulham": "#1A1A1A", "Huddersfield": "#0E63AD",
    "Hull": "#F5A12D", "Ipswich": "#3A64A3", "Leeds": "#FFCD00", "Leicester": "#003090",
    "Liverpool": "#C8102E", "Luton": "#F78F1E", "Man City": "#6CABDD", "Man United": "#DA291C",
    "Middlesbrough": "#E11B22", "Newcastle": "#241F20", "Norwich": "#00A650", "Nott'm Forest": "#DD0000",
    "Sheffield United": "#EE2737", "Southampton": "#D71920", "Stoke": "#E03A3E", "Sunderland": "#EB172B",
    "Swansea": "#121212", "Tottenham": "#132257", "Watford": "#FBEE23", "West Brom": "#122F67",
    "West Ham": "#7A263A", "Wolves": "#FDB913",
}
BIG_SIX = ["Man City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Man United"]


def club_color(team):
    """Kit colour for a club; falls back to PL purple for unknown names."""
    return CLUB_COLORS.get(team, PL["purple"])


def text_on(hex_color):
    """Choose purple or white text, whichever is more readable on the given background colour."""
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return PL["purple"] if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else "white"


# ── Plotly template ──────────────────────────────────────────────────────────
PL_SEQ = [[0, PL["light"]], [0.5, PL["pink"]], [1, PL["purple"]]]          # low → high
PL_DIV = [[0, PL["pink"]], [0.5, "#FFFFFF"], [1, PL["green_dk"]]]          # negative ↔ positive

pio.templates["premier_league"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=PL_FONT, size=13, color=PL["purple"]),
        title=dict(font=dict(family=PL_FONT, size=21, color=PL["purple"]), x=0.01, xanchor="left", y=0.96),
        paper_bgcolor="white", plot_bgcolor="white", colorway=PL_COLORWAY,
        margin=dict(l=60, r=30, t=110, b=60),
        xaxis=dict(showgrid=False, linecolor=PL["purple"], linewidth=2, ticks="outside",
                   tickcolor=PL["purple"], zeroline=False, title=dict(standoff=12)),
        yaxis=dict(gridcolor=PL["grid"], zeroline=False, linecolor=PL["grid"], title=dict(standoff=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PL["purple"], bordercolor=PL["pink"],
                        font=dict(family=PL_FONT, color="white", size=12)),
        colorscale=dict(sequential=PL_SEQ, diverging=PL_DIV),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12)),
    ),
    data=dict(scatter=[go.Scatter(marker=dict(line=dict(width=1, color="white")))]),
)
pio.templates.default = "premier_league"
px.defaults.color_continuous_scale = PL_SEQ


def pl_title(title, subtitle=None):
    """Bold PL-style chart title with an optional grey subtitle line (Plotly)."""
    sub = f"<br><span style='font-size:13px;color:{PL['grey']}'>{subtitle}</span>" if subtitle else ""
    return f"<b>{title}</b>{sub}"


# ── Matplotlib / seaborn theme ───────────────────────────────────────────────
sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PL["purple"], "axes.labelcolor": PL["purple"], "text.color": PL["purple"],
    "xtick.color": PL["purple"], "ytick.color": PL["purple"],
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": PL["grid"], "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 15, "axes.titlelocation": "left",
    "axes.labelsize": 11, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.prop_cycle": plt.cycler(color=PL_COLORWAY), "legend.frameon": False,
})
PL_CMAP = LinearSegmentedColormap.from_list("pl_seq", [PL["light"], PL["pink"], PL["purple"]])


def pl_mpl_title(ax, title, subtitle=None):
    """Left-aligned bold title + grey subtitle, with the PL purple/magenta header strip on the figure."""
    ax.set_title(title, loc="left", pad=28 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=PL["grey"], va="bottom")
    fig = ax.figure
    fig.add_artist(Rectangle((0, 0.985), 1, 0.015, transform=fig.transFigure, color=PL["purple"], zorder=10))
    fig.add_artist(Rectangle((0, 0.977), 0.2, 0.008, transform=fig.transFigure, color=PL["pink"], zorder=10))


def show_mpl(fig):
    """Tight layout that leaves room for the header strip, then render."""
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    plt.show()


# ── HTML helpers: KPI tiles, result cards and styled tables ─────────────────
def stat_tiles(items):
    """Row of purple KPI tiles. items = [(label, value), ...]"""
    tiles = "".join(
        f"<div style='flex:1;min-width:130px;background:{PL['purple']};color:white;border-radius:10px;"
        f"padding:12px 14px;border-bottom:4px solid {PL['green']}'>"
        f"<div style='font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:{PL['cyan']}'>{label}</div>"
        f"<div style='font-size:24px;font-weight:800;margin-top:2px'>{value}</div></div>"
        for label, value in items
    )
    display(HTML(f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px;font-family:{PL_FONT}'>{tiles}</div>"))


def result_card(kicker, headline, detail="", accent=None):
    """Highlighted answer box: small cyan label, large headline, supporting detail."""
    accent = accent or PL["pink"]
    display(HTML(
        f"<div style='font-family:{PL_FONT};background:linear-gradient(100deg,{PL['purple']} 0%,{PL['purple_mid']} 75%,{accent} 150%);"
        f"color:white;border-radius:12px;padding:16px 22px;margin:10px 0;border-left:8px solid {accent}'>"
        f"<div style='font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{PL['cyan']};font-weight:700'>{kicker}</div>"
        f"<div style='font-size:26px;font-weight:800;margin:4px 0 6px'>{headline}</div>"
        f"<div style='font-size:14px;line-height:1.55;opacity:.93'>{detail}</div></div>"
    ))


def pl_table(df, precision=2, highlight_first=True):
    """Styled table: purple header, zebra rows, highlighted leader row.

    Every cell gets an explicit background AND text colour, so the table stays readable
    in both light and dark Jupyter / VS Code themes.
    """
    styler = (
        df.style.format(precision=precision, thousands=",")
        .hide(axis="index")
        .set_table_styles([
            {"selector": "th", "props": f"background-color:{PL['purple']};color:white;font-weight:700;"
                                        f"text-align:left;padding:7px 10px;border:none;font-family:{PL_FONT}"},
            {"selector": "td", "props": f"background-color:white;color:{PL['purple']};padding:5px 10px;"
                                        f"border-bottom:1px solid {PL['grid']};font-family:{PL_FONT}"},
            {"selector": "tbody tr:nth-child(even) td", "props": f"background-color:{PL['light']}"},
        ])
    )
    if highlight_first and len(df):
        first = df.index[0]
        styler = styler.apply(lambda row: [f"font-weight:800;background-color:#E5FFF2;color:{PL['purple']}"
                                           if row.name == first else "" for _ in row], axis=1)
    return styler


def takeaways(*bullets, title="Key takeaways"):
    """Render bullet points as Markdown. Every number in them is computed, so the text never goes stale."""
    lines = [f"**{title}**", ""] + [f"* {b}" for b in bullets if b]
    display(Markdown("\n".join(lines)))


def ordinal(n):
    """1 → '1st', 2 → '2nd', 13 → '13th' (for generated text)."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
