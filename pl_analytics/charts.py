"""Chart helpers shared by the notebook and the dashboard."""
import numpy as np
import plotly.graph_objects as go

from .theme import PL, PL_FONT, POS_COLORS, pl_title


def draw_xi_pitch(xi, score_col, title, subtitle):
    """Plot the XI on a striped pitch: one row per position, predicted points inside each shirt circle."""
    W, L = 68, 105                                                  # pitch dimensions in metres
    fig = go.Figure()
    for i in range(10):                                             # mown stripes
        fig.add_shape(type="rect", x0=0, x1=W, y0=i * L / 10, y1=(i + 1) * L / 10, line_width=0, layer="below",
                      fillcolor="#2F8F46" if i % 2 else "#38A152")
    line = dict(color="rgba(255,255,255,0.85)", width=2)
    fig.add_shape(type="rect", x0=0, y0=0, x1=W, y1=L, line=line)
    fig.add_shape(type="line", x0=0, y0=L / 2, x1=W, y1=L / 2, line=line)
    fig.add_shape(type="circle", x0=W / 2 - 9.15, y0=L / 2 - 9.15, x1=W / 2 + 9.15, y1=L / 2 + 9.15, line=line)
    for y0, y1 in [(0, 16.5), (L - 16.5, L)]:                        # penalty areas
        fig.add_shape(type="rect", x0=W / 2 - 20.15, x1=W / 2 + 20.15, y0=y0, y1=y1, line=line)
    for y0, y1 in [(0, 5.5), (L - 5.5, L)]:                          # six-yard boxes
        fig.add_shape(type="rect", x0=W / 2 - 9.15, x1=W / 2 + 9.15, y0=y0, y1=y1, line=line)

    for pos, y in {"GK": 10, "DEF": 34, "MID": 60, "FWD": 86}.items():
        grp = xi[xi["position"] == pos].sort_values(score_col, ascending=False)
        xs = np.linspace(0, W, len(grp) + 2)[1:-1]                   # spread players evenly across the width
        fig.add_trace(go.Scatter(
            x=xs, y=[y] * len(grp), mode="markers+text", name=pos,
            marker=dict(size=50, color=POS_COLORS[pos], line=dict(color="white", width=3)),
            text=[f"<b>{p:.0f}</b>" for p in grp[score_col]], textfont=dict(color="white", size=15),
            customdata=grp[["player", "team", "total_points", "now_cost"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Predicted: %{text} pts"
                          "<br>Last season: %{customdata[2]} pts · £%{customdata[3]}m<extra></extra>",
        ))
        labels = grp["web_name"] + np.where(grp.get("captain", False) == True, " (C)", "")
        for x, name, club in zip(xs, labels, grp["team"]):
            fig.add_annotation(x=x, y=y - 7.5, showarrow=False, bgcolor=PL["purple"], borderpad=4,
                               text=f"<b>{name}</b><br><span style='font-size:10px;color:{PL['cyan']}'>{club}</span>",
                               font=dict(color="white", size=12))
    fig.update_xaxes(visible=False, range=[-1, W + 1])
    fig.update_yaxes(visible=False, range=[-1, L + 1], scaleanchor="x", scaleratio=1)
    fig.update_layout(title=pl_title(title, subtitle), height=900, width=680, showlegend=False,
                      margin=dict(l=20, r=20, t=100, b=20))
    return fig
