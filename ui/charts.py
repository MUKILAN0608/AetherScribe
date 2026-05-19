import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from sklearn.decomposition import PCA

    HAS_PCA = True
except ImportError:
    HAS_PCA = False

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
    height=300,
    margin=dict(l=12, r=12, t=28, b=40),
    font_size=11,
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", linecolor="#30363d"),
)


def _layout(fig, title: str | None = None, **kwargs):
    layout = {**CHART_LAYOUT, **kwargs}
    if title:
        layout["title"] = dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=14, color="#f0f6fc"),
        )
        layout["margin"] = dict(l=48, r=24, t=52, b=48)
    fig.update_layout(**layout)
    return fig


def chart_with_title(fig, title: str):
    fig = _layout(fig, title=title)
    fig.update_layout(title_x=0.5, title_xanchor="center")
    return fig


def _auto_yrange(values, pad=0.08):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo == hi:
        lo -= pad
        hi += pad
    else:
        span = hi - lo
        lo -= span * pad
        hi += span * pad
    return [lo, hi]


def has_chart_data(values) -> bool:
    if values is None or len(values) == 0:
        return False
    arr = np.asarray(values, dtype=float)
    return bool(np.isfinite(arr).any())


def line_chart(steps, values, color="#58a6ff", y_range=None, fill=False):
    trace = go.Scatter(
        x=steps,
        y=values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=7),
    )
    if fill:
        trace = go.Scatter(
            x=steps,
            y=values,
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=color, width=2),
            marker=dict(size=7),
        )
    fig = go.Figure(data=trace)
    layout_kw = {}
    if y_range is not None:
        layout_kw["yaxis_range"] = y_range
    elif has_chart_data(values):
        layout_kw["yaxis_range"] = _auto_yrange(values)
    return _layout(fig, **layout_kw)


def tension_chart(steps, values, title: str | None = None):
    """Tension / coherence / posterior — typically 0–1."""
    y_range = [0, 1] if has_chart_data(values) and max(values) <= 1.05 and min(values) >= 0 else None
    fig = line_chart(steps, values, color="#58a6ff", y_range=y_range)
    if title:
        return _layout(fig, title=title)
    return fig


def reward_chart(steps, values):
    fig = go.Figure(
        data=go.Bar(
            x=steps,
            y=values,
            marker_color="#3fb950",
            marker_line=dict(color="#2ea043", width=1),
        )
    )
    yr = _auto_yrange(values) if has_chart_data(values) else None
    return _layout(fig, yaxis_range=yr)


def entropy_chart(steps, values):
    return line_chart(steps, values, color="#d29922")


def attribution_pie(labels, values):
    vals = [max(float(v), 0.001) for v in values]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=vals,
                hole=0.45,
                marker=dict(colors=["#58a6ff", "#f85149", "#3fb950"]),
                textinfo="label+percent",
                textfont=dict(color="#e6edf3", size=11),
            )
        ]
    )
    fig = _layout(
        fig,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=52, b=72),
    )
    return fig


def clarity_chart(steps, gaps):
    return line_chart(steps, gaps, color="#bc8cff", fill=True)


def bonds_chart(steps, trust, enmity, trust_name, enmity_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=trust,
            name=f"Trust: {trust_name}",
            mode="lines+markers",
            line=dict(color="#3fb950", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=enmity,
            name=f"Antagonism: {enmity_name}",
            mode="lines+markers",
            line=dict(color="#f85149", width=2),
        )
    )
    yr = _auto_yrange(list(trust) + list(enmity))
    return _layout(
        fig,
        yaxis_range=yr if yr else [0, 1],
        showlegend=True,
        legend=dict(orientation="h", y=-0.22),
    )


def control_heatmap(steps, agent_names, attr_data):
    arr = np.array(attr_data, dtype=float).T
    fig = px.imshow(
        arr,
        x=steps,
        y=agent_names,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_coloraxes(cmin=0, cmax=max(1.0, float(arr.max()) if arr.size else 1.0))
    return _layout(fig, coloraxis_showscale=True)


def narrative_shape(steps, scenes):
    if not HAS_PCA or len(scenes) < 2:
        return None
    rows = []
    for s in scenes:
        rows.append(
            [
                float(s["state"]["tension"]),
                float(s.get("reward", 0)),
                float(s.get("entropy", 0)),
                float(s.get("risk", 0)),
            ]
        )
    arr = np.array(rows, dtype=float)
    if arr.shape[0] < 2:
        return None
    coords = PCA(n_components=2).fit_transform(arr)
    fig = go.Figure(
        data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=steps,
            textposition="top center",
            marker=dict(size=14, color="#58a6ff"),
        )
    )
    return _layout(fig)


def pressure_utility(tension, reward, step_labels=None):
    labels = step_labels or [str(i + 1) for i in range(len(tension))]
    fig = go.Figure(
        data=go.Scatter(
            x=tension,
            y=reward,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=14, color="#58a6ff", symbol="diamond", line=dict(width=1, color="#388bfd")),
        )
    )
    return _layout(
        fig,
        xaxis_title="Tension",
        yaxis_title="Utility",
        xaxis_range=_auto_yrange(tension) if has_chart_data(tension) else None,
        yaxis_range=_auto_yrange(reward) if has_chart_data(reward) else None,
    )
