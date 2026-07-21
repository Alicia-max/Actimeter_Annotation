"""
plots.py
========
All Plotly figure-building code for the actigraphy sleep pipeline.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from compute import extract_episodes
from dash import html
_C: Dict[str, str] = {
    # Data traces
    "activity":    "#0b0a3a",  # dark navy — readable bars without washing out overlays
    "roe":         "#2bd126",
    "nonwear":     "#D3D3D3",

    # Tier 1 — Diary (light blue, dashdot, lower 40 % of plot)
    "bed":         "#5ab5f2",
    "wake_diary":  "#5ab5f2",
    "getup":       "#5ab5f2",

    # Tier 2 — Algorithm (purple, dashed, middle 40–80 %)
    "algo_onset":  "#7B2FBE",
    "algo_offset": "#7B2FBE",

    # Tier 3 — Validated (red, solid, top 80–100 %)
    "val_onset":   "#E63946",
    "val_offset":  "#E63946",

    # Shading fills (same hue as the tier, used with low opacity)
    "shade_diary": "#5ab5f2",
    "shade_algo":  "#7B2FBE",
    "shade_val":   "#E63946",
}


def _iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _robust_ymax(act: pd.Series, percentile: float = 98.0) -> float:
    """Y-axis ceiling that ignores rare huge peaks so the bulk of the signal stays visible.

    Bars taller than this are clipped by the axis range; hover still shows the true count.
    """
    vals = act.astype(float).dropna()
    if len(vals) == 0:
        return 1.0
    mx = float(vals.max())
    if mx <= 0:
        return 1.0
    p = float(vals.quantile(percentile / 100.0))
    # Prefer the robust ceiling when spikes dominate; otherwise keep the true max.
    if p > 0 and mx > 1.5 * p:
        return max(p * 1.08, 1.0)
    return max(mx * 1.02, 1.0)


def _vlines_and_annotations(diary_row, algo_onset, algo_offset, val_onset, val_offset):
    """Return (shapes, annotations) for the night-validation figure.

    Three visual tiers prevent label overlap even when times are close:
    Tier 1 – DIARY    light-blue dashdot  lower  0–40 %  label y=1.00
    Tier 2 – ALGO     purple dashed       middle 40–80 %  label y=1.08
    Tier 3 – VALIDATED red solid thick    top    80–100 % label y=1.16
    """
    shapes: List[Dict] = []
    annotations: List[Dict] = []
    
    def _shade(x0, x1, color, opacity):
        if x0 is None or x1 is None:
            return
        shapes.append(dict(
            type="rect",
            x0=_iso(x0), x1=_iso(x1),
            y0=0, y1=1, yref="paper",
            fillcolor=color, opacity=opacity,
            line_width=0, layer="below",
        ))

    if diary_row is not None:
        bed_dt  = diary_row.get("bed_time_dt")
        wake_dt = diary_row.get("wake_time_dt") or diary_row.get("get_up_time_dt")
        _shade(bed_dt, wake_dt, _C["shade_diary"], 0.08)
    _shade(algo_onset, algo_offset, _C["shade_algo"], 0.10)
    _shade(val_onset,  val_offset,  _C["shade_val"],  0.18)

    def _line(ts, color, dash, y0, y1, width):
        if ts is None:
            return
        shapes.append(dict(
            type="line",
            x0=_iso(ts), x1=_iso(ts),
            y0=y0, y1=y1, yref="paper",
            line=dict(color=color, width=width, dash=dash),
        ))

    def _label(ts, color, text, y_ann):
        if ts is None:
            return
        annotations.append(dict(x=_iso(ts), y=y_ann, yref="paper",text=f"<b>{text}</b>",showarrow=False,font=dict(size=9, color=color), textangle=-55, xanchor="left"))

    if diary_row is not None:
        bed_dt   = diary_row.get("bed_time_dt")
        wake_dt  = diary_row.get("wake_time_dt")
        getup_dt = diary_row.get("get_up_time_dt")
        _line(bed_dt,   _C["bed"],        "dashdot", 0.00, 0.40, 2.0)
        _line(wake_dt,  _C["wake_diary"], "dashdot", 0.00, 0.40, 2.0)
        _line(getup_dt, _C["getup"],      "dot",     0.00, 0.40, 1.5)
        _label(bed_dt,   _C["bed"],        "Bed",         1.00)
        _label(wake_dt,  _C["wake_diary"], "Wake(diary)",  1.00)
        _label(getup_dt, _C["getup"],      "GetUp",        1.00)
        
    # Values from the roe algorithm
    _line(algo_onset,  _C["algo_onset"],  "dash", 0.40, 0.80, 2.5)
    _line(algo_offset, _C["algo_offset"], "dash", 0.40, 0.80, 2.5)
    _label(algo_onset,  _C["algo_onset"],  "Onset(algo)",  1.08)
    _label(algo_offset, _C["algo_offset"], "Offset(algo)", 1.08)

    # Values to validate the onset and offset
    _line(val_onset,  _C["val_onset"],  "solid", 0.80, 1.00, 3.5)
    _line(val_offset, _C["val_offset"], "solid", 0.80, 1.00, 3.5)
    _label(val_onset,  _C["val_onset"],  "Onset(val)",  1.16)
    _label(val_offset, _C["val_offset"], "Offset(val)", 1.16)

    return shapes, annotations

def build_night_figure(
    activity: pd.Series,
    roe: pd.Series,
    nonwear: pd.Series,
    window: Dict,
    diary_row,
    algo_onset,
    algo_offset,
    val_onset,
    val_offset,
    epoch_s: int,
    title: str = "",
) -> go.Figure:
    """Interactive per-window plot used during manual validation.
    epoch_s : int (from Config).  Used to set the bar width in milliseconds so every epoch fills its slot exactly.
    """
    ws, we = window["window_start"], window["window_end"]
    act = activity.loc[ws:we]
    y_max = _robust_ymax(act, percentile=98.0)
    t = act.index
    bar_width_ms = epoch_s * 1000

    # Non-wear shading
    nw_shapes = []
    nw_win = nonwear.loc[ws:we]
    for ep_s, ep_e in extract_episodes(nw_win.astype(int), value=1):
        nw_shapes.append(dict(
            type="rect",
            x0=ep_s.isoformat(), x1=ep_e.isoformat(),
            y0=0, y1=1, yref="paper",
            fillcolor=_C["nonwear"], opacity=0.35,
            line_width=0, layer="below",
        ))

    vl_shapes, annotations = _vlines_and_annotations(
        diary_row, algo_onset, algo_offset, val_onset, val_offset)

    roe_w = roe.loc[ws:we].astype(float)

    traces = [
        go.Bar(
            x=t, y=act.values, name="Activity (counts)",
            marker_color=_C["activity"], opacity=0.95,
            width=bar_width_ms,
            hovertemplate="%{x|%H:%M:%S}<br>Activity: %{y}<extra></extra>",
        ),
        go.Scatter(
            x=t, y=roe_w.values * y_max,
            name="Roenneberg (1=sleep)",
            mode="lines", line=dict(color=_C["roe"], width=2),
            hovertemplate="%{x|%H:%M:%S}<br>Roe=%{text}<extra></extra>",
            text=roe_w.values.astype(int).astype(str).tolist(),
        ),
    ]

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=title or f"Night — {window['analysis_date']}",
            height=500,
            hovermode="x unified",
            xaxis=dict(
                title="Time",
                type="date",
                tickformat="%H:%M",
                dtick=3_600_000,    # 1-hour ticks in milliseconds
                tickangle=-45,
                range=[ws.isoformat(), we.isoformat()],
            ),
            yaxis=dict(
                title="Activity counts",
                range=[0, y_max],
                fixedrange=False,  # allow zoom to inspect clipped peaks
            ),
            legend=dict(orientation="h", y=-0.28),
            shapes=nw_shapes + vl_shapes,
            annotations=annotations,
            margin=dict(t=110, b=130, l=60, r=40),
        ),
    )


def build_overview_figure(activity: pd.Series, roe: pd.Series,
                          nonwear: pd.Series, windows: List[Dict], epoch_s: int) -> go.Figure:
    """
    Downsampled overview of the full recording.
    Shown once at the start — for orientation only, not for validation.
    """
    # Downsample to ≈ 3 000 points for responsive rendering
    step = max(1, len(activity) // 3000)
    act     = activity.iloc[::step]
    roe_d   = roe.iloc[::step].astype(float)
    nw_d    = nonwear.iloc[::step]
    t       = act.index
    # Cap y-axis at ~98th percentile so rare spikes don't flatten the rest
    y_max = _robust_ymax(act, percentile=98.0)
    bar_width_ms = step * epoch_s * 1000

    # Window-boundary lines (paper yref so they stay full-height under any zoom)
    shapes = [dict(
        type="line",
        x0=w["window_start"].isoformat(), x1=w["window_start"].isoformat(),
        y0=0, y1=1, yref="paper",
        line=dict(color="grey", width=1, dash="dot"),
    ) for w in windows]

    # Non-wear shading
    for ep_s, ep_e in extract_episodes(nw_d.astype(int), value=1):
        shapes.append(dict(
            type="rect",
            x0=ep_s.isoformat(), x1=ep_e.isoformat(),
            y0=0, y1=1, yref="paper",
            fillcolor=_C["nonwear"], opacity=0.3,
            line_width=0, layer="below",
        ))

    return go.Figure(
        data=[
            go.Bar(
                x=t, y=act.values, name="Activity (counts)",
                marker_color=_C["activity"], opacity=0.95,
                width=bar_width_ms,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Activity: %{y}<extra></extra>",
            ),
            go.Scatter(
                x=t, y=roe_d.values * y_max * 0.6,
                name="Roenneberg (sleep=1)", mode="lines",
                line=dict(color=_C["roe"], width=1.5),
            ),
        ],
        layout=go.Layout(
            title="Full recording (overview only)",
            height=400,
            hovermode="x unified",
            xaxis=dict(
                title="Date / Time", type="date",
                tickformat="%d %b\n%H:%M", tickangle=-30,
            ),
            yaxis=dict(
                title="Activity counts",
                range=[0, y_max],
                fixedrange=False,
            ),
            shapes=shapes,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(b=100),
        ),
    )

def build_plot_legend_panel():
    """Return a Dash html.Div describing the three-tier plot legend."""
    def _tier_label(color, symbol, label, extra=""):
        return html.Span([html.Span(symbol, style={"color": color, "fontWeight": "bold","marginRight": "4px", "fontSize": "14px"}),
                          html.Span(label,  style={"marginRight": "14px", "fontSize": "12px"}),
                          html.Span(extra,  style={"color": "#888", "fontSize": "11px","marginRight": "14px"}) if extra else html.Span()])

    return html.Div([ html.Small("Plot legend — three tiers:",style={"color": "#555", "marginRight": "10px"}),
                     _tier_label(_C["bed"], "— —", "Bed / Wake (diary)"),
                     _tier_label(_C["algo_onset"], "- -", "Onset / Offset (algo)"),
                     _tier_label(_C["val_onset"],  "—","Onset / Offset (val)"),
                     _tier_label(_C["nonwear"],"■",   "Non-wear"), ], style={"background": "#f8f9fa", "borderRadius": "6px", "padding": "6px 10px", "marginBottom": "8px","fontSize": "12px", "lineHeight": "2"})
