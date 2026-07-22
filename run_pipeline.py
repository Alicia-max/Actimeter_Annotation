"""
Interactive actigraphy sleep analysis pipeline 
Usage (subject specific data)
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from config import Config
from compute import *
from plots import build_night_figure, build_overview_figure, build_nap_figure, build_plot_legend_panel
warnings.filterwarnings("ignore", category=UserWarning)
import dash
from dash import html, dcc, Input, Output, State, callback_context as ctx, ALL, no_update
# Server-side data store (loaded once before the app starts, never serialised)
_D: Dict[str, Any] = {} # Dictionary to store the data
_BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" #URL of the Bootstrap stylesheet (pre-written CSS rules)
_CARD = {"className": "card p-3 mb-3 shadow-sm"} #CSS class for the card - margin bottom 3, shadow small, padding 3
_BTN  = {"className": "btn me-2"} #CSS class for the button - margin right 2

def _serializable(obj):
    if isinstance(obj, dict): #object to dict
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): #list to list
        return [_serializable(i) for i in obj]
    if isinstance(obj, pd.Timestamp): #timestamp to string
        return str(obj)
    if isinstance(obj, (np.generic,)): #generic to generic
        return obj.item()
    if isinstance(obj, bool): #bool to bool
        return obj
    return obj


def _setup_logging(output_dir: Path, log_filename: str) -> logging.Logger:
    """
    Setup logging to a file and console.
    """
    #setup 
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sleep_pipeline")
    logger.setLevel(logging.DEBUG) # Set logging level for file to DEBUG 
    
    if logger.handlers: # Clear any existing handlers
        logger.handlers.clear()
        
    fh = logging.FileHandler(output_dir / log_filename, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")) 
    
    # Stream handler for console output (only INFO)
    ch = logging.StreamHandler(sys.stdout) 
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s  %(message)s"))
    
    # Add handlers to logger
    logger.addHandler(fh) 
    logger.addHandler(ch)
    return logger


def save_results(nights: List[Dict], naps: List[Dict], output_dir: Path,
                 subject_id: str, logger) -> None:
    """
    Save results as subject-prefixed CSVs:
      {subject}_sleep_parameters.csv  — validated nights only
      {subject}_excluded-night.csv   — discarded / recording-discarded nights
      {subject}_naps.csv             — naps (if any)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    validated = [r for r in nights if r.get("validation_status") == "validated"]
    excluded  = [r for r in nights
                 if r.get("validation_status") in ("discarded", "recording_discarded")]

    validated_path = output_dir / f"{subject_id}_sleep_parameters.csv"
    excluded_path  = output_dir / f"{subject_id}_excluded-night.csv"
    naps_path      = output_dir / f"{subject_id}_naps.csv"

    if validated:
        pd.DataFrame(validated).to_csv(validated_path, index=False)
    elif validated_path.exists():
        validated_path.unlink()

    if excluded:
        pd.DataFrame(excluded).to_csv(excluded_path, index=False)
    elif excluded_path.exists():
        excluded_path.unlink()

    if naps:
        pd.DataFrame(naps).to_csv(naps_path, index=False)
    elif naps_path.exists():
        naps_path.unlink()


def save_state(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serializable(state), f, indent=2, default=str)


def load_state(path: Path) -> Optional[Dict]:
    """
    Load the state from a JSON file.
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ts_display(ts) -> str:
    """
    Display the timestamp in the format YYYY-MM-DD HH:MM:SS.
    """
    return str(ts)[:19] if ts else ""


# ── Layout renderers ─────────────────────────────────────────────────────────

def _render_overview(state: Dict):
    """
    Render the overview page.
    """
    
    D = _D
    fig = build_overview_figure(D["activity"], D["roe"],
                                D["nonwear"], D["windows"], D["config"].epoch_s)
    n_wins = len(D["windows"])
    incomplete = sum(1 for w in D["windows"] if w["incomplete_window"])
    return html.Div([
        html.Div([
            html.H4(f"Recording overview — {D['participant_id']}", className="mb-1"),
            html.P([
                html.B("File: "), D["source_file"], html.Br(),
                html.B("Duration: "), f"{D['rec_info']['duration_h']:.1f} h", html.Br(),
                html.B("Analysis windows: "), f"{n_wins} ({incomplete} incomplete)", html.Br(),
                html.B("Non-wear candidates: "), str(D["rec_info"]["n_nonwear_candidates"]),
            ], className="text-muted small"),
        ], **_CARD),
        html.Div([
            dcc.Graph(figure=fig, config={"scrollZoom": True}),
            html.P("⚠ This plot is for orientation only. Validation is done night by night.",
                   className="text-warning small mt-1"),
        ], **_CARD),
        html.Div([
            html.Button("Start night-by-night validation", id="start-btn",
                        className="btn btn-primary btn-lg"),
        ], className="text-center mb-4"),
    ])


def _render_night_validation(state: Dict):
    """
    Render the night validation page.
    """
    D   = _D
    cfg = D["config"]
    idx = state["current_window_idx"]
    if idx >= len(D["windows"]):
        return _render_done(state)

    window    = D["windows"][idx]
    diary_row = match_diary_to_window(window, D["diary"], D["config"].analysis_day_start_hour)
    algo_onset, algo_offset = detect_main_sleep(D["roe"], window, cfg)

    # Use the stored edit value if present, else fall back to algo proposal
    val_onset_str  = state.get("edit_onset")  or _ts_display(algo_onset)
    val_offset_str = state.get("edit_offset") or _ts_display(algo_offset)
    val_onset  = parse_ts_input(val_onset_str,  window["window_start"])
    val_offset = parse_ts_input(val_offset_str, window["window_start"])

    fig = build_night_figure(
        D["activity"], D["roe"], D["nonwear"],
        window, diary_row, algo_onset, algo_offset, val_onset, val_offset,
        epoch_s=D["config"].epoch_s,
        title=f"Night {idx+1}/{len(D['windows'])} — {window['analysis_date']}" +
              ("INCOMPLETE" if window["incomplete_window"] else ""))

    diary_lines = []
    if diary_row is not None:
        for col, label in [("bed_time_dt","Bed"),("sleep_onset_time_dt","Diary onset"),
                            ("wake_time_dt","Wake"),("get_up_time_dt","Get-up")]:
            v = diary_row.get(col)
            if v is not None:
                diary_lines.append(html.Li(f"{label}: {str(v)[:16]}"))

    msg = state.get("message", "")
    msg_color = "text-danger" if "⚠" in msg else "text-success"
    return html.Div([html.Div([dcc.Graph(id="night-graph",figure=fig,config={"scrollZoom": True}),build_plot_legend_panel()],**_CARD),

        # Diary information

        (html.Div([html.H6("Sleep diary"),html.Ul(diary_lines)if diary_lines else html.P("No diary data for this night.",className="text-muted")],**_CARD)
         if diary_row is not None
         else html.Div()),

        # Edit timestamps

        html.Div([html.H6("Edit sleep onset / offset"),html.P("Format: YYYY-MM-DD HH:MM or HH:MM ""(date inferred from window start)", className="text-muted small"),
                  html.Div([html.Div([html.Label("Sleep onset"),dcc.Input(id="onset-input",value=val_onset_str,placeholder="YYYY-MM-DD HH:MM",type="text",className="form-control")],className="col-md-4"),
                            html.Div([html.Label("Sleep offset"),dcc.Input(id="offset-input",value=val_offset_str,placeholder="YYYY-MM-DD HH:MM",type="text",className="form-control")],className="col-md-4")],className="row mb-3"),
                html.Button("Update plot",id="update-btn",className="btn btn-info me-2"),
                html.Button("Validate night",id="validate-btn",className="btn btn-success me-2"),
                html.Button("Previous night",id="prev-night-btn",className="btn btn-secondary me-2",disabled=(idx == 0),title="Undo the previous night and go back to it")],**_CARD),

        # Discard section
        html.Div([html.H6("Discard"),html.Div([dcc.Dropdown(id="discard-reason",options=[{"label": reason, "value": reason} for reason in cfg.discard_reasons],
                                                            value=cfg.discard_reasons[0],clearable=False,style={"width": "360px", "display": "inline-block","marginRight": "8px"}),
                                            html.Button("Discard night",id="discard-night-btn",className="btn btn-warning me-2"),
                                            html.Button("Discard recording",id="discard-rec-btn",className="btn btn-danger")],className="d-flex align-items-center")],**_CARD),
        (html.P(msg, className=f"{msg_color} fw-bold") if msg else html.Div()),
        html.P(f"Window {idx + 1} of {len(D['windows'])}",className="text-muted text-end small")])


def _render_nap_validation(state: Dict):
    D   = _D
    idx = state["current_window_idx"]
    window = D["windows"][idx]
    naps = state.get("current_candidate_naps", [])
    validated = state.get("current_validated") or {}
    main_onset  = parse_ts_input(str(validated.get("sleep_onset_validated") or ""),
                                 window["window_start"])
    main_offset = parse_ts_input(str(validated.get("sleep_offset_validated") or ""),
                                 window["window_start"])

    fig = build_nap_figure(
        D["activity"], D["roe"], D["nonwear"],
        window, naps,
        main_onset=main_onset, main_offset=main_offset,
        epoch_s=D["config"].epoch_s,
        title=f"Nap validation — window {idx+1}/{len(D['windows'])} — {window['analysis_date']}",
    )

    nap_cards = []
    for i, nap in enumerate(naps):
        nap_cards.append(html.Div([
            html.H6(f"Candidate nap {i+1}: "
                    f"{str(nap['nap_onset_algorithm'])[:16]} → "
                    f"{str(nap['nap_offset_algorithm'])[:16]} "
                    f"({nap['nap_duration_min']:.0f} min)"),
            html.Div([
                html.Div([
                    html.Label("Onset"),
                    dcc.Input(id={"type":"nap-onset","index":i},
                              value=str(nap["nap_onset_algorithm"])[:19],
                              type="text", className="form-control"),
                ], className="col-md-3"),
                html.Div([
                    html.Label("Offset"),
                    dcc.Input(id={"type":"nap-offset","index":i},
                              value=str(nap["nap_offset_algorithm"])[:19],
                              type="text", className="form-control"),
                ], className="col-md-3"),
                html.Div([
                    html.Label("Decision"),
                    dcc.RadioItems(
                        id={"type":"nap-decision","index":i},
                        options=[{"label":"Accept","value":"accept"},
                                 {"label":"Reject","value":"reject"}],
                        value="accept",
                        inline=True,
                        className="mt-2",
                    ),
                ], className="col-md-3"),
                html.Div([
                    html.Label("Reject reason (if rejected)"),
                    dcc.Input(id={"type":"nap-reason","index":i},
                              placeholder="inactivity / non-wear / other",
                              type="text", className="form-control"),
                ], className="col-md-3"),
            ], className="row"),
        ], className="card p-3 mb-2"))

    if not nap_cards:
        nap_cards = [html.P("No candidate naps found for this window.", className="text-muted")]

    return html.Div([
        html.Div([
            html.H4(f"Nap validation — window {idx+1}, {window['analysis_date']}"),
            html.P(f"{len(naps)} candidate nap(s) found "
                   f"(>{D['config'].nap_min_duration_min} min, "
                   f"{D['config'].daytime_start}–{D['config'].daytime_end})",
                   className="text-muted"),
            dcc.Graph(id="nap-graph", figure=fig, config={"scrollZoom": True}),
            html.Small([
                html.Span("■ ", style={"color": "#E76F51", "fontWeight": "bold"}),
                "Highlighted bars = detected nap activity  ·  ",
                html.Span("■ ", style={"color": "#F4A261", "fontWeight": "bold"}),
                "Orange shade = nap window  ·  ",
                html.Span("■ ", style={"color": "#E63946", "fontWeight": "bold"}),
                "Red shade = main night",
            ], className="text-muted"),
        ], **_CARD),
        *nap_cards,
        html.Div([
            html.Button("  Done with naps",     id="done-naps-btn",    className="btn btn-primary me-2"),
            html.Button(" Back to onset/offset", id="back-to-night-btn", className="btn btn-secondary"),
            html.P(state.get("message",""), className="text-danger mt-2"),
        ], **_CARD),
    ])


def _render_parameter_review(state: Dict):
    from dash import html, dcc
    D   = _D
    params = state.get("current_params", {})
    flags  = params.get("quality_flags","")

    display_cols = [
        ("participant_id", "Participant ID"),
        ("analysis_date", "Analysis date"),
        ("bed_time_diary", "Bed time (diary)"),
        ("get_up_time_diary", "Get-up time (diary)"),
        ("sleep_onset_validated", "Sleep onset (validated)"),
        ("sleep_offset_validated", "Sleep offset (validated)"),
        ("TinBed_min", "Time in bed (min)"),
        ("TSP_min", "Total sleep period (min)"),
        ("TST_min", "Total sleep time (min)"),
        ("SE", "Sleep efficiency (%)"),
        ("FI", "Fragmentation index (%)"),
        ("SL_min", "Sleep latency (min)"),
        ("n_SB", "Sleep bouts (n)"),
        ("SB_mean_duration_min", "Sleep bout mean duration (min)"),
        ("SB_total_duration_min", "Sleep bout total duration (min)"),
        ("n_WB", "Wake bouts (n)"),
        ("WB_mean_duration_min", "Wake bout mean duration (min)"),
        ("WB_total_duration_min", "Wake bout total duration (min)"),
        ("mean_mvt_night", "Mean movement night"),
        ("mean_mobility", "Mean mobility (>0)"),
        ("taux_immobilite", "Taux immobilité (% repos)"),
        ("taux_mobilite", "Taux mobilité (% mouvement)"),
        ("quality_flags", "Quality flags"),
        ("manual_onset_correction", "Manual onset correction"),
        ("manual_offset_correction", "Manual offset correction"),
    ]

    rows = []
    for key, label in display_cols:
        val = params.get(key, "")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val_str = "—"
            style   = {"color":"grey"}
        else:
            val_str = str(val)
            style   = {"color":"red","fontWeight":"bold"} if key=="quality_flags" and val_str else {}
        rows.append(html.Tr([
            html.Td(label, style={"fontWeight":"bold","padding":"3px 10px","whiteSpace":"nowrap"}),
            html.Td(val_str, style={**style,"padding":"3px 10px"}),
        ]))

    warn_block = html.Div(
        f"Quality flags: {flags}",
        className="alert alert-warning",
    ) if flags else html.Div()

    return html.Div([
        html.Div([
            html.H4("Parameter review"),
            warn_block,
            html.Table(rows, className="table table-sm table-hover"),
        ], **_CARD),
        html.Div([
            html.Button("Save & continue",           id="confirm-btn",        className="btn btn-success me-2"),
            html.Button("Go back to onset/offset",   id="go-back-btn",        className="btn btn-info me-2"),
            html.Button("Discard this night",         id="review-discard-btn", className="btn btn-warning"),
        ], **_CARD),
        html.P(state.get("message",""), className="text-danger fw-bold"),
    ])


def _render_done(state: Dict):
    D = _D
    nights = state.get("nights", [])
    naps   = state.get("naps",   [])
    n_val  = sum(1 for r in nights if r.get("validation_status") == "validated")
    n_disc = sum(1 for r in nights if r.get("validation_status") == "discarded")
    n_naps = sum(1 for r in naps   if r.get("validation_status") == "accepted")

    return html.Div([
        html.Div([
            html.H3("Pipeline complete", className="text-success"),
            html.Ul([
                html.Li(f"Validated nights: {n_val}"),
                html.Li(f"Discarded nights: {n_disc}"),
                html.Li(f"Accepted naps:    {n_naps}"),
            ]),
            html.P(["Results saved in: ", html.Code(str(D["output_dir"]))]),
            html.Ul([
                html.Li(html.Code(f"{D['subject']}_sleep_parameters.csv — validated nights")),
                html.Li(html.Code(f"{D['subject']}_excluded-night.csv — excluded nights")),
                html.Li(html.Code(f"{D['subject']}_naps.csv — naps")),
                html.Li(html.Code(f"{D['config'].log_filename} — processing log")),
            ]),
        ], className="card p-4 shadow-sm border-success"),
    ])


# ── Callback registration ────────────────────────────────────────────────────

def _register_callbacks(app):
    @app.callback(Output("content", "children"), Input("state", "data"))
    def render_content(state):
        step = state.get("step", "overview")
        if step == "overview":            return _render_overview(state)
        if step == "night_validation":    return _render_night_validation(state)
        if step == "nap_validation":      return _render_nap_validation(state)
        if step == "parameter_review":    return _render_parameter_review(state)
        return _render_done(state)

    # Overview → start
    @app.callback(
        Output("state", "data", allow_duplicate=True),
        Input("start-btn", "n_clicks"),
        State("state", "data"),
        prevent_initial_call=True,
    )
    def on_start(n, state):
        # Ignore ghost fires when the button is removed from the layout
        if not n or state.get("step") != "overview":
            return no_update
        state["step"] = "night_validation"
        state["edit_onset"] = state["edit_offset"] = None
        state["message"] = ""
        return state

    # Night validation → update graph only
    @app.callback(
        Output("night-graph", "figure"),
        Input("update-btn", "n_clicks"),
        State("onset-input",  "value"),
        State("offset-input", "value"),
        State("state", "data"),
        prevent_initial_call=True,
    )
    def update_graph(n, onset_val, offset_val, state):
        if not n or state.get("step") != "night_validation":
            return no_update
        D   = _D
        idx = state["current_window_idx"]
        window    = D["windows"][idx]
        diary_row = match_diary_to_window(window, D["diary"], D["config"].analysis_day_start_hour)
        algo_onset, algo_offset = detect_main_sleep(D["roe"], window, D["config"])
        val_onset  = parse_ts_input(onset_val,  window["window_start"])
        val_offset = parse_ts_input(offset_val, window["window_start"])
        return build_night_figure(
            D["activity"], D["roe"], D["nonwear"],
            window, diary_row, algo_onset, algo_offset, val_onset, val_offset,
            epoch_s=D["config"].epoch_s,
        )

    # Night validation → validate / discard night / discard recording / go back
    @app.callback(
        Output("state", "data", allow_duplicate=True),
        Input("validate-btn",      "n_clicks"),
        Input("discard-night-btn", "n_clicks"),
        Input("discard-rec-btn",   "n_clicks"),
        Input("prev-night-btn",    "n_clicks"),
        State("onset-input",    "value"),
        State("offset-input",   "value"),
        State("discard-reason", "value"),
        State("state", "data"),
        prevent_initial_call=True,
    )
    def handle_night_actions(v_n, dn_n, dr_n, prev_n, onset_val, offset_val, reason, state):
        D   = _D
        tid = ctx.triggered_id
        # Ghost fire when buttons are torn down on step change (n_clicks becomes None)
        click_map = {
            "validate-btn": v_n,
            "discard-night-btn": dn_n,
            "discard-rec-btn": dr_n,
            "prev-night-btn": prev_n,
        }
        if tid not in click_map or not click_map[tid]:
            return no_update
        if state.get("step") != "night_validation":
            return no_update

        idx = state["current_window_idx"]
        window = D["windows"][idx]

        if tid == "validate-btn":
            onset  = parse_ts_input(onset_val,  window["window_start"])
            offset = parse_ts_input(offset_val, window["window_start"])
            if onset is None or offset is None:
                state["message"] = " Both onset and offset must be set."
                return state
            if onset >= offset:
                state["message"] = " Onset must be before offset."
                return state

            diary_row = match_diary_to_window(window, D["diary"], D["config"].analysis_day_start_hour)
            algo_onset, algo_offset = detect_main_sleep(D["roe"], window, D["config"])

            validated = {
                "participant_id":          D["participant_id"],
                "source_file":             D["source_file"],
                "sleep_onset_algorithm":   ts_str(algo_onset),
                "sleep_offset_algorithm":  ts_str(algo_offset),
                "sleep_onset_validated":   onset,
                "sleep_offset_validated":  offset,
                "manual_onset_correction": onset != algo_onset if algo_onset else True,
                "manual_offset_correction":offset != algo_offset if algo_offset else True,
                "validation_status":       "validated",
                "discard_reason":          "",
            }

            params = calculate_sleep_parameters(
                window, validated, diary_row,
                D["activity"],
                D["config"], D["logger"],
            )

            candidate_naps = detect_candidate_naps(
                D["roe"], window, onset, offset, D["nonwear"], D["config"])

            state["current_validated"]      = _serializable(validated)
            state["current_params"]         = _serializable(params)
            state["current_candidate_naps"] = _serializable(candidate_naps)
            state["edit_onset"]  = None
            state["edit_offset"] = None
            state["message"]     = ""
            state["step"]        = "nap_validation"
            return state

        if tid == "discard-night-btn":
            D["logger"].warning(f"Night {idx+1} discarded: {reason}")
            row = {
                "participant_id": D["participant_id"],
                "source_file":    D["source_file"],
                "analysis_date":  window["analysis_date"],
                "window_start":   str(window["window_start"]),
                "window_end":     str(window["window_end"]),
                "incomplete_window": window["incomplete_window"],
                "validation_status": "discarded",
                "discard_reason":    reason,
                "validation_timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            state["nights"].append(row)
            _persist(state, D)
            state["current_window_idx"] += 1
            state["edit_onset"] = state["edit_offset"] = None
            state["message"]    = ""
            if state["current_window_idx"] >= len(D["windows"]):
                state["step"] = "done"
            else:
                state["step"] = "night_validation"
            return state

        if tid == "discard-rec-btn":
            D["logger"].warning(f"Recording discarded: {reason}")
            state["nights"].append({
                "participant_id": D["participant_id"],
                "source_file":    D["source_file"],
                "validation_status": "recording_discarded",
                "discard_reason":    reason,
                "validation_timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            _persist(state, D)
            state["step"] = "done"
            return state

        if tid == "prev-night-btn":
            if idx <= 0:
                state["message"] = " Already at the first window, cannot go back."
                return state
            # Remove the last saved night (undo it) and step back
            if state["nights"]:
                state["nights"].pop()
            state["current_window_idx"] -= 1
            state["edit_onset"]  = None
            state["edit_offset"] = None
            state["message"]     = "Returned to previous night"
            state["step"]        = "night_validation"
            _persist(state, D)
            return state

        return no_update

    # Nap validation → done / back to night validation
    @app.callback(
        Output("state", "data", allow_duplicate=True),
        Input("done-naps-btn",     "n_clicks"),
        Input("back-to-night-btn", "n_clicks"),
        State({"type": "nap-onset",    "index": ALL}, "value"),
        State({"type": "nap-offset",   "index": ALL}, "value"),
        State({"type": "nap-decision", "index": ALL}, "value"),
        State({"type": "nap-reason",   "index": ALL}, "value"),
        State("state", "data"),
        prevent_initial_call=True,
    )
    def done_naps(n, back_n, onsets, offsets, decisions, reasons, state):
        D   = _D
        tid = ctx.triggered_id
        click_map = {"done-naps-btn": n, "back-to-night-btn": back_n}
        if tid not in click_map or not click_map[tid]:
            return no_update
        if state.get("step") != "nap_validation":
            return no_update

        if tid == "back-to-night-btn":
            # Return to night validation for the same window (onset/offset preserved)
            state["step"]    = "night_validation"
            state["message"] = "Back to onset/offset."
            return state

        idx = state["current_window_idx"]
        candidate_naps = state.get("current_candidate_naps", [])

        validated_naps = []
        for i, (nap, onset_s, offset_s, decision, reason) in enumerate(
                zip(candidate_naps, onsets, offsets, decisions, reasons)):
            window    = D["windows"][idx]
            accepted  = decision == "accept"
            nap_onset  = parse_ts_input(onset_s,  window["window_start"]) if accepted else None
            nap_offset = parse_ts_input(offset_s, window["window_start"]) if accepted else None
            dur = ((nap_offset - nap_onset).total_seconds()/60
                   if accepted and nap_onset and nap_offset
                   else nap["nap_duration_min"])
            validated_naps.append({
                "participant_id":       D["participant_id"],
                "source_file":          D["source_file"],
                "analysis_date":        D["windows"][idx]["analysis_date"],
                "nap_number":           i + 1,
                "nap_onset_algorithm":  str(nap["nap_onset_algorithm"])[:19],
                "nap_offset_algorithm": str(nap["nap_offset_algorithm"])[:19],
                "nap_onset_validated":  ts_str(nap_onset),
                "nap_offset_validated": ts_str(nap_offset),
                "nap_duration_min":     round(dur, 1),
                "manual_correction": (
                    onset_s != str(nap["nap_onset_algorithm"])[:19]
                    or offset_s != str(nap["nap_offset_algorithm"])[:19]
                ),
                "validation_status": "accepted" if accepted else "rejected",
                "rejection_reason":  reason or "",
            })

        accepted_count = sum(1 for nap in validated_naps if nap["validation_status"] == "accepted")
        params = state.get("current_params", {})
        params["number_of_naps"] = accepted_count
        state["current_params"]  = params
        state["naps"].extend(validated_naps)
        state["step"] = "parameter_review"
        state["message"] = ""
        return state

    # Parameter review → confirm / go back / discard
    @app.callback(
        Output("state", "data", allow_duplicate=True),
        Input("confirm-btn",        "n_clicks"),
        Input("go-back-btn",        "n_clicks"),
        Input("review-discard-btn", "n_clicks"),
        State("state", "data"),
        prevent_initial_call=True,
    )
    def handle_review(c_n, gb_n, d_n, state):
        D   = _D
        tid = ctx.triggered_id
        # Ghost fire: after Save & continue, confirm-btn is removed from the DOM and
        # Dash re-triggers this callback — without a guard that advances a second time.
        click_map = {
            "confirm-btn": c_n,
            "go-back-btn": gb_n,
            "review-discard-btn": d_n,
        }
        if tid not in click_map or not click_map[tid]:
            return no_update
        if state.get("step") != "parameter_review":
            return no_update

        idx = state["current_window_idx"]

        if tid == "confirm-btn":
            params = state.get("current_params", {})
            state["nights"].append(params)
            D["logger"].info(f"Night {idx+1} validated and saved.")
            _persist(state, D)
            state["current_window_idx"] += 1
            state["edit_onset"] = state["edit_offset"] = None
            state["message"]    = ""
            state["step"] = "done" if state["current_window_idx"] >= len(D["windows"]) \
                            else "night_validation"
            return state

        if tid == "go-back-btn":
            state["step"]    = "night_validation"
            state["message"] = ""
            return state

        if tid == "review-discard-btn":
            window = D["windows"][idx]
            row = {
                "participant_id":    D["participant_id"],
                "source_file":       D["source_file"],
                "analysis_date":     window["analysis_date"],
                "window_start":      str(window["window_start"]),
                "window_end":        str(window["window_end"]),
                "incomplete_window": window["incomplete_window"],
                "validation_status": "discarded",
                "discard_reason":    "discarded from parameter review",
                "validation_timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            state["nights"].append(row)
            _persist(state, D)
            state["current_window_idx"] += 1
            state["edit_onset"] = state["edit_offset"] = None
            state["message"]    = ""
            state["step"] = "done" if state["current_window_idx"] >= len(D["windows"]) \
                            else "night_validation"
            return state

        return no_update


def _persist(state: Dict, D: Dict) -> None:
    """Save current results + state to disk after each validated/discarded night."""
    save_results(state["nights"], state["naps"], D["output_dir"], D["subject"], D["logger"])
    save_state(state, D["output_dir"] / f'{D["subject"]}_{D["config"].state_filename}')


# ── Main entry point ─────────────────────────────────────────────────────────

def _initial_state(resume_path: Optional[Path] = None) -> Dict:
    base = {
        "step":                    "overview",
        "current_window_idx":      0,
        "nights":                  [],
        "naps":                    [],
        "current_validated":       None,
        "current_candidate_naps":  [],
        "current_params":          None,
        "edit_onset":              None,
        "edit_offset":             None,
        "message":                 "",
    }
    if resume_path is not None:
        saved = load_state(resume_path)
        if saved:
            base.update(saved)
            # Resume resumes from night_validation if mid-way, not overview
            if base.get("current_window_idx", 0) > 0:
                base["step"] = "night_validation"
                base["edit_onset"] = base["edit_offset"] = None
                base["message"] = ""
    return base


def run_pipeline(subject: str, agd_path: str, diary_path: Optional[str], output_dir: str,
                 resume: bool, config: Config, port: int = 8050) -> None:
    """Load all data, build the Dash app, and start the server."""

    global _D
    out = Path(output_dir)
    log = _setup_logging(out, config.log_filename)
    log.info("SLEEP PIPELINE — START")
    log.info(f"  AGD   : {agd_path}")
    log.info(f"  Diary : {diary_path or 'None'}")
    log.info(f"  Output: {out}")
    log.info(f"  Subject: {subject}")

    # Load
    raw  = load_actigraphy_file(agd_path, log)
    info = validate_recording(raw, config, log)
    roe = run_sleep_wake_algorithm(raw, config, log)
    nonwear = detect_nonwear(raw.data, config)
    windows = create_analysis_windows(raw, config, log)
    diary = None
    if diary_path:
        try:
            diary = load_sleep_diary(diary_path, log,nights_of_interest=config.diary_nights_of_interest)
        except FileNotFoundError as e:
            log.warning(str(e))
    fname  = Path(agd_path).stem
    pid    = fname.split(" ")[0] if " " in fname else fname

    _D = {
        "raw":               raw,
        "activity":          raw.data,
        "roe":               roe,
        "nonwear":           nonwear,
        "windows":           windows,
        "diary":             diary,
        "rec_info":          info,
        "subject":           subject,
        "participant_id":    pid,
        "source_file":       Path(agd_path).name,
        "output_dir":        out,
        "config":            config,
        "logger":            log,
    }

    state_path   = out / f"{subject}_{config.state_filename}"
    initial_data = _initial_state(state_path if resume else None)
    if resume and initial_data.get("current_window_idx", 0) > 0:
        log.info(f"Resuming from window {initial_data['current_window_idx']+1}")

    # Build app
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[_BS],
    )
    app.title = f"Sleep Pipeline — {pid}"
    app.layout = html.Div([
        # Header
        html.Div([html.H3(f"Sleep Pipeline — {pid}",style={"color":"white","margin":0})], style={"background":"#343a40","padding":"12px 24px","marginBottom":"20px"}),
        # Main content driven by state store
        dcc.Store(id="state", data=initial_data),
        html.Div(id="content", className="container-fluid px-4")])

    _register_callbacks(app)

    print(f"\n{'='*60}")
    print(f"  Open your browser at:  http://localhost:{port}")
    print(f"  Press Ctrl+C to quit.")
    print(f"{'='*60}\n")
    app.run(debug=False, port=port, use_reloader=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive actigraphy sleep analysis pipeline (Dash web app)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agd",    required=True,  help="Path to *30sec.agd file")
    parser.add_argument("--subject", required=True, help="Subject ID")
    parser.add_argument("--diary",  default=None,   help="Path to sleep diary CSV (optional)")
    parser.add_argument("--output", default=None,   help="Output directory (overrides config)")
    parser.add_argument("--config", default=None,   help="Path to config.yaml (optional)")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--port",   type=int, default=8050, help="Port for Dash server")
    # Individual overrides — any supplied flag wins over the YAML value
    parser.add_argument("--roe-threshold",   type=float, default=None)
    parser.add_argument("--epoch-s",         type=int,   default=None)
    parser.add_argument("--nap-min",         type=int,   default=None, dest="nap_min")
    parser.add_argument("--nap-max",         type=int,   default=None, dest="nap_max")
    parser.add_argument("--nonwear-min-min", type=float, default=None, dest="nonwear_min")
    parser.add_argument("--diary-nights",    default=None, dest="diary_nights",
                        help="Comma-separated night numbers, e.g. 2,3  (overrides config)")
    args = parser.parse_args()

    # Resolve config file: --config flag  >  config.yaml next to the script
    _script_dir   = Path(__file__).parent
    _default_yaml = _script_dir / "config.yaml"

    config_path = args.config or (_default_yaml if _default_yaml.exists() else None)

    if config_path:
        cfg = Config.from_yaml(config_path)
        print(f"Configuration loaded from: {config_path}")
    else:
        raise SystemExit(
            "\n[error] No configuration file found.\n"
            f"Create config.yaml next to {Path(__file__).name}, or pass --config <path>.\n"
            "A template config.yaml is included in the repository."
        )

    # 2. Apply any explicit CLI overrides (None = not supplied → keep YAML value)
    if args.epoch_s       is not None: cfg.epoch_s                          = args.epoch_s
    if args.roe_threshold is not None: cfg.roe_threshold                    = args.roe_threshold
    if args.nap_min       is not None: cfg.nap_min_duration_min             = args.nap_min
    if args.nap_max       is not None: cfg.nap_max_duration_min             = args.nap_max
    if args.nonwear_min   is not None: cfg.nonwear_min_consecutive_zero_min = args.nonwear_min
    if args.output        is not None: cfg.output_dir                       = args.output
    if args.diary_nights  is not None:
        cfg.diary_nights_of_interest = [int(x.strip()) for x in args.diary_nights.split(",")]

    run_pipeline(
        subject    = args.subject,
        agd_path   = args.agd,
        diary_path = args.diary,
        output_dir = cfg.output_dir,
        resume     = args.resume,
        config     = cfg,
        port       = args.port,
    )


if __name__ == "__main__":
    main()
