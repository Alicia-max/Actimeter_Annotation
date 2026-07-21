"""
compute.py
==========
All data loading, algorithm, and metric computation for the sleep pipeline.
"""

from __future__ import annotations
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyActigraphy
from config import Config

def load_actigraphy_file(path: str, logger: logging.Logger) -> Any:
    """Load the actigraphy file and return a pyActigraphy raw object."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"AGD file not found: {src}")
    tmp_dir = Path(tempfile.gettempdir()) / "actigraphy_agd"
    tmp_dir.mkdir(exist_ok=True)
    dst = tmp_dir / src.name
    shutil.copy2(src, dst)
    logger.info(f"Loading: {src.name}")
    raw = pyActigraphy.io.read_raw_agd(str(dst))
    logger.info(f"  {len(raw.data)} epochs  {raw.data.index[0]} → {raw.data.index[-1]}")
    return raw

def validate_recording(raw: Any, config: Config, logger: logging.Logger) -> Dict:
    """Validate the recording by checking for gaps and non-wear periods."""
    data = raw.data
    idx = data.index
    diffs = pd.Series(idx).diff().dropna()
    duration_h = (idx[-1] - idx[0]).total_seconds() / 3600
    gaps = diffs[diffs > 2 * pd.Timedelta(seconds=config.epoch_s)] #gaps are more than 2 epochs
    gap_list = [{"start": str(idx[i]), "duration_min": round(g.total_seconds() / 60, 1)}for i, g in zip(gaps.index, gaps)]
    nonwear_periods = _find_consecutive_zeros(data, int(config.nonwear_min_consecutive_zero_min * 60 / config.epoch_s), config.epoch_s) #define non wear periods based on consecutive gaps
    info = dict(start=str(idx[0]), end=str(idx[-1]), n_epochs=len(data),duration_h=round(duration_h, 2), duration_ok=duration_h >= config.min_recording_hours,n_gaps=len(gap_list),
                gaps=gap_list, n_duplicates=int(idx.duplicated().sum()),nonwear_candidates=nonwear_periods, n_nonwear_candidates=len(nonwear_periods))
   
    #log the information
    
    for k, v in info.items():
        if k not in ("gaps", "nonwear_candidates"):
            logger.info(f"{k}: {v}")
    if gap_list:
        logger.warning(f"{len(gap_list)} gap(s) in recording")
    if nonwear_periods:
        logger.warning(f"{len(nonwear_periods)} non-wear candidate(s) >{config.nonwear_min_consecutive_zero_min} min zeros)")
    return info


def _find_consecutive_zeros(data: pd.Series, min_epochs: int, epoch_s: int) -> List[Dict]:
    """Find consecutive zeros in the data and return a list of dictionaries with the start, end, and duration of the non-wear periods."""
    arr, periods, i = data.values, [], 0
    while i < len(arr):
        if arr[i] == 0:
            j = i
            while j < len(arr) and arr[j] == 0:
                j += 1
            if (j - i) >= min_epochs:
                periods.append({"start": str(data.index[i]),"end": str(data.index[j - 1]),"duration_min": round((j - i) * epoch_s / 60, 1)})
            i = j
        else:
            i += 1
    return periods


def detect_nonwear(data: pd.Series, config: Config) -> pd.Series:
    min_ep = int(config.nonwear_min_consecutive_zero_min * 60 / config.epoch_s)
    nw = pd.Series(False, index=data.index)
    arr, i = data.values, 0
    while i < len(arr):
        if arr[i] == 0:
            j = i
            while j < len(arr) and arr[j] == 0:
                j += 1
            if (j - i) >= min_ep:
                nw.iloc[i:j] = True
            i = j
        else:
            i += 1
    return nw



def run_sleep_wake_algorithm(raw, config: Config, logger: logging.Logger):
    """Run Roenneberg for main sleep window boundaries (1 = sleep/rest)."""
    seed = config.roe_min_seed_period
    roe = None
    try:
        logger.info(f"Roenneberg (threshold={config.roe_threshold}, seed={seed})")
        roe = raw.Roenneberg(threshold=config.roe_threshold, min_seed_period=seed)
    except ValueError as exc:
        if "negative dimensions" in str(exc):
            logger.warning(f"Roenneberg seed={seed} failed ({exc}), retrying with larger seed…")
        else:
            raise
    if roe is None:
        raise RuntimeError("Roenneberg failed for all seed candidates. Recording may be too short or fragmented.")
    return roe


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS WINDOWS & DIARY
# ─────────────────────────────────────────────────────────────────────────────

def create_analysis_windows(raw, config: Config, logger: logging.Logger) -> List[Dict]:
    """Split the recording into 17:00 D to 17:00 D+1 windows."""
    start = raw.data.index[0]
    end = raw.data.index[-1]
    h = config.analysis_day_start_hour
    first_bound = start.normalize() + pd.Timedelta(hours=h)
    if first_bound <= start:
        first_bound += pd.Timedelta(days=1)

    windows, win_start, win_end = [], start, first_bound
    bound_time = pd.Timestamp(2000, 1, 1, h).time()
    while win_start < end:
        actual_end = min(win_end, end)
        label_date = (win_start + pd.Timedelta(hours=6)).date()
        # Incomplete = truncated at recording edges (not a full 17:00→17:00 day)
        incomplete = (win_start.time() != bound_time) or (actual_end < win_end)
        windows.append({
            "window_idx":         len(windows),
            "window_start":       win_start,
            "window_end":         actual_end,
            "analysis_date":      str(label_date),
            "incomplete_window":  incomplete,
        })

        win_start = win_end
        win_end = win_start + pd.Timedelta(days=1)

    logger.info(f"  {len(windows)} analysis windows created.")
    return windows


def load_sleep_diary(path: str, logger: logging.Logger,
                     nights_of_interest: Optional[List[int]] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sleep diary not found: {p}")
    diary = pd.read_csv(p)
    logger.info(f"Diary: {len(diary)} rows from {p.name}")

    if nights_of_interest is not None:
        before = len(diary)
        diary = diary[diary["night_number"].isin(nights_of_interest)].reset_index(drop=True)
        logger.info(f"  Keeping nights {nights_of_interest} — {len(diary)}/{before} rows retained.")

    for col in ("bed_time", "sleep_onset_time", "wake_time", "get_up_time"):
        if col in diary.columns:
            diary[f"{col}_dt"] = diary.apply(
                lambda row, c=col: _parse_diary_datetime(row["sleep_date"], row[c]), axis=1)
    return diary


def _parse_diary_datetime(sleep_date_str: str, time_str: str) -> Optional[pd.Timestamp]:
    """h >= 12  keep sleep_date;  h < 12 sleep_date + 1 day."""
    if pd.isna(time_str) or str(time_str).strip() in ("", "NA", "nan"):
        return None
    try:
        base = pd.Timestamp(sleep_date_str)
        parts = str(time_str).strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        ts = base.replace(hour=h, minute=m, second=0, microsecond=0)
        if h < 12:
            ts += pd.Timedelta(days=1)
        return ts
    except Exception:
        return None


def match_diary_to_window(window: Dict, diary: Optional[pd.DataFrame],
                          analysis_day_start_hour: int = 17) -> Optional[pd.Series]:
    """Match diary sleep_date to the window that starts at analysis_day_start_hour on that date."""
    if diary is None or diary.empty:
        return None
    ws = window["window_start"]
    if ws.hour < analysis_day_start_hour:
        return None
    ws_date = ws.date()
    for _, row in diary.iterrows():
        try:
            sd = pd.Timestamp(row["sleep_date"]).date()
        except Exception:
            continue
        if sd == ws_date:
            return row
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SLEEP EPISODE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_episodes(series: pd.Series, value: int = 1) -> List[Tuple]:
    """Return (start_ts, end_ts) for each contiguous run of *value*."""
    episodes, in_ep, ep_start = [], False, None
    for ts, v in series.items():
        if v == value and not in_ep:
            in_ep, ep_start = True, ts
        elif v != value and in_ep:
            in_ep = False
            episodes.append((ep_start, ts))
    if in_ep and ep_start is not None:
        episodes.append((ep_start, series.index[-1]))
    return episodes


def detect_main_sleep(roe: pd.Series, window: Dict, config: Config):
    ws, we = window["window_start"], window["window_end"]
    win_roe = roe.loc[ws:we].astype(int)
    if win_roe.empty or win_roe.sum() == 0:
        return None, None
    episodes = extract_episodes(win_roe, value=1)
    if not episodes:
        return None, None
    episodes.sort(key=lambda e: e[1] - e[0], reverse=True)
    return episodes[0]


def detect_candidate_naps(roe, window, main_onset, main_offset,
                          nonwear, config: Config) -> List[Dict]:
    ws, we = window["window_start"], window["window_end"]
    ds = int(config.daytime_start.split(":")[0])
    de = int(config.daytime_end.split(":")[0])
    win_roe = roe.loc[ws:we].astype(int)
    episodes = extract_episodes(win_roe, value=1)
    naps = []
    for onset, offset in episodes:
        if main_onset is not None and main_offset is not None:
            overlap = (min(offset, main_offset) - max(onset, main_onset)).total_seconds()
            if overlap > 0:
                continue
        mid = onset + (offset - onset) / 2
        if not (ds <= mid.hour < de):
            continue
        dur_min = (offset - onset).total_seconds() / 60
        if not (config.nap_min_duration_min <= dur_min <= config.nap_max_duration_min):
            continue
        if nonwear.loc[onset:offset].any():
            continue
        naps.append({
            "nap_onset_algorithm":  onset,
            "nap_offset_algorithm": offset,
            "nap_duration_min":     round(dur_min, 1),
        })
    return naps


def _run_spans(values: np.ndarray, target: float) -> List[Tuple[int, int]]:
    """
    Return half-open intervals [start, stop) for contiguous runs equal to target.

    Example:
        [0, 1, 1, 0] with target=1 -> [(1, 3)]
    """
    mask = values == target
    spans: List[Tuple[int, int]] = []

    start = None
    for idx, is_target in enumerate(mask):
        if is_target and start is None:
            start = idx
        elif not is_target and start is not None:
            spans.append((start, idx))
            start = None

    if start is not None:
        spans.append((start, len(values)))

    return spans


def _fill_isolated_mvt_epochs(values: np.ndarray,fill: float = 0.5,) -> np.ndarray:
    """
    Replace isolated movement epochs with `fill`.

    An isolated movement epoch is a value > 0 surrounded by zeros:

        [0, movement, 0] -> [0, fill, 0]

    Example
    -------
    Input:
        [0, 0, 4, 0, 0, 3, 2, 0]

    Output with fill=0.5:
        [0, 0, 0.5, 0, 0, 3, 2, 0]

    The consecutive movement epochs [3, 2] are not modified.
    """
    result = np.asarray(values, dtype=float).copy()

    if result.size < 3:
        return result

    isolated_movement = (
        (result[:-2] == 0)
        & (result[1:-1] > 0)
        & (result[2:] == 0)
    )

    result[1:-1][isolated_movement] = fill

    return result

def _compute_bout_metrics(
    activity: pd.Series,
    config: Config,
    tib: float,TSP_min: float
) -> Dict[str, Any]:
    """
    Score wake/sleep from raw activity within one validated night.

    Pre-process: isolated single-epoch zeros → 0.5 (light movement).
    Then: activity > active_epoch_threshold = wake/mobile; else sleep/immobile.
    """
    out: Dict[str, Any] = {
        "n_SB": np.nan,
        "SB_mean_duration_min": np.nan,
        "SB_total_duration_min": np.nan,
        "n_WB": np.nan,
        "WB_mean_duration_min": np.nan,
        "WB_total_duration_min": np.nan,
        "WASO_min": np.nan,
        "TST_min": np.nan,
        "SE": np.nan,
        "mean_mvt_night": np.nan,
        "mean_mobility": np.nan,
        "taux_immobilite": np.nan,
        "taux_mobilite": np.nan,
    }
    epoch_min = config.epoch_s / 60.0
    activity_threshold = config.active_epoch_threshold
    sb_min = config.sb_min_consecutive_immobile_epochs
    wb_min = config.wb_min_consecutive_movement_epochs

    if activity is not None and not activity.empty:
        activity_numeric = pd.to_numeric(activity, errors="coerce").dropna()

        if not activity_numeric.empty:
            
            # Calculate mean movement and mobility
            out["mean_mvt_night"] = round(float(activity_numeric.mean()),2)
            mobile_epochs = activity_numeric[activity_numeric > activity_threshold]
            out["mean_mobility"] = (round(float(mobile_epochs.mean()), 2) if not mobile_epochs.empty else np.nan)
            n_epochs = len(activity_numeric)
            n_mobile = int((activity_numeric > activity_threshold).sum())
            n_immobile = int((activity_numeric <= activity_threshold).sum())
            out["taux_mobilite"] = round(n_mobile / n_epochs,4)*100
            out["taux_immobilite"] = round(n_immobile / n_epochs,4)*100
            
            #Calculate sleep and wake bouts
            new_activity = _fill_isolated_mvt_epochs(activity_numeric.to_numpy(dtype=float),fill=0.5)
            binary_activity = (new_activity <= activity_threshold).astype(int)
            wake_runs = _run_spans(binary_activity, target=0)
            wb_spans = [(start, stop) for start, stop in wake_runs if (stop - start) >= wb_min] 
            wb_lens = [stop - start for start, stop in wb_spans]
            binary_for_sleep_bouts = np.ones_like(binary_activity)
            for start, stop in wb_spans:
                binary_for_sleep_bouts[start:stop] = 0
            sleep_runs = _run_spans(binary_for_sleep_bouts, target=1)
            sb_spans = [(start, stop) for start, stop in sleep_runs if (stop - start) >= sb_min]
            sb_lens = [stop - start for start, stop in sb_spans]
            n_sb = len(sb_lens)
            n_wb = len(wb_lens)
            sb_total_min = sum(sb_lens) * epoch_min
            wb_total_min = sum(wb_lens) * epoch_min
            out["n_SB"] = n_sb
            out["n_WB"] = n_wb
            out["SB_total_duration_min"] = round(sb_total_min, 1)
            out["WB_total_duration_min"] = round(wb_total_min, 1)
            out["SB_mean_duration_min"] = (round(sb_total_min / n_sb, 1) if n_sb > 0 else np.nan)
            out["WB_mean_duration_min"] = (round(wb_total_min / n_wb, 1) if n_wb > 0 else np.nan)
            out["TST_min"] = round(TSP_min-wb_total_min, 1)
            out["SE"] = round(100.0 * out["TST_min"] / tib, 1) if tib and tib > 0 else np.nan
            out['FI'] = round((n_wb/n_sb), 1)*100


    return out


def calculate_sleep_parameters(window, validated, diary_row, activity,
                               config: Config, logger) -> Dict:
    """Night metrics after validated onset/offset (raw activity wake/sleep)."""
    onset:  Optional[pd.Timestamp] = validated.get("sleep_onset_validated")
    offset: Optional[pd.Timestamp] = validated.get("sleep_offset_validated")

    params: Dict[str, Any] = {
        "participant_id":           validated.get("participant_id", ""),
        "source_file":              validated.get("source_file", ""),
        "analysis_date":            window["analysis_date"],
        "window_start":             str(window["window_start"]),
        "window_end":               str(window["window_end"]),
        "incomplete_window":        window["incomplete_window"],
        "sleep_onset_algorithm":    ts_str(validated.get("sleep_onset_algorithm")),
        "sleep_offset_algorithm":   ts_str(validated.get("sleep_offset_algorithm")),
        "sleep_onset_validated":    ts_str(onset),
        "sleep_offset_validated":   ts_str(offset),
        "manual_onset_correction":  validated.get("manual_onset_correction", False),
        "manual_offset_correction": validated.get("manual_offset_correction", False),
        "bed_time_diary":           None,
        "sleep_onset_diary":        None,
        "get_up_time_diary":        None,
        "wake_time_diary":          None,
        "TinBed_min":               np.nan,
        "SL_min":                   np.nan,
        "TSP_min":                  np.nan,
        "TST_min":                  np.nan,
        "SE":                       np.nan,
        "FI":                 np.nan,
        "n_SB":                     np.nan,
        "SB_mean_duration_min":     np.nan,
        "SB_total_duration_min":    np.nan,
        "n_WB":                     np.nan,
        "WB_mean_duration_min":     np.nan,
        "WB_total_duration_min":    np.nan,
        "mean_mvt_night":           np.nan,
        "mean_mobility":            np.nan,
        "taux_immobilite":          np.nan,
        "taux_mobilite":            np.nan,
        "validation_status":        validated.get("validation_status", ""),
        "discard_reason":           validated.get("discard_reason", ""),
        "quality_flags":            "",
    }

    bed_dt = getup_dt = None
    wake_dt = None
    flags: List[str] = []
    if diary_row is not None:
        bed_dt   = diary_row.get("bed_time_dt")
        getup_dt = diary_row.get("get_up_time_dt")
        wake_dt  = diary_row.get("wake_time_dt")
        sleep_onset = diary_row.get("sleep_onset_time_dt")
        params["bed_time_diary"]    = ts_str(bed_dt)
        params["get_up_time_diary"] = ts_str(getup_dt)
        params["sleep_onset_diary"] = ts_str(sleep_onset)
        params["wake_time_diary"]   = ts_str(wake_dt)
        if bed_dt is not None and getup_dt is not None:
            params["TinBed_min"] = round((getup_dt - bed_dt).total_seconds() / 60, 1)
        if sleep_onset is not None and onset is not None:
            params["SL_min"] = round((sleep_onset - onset).total_seconds() / 60, 1)

    params["TSP_min"] = round((offset - onset).total_seconds() / 60, 1)
    try:
        act_win = activity.loc[onset:offset]
        if len(act_win) > 0:
            params.update(_compute_bout_metrics(act_win, config, params["TinBed_min"], params["TSP_min"]))
    except Exception as e:
        flags.append("bout_metrics_error")
        logger.warning(f"  Bout metrics error: {e}")

    if onset >= offset:
        flags.append("onset_after_offset")
    if wake_dt is not None and onset >= wake_dt:
        flags.append("onset_after_diary_wake")

    params["quality_flags"] = "|".join(flags) if flags else ""
    return params

def ts_str(ts) -> Optional[str]:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    return str(ts)[:19]

def parse_ts_input(text: str, reference: Optional[pd.Timestamp] = None) -> Optional[pd.Timestamp]:
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            pass
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(text, fmt)
            base = reference.normalize() if reference is not None else pd.Timestamp.now().normalize()
            ts = base.replace(hour=t.hour, minute=t.minute, second=t.second)
            if reference is not None and ts < reference:
                ts += pd.Timedelta(days=1)
            return ts
        except ValueError:
            pass
    return None
