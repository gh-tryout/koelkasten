"""Streamlit-GUI voor EN 62552-3 energie: meerdere metingen, interpolatie, jaarverbruik."""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys
import traceback
from dataclasses import fields, replace
from importlib import reload
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

import en62552
import main as analysis

# Streamlit herlaadt app.py bij elke interactie. main/en62552 niet opnieuw
# importeren: reload() voelt als een crash/herstart bij file-upload.


def _project_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / "power_meetdata_map").is_dir() or (exe_dir / "temp_meetdata_map").is_dir():
            return exe_dir
        return Path(getattr(sys, "_MEIPASS", exe_dir))
    return Path(__file__).resolve().parent


PROJECT_DIR = _project_dir()
os.chdir(PROJECT_DIR)

DEFAULT_COMPARTMENTS = [
    {
        "name": "fridge",
        "table_type": "Fresh food",
        "volume": 233.0,
        "sensors": "105 (C), 202 (C), 209 (C)",
    },
    {
        "name": "freezer 3 star",
        "table_type": "Three-star",
        "volume": 107.0,
        "sensors": "102 (C), 103 (C), 201 (C), 203 (C), 107 (C)",
    },
]


VS_TARGET_OPTIONS = {
    "onder": "Kouder dan doeltemperatuur",
    "boven": "Warmer dan doeltemperatuur",
}
_KIND_NL = {
    "compressor_run_time": "compressor-run-time (Annex D.3)",
    "elapsed_time": "elapsed-time (Annex D.2)",
    "variable": "variable/demand-defrost",
    "onvoldoende_data": "onvoldoende data",
}
_TESTPOINT_FIELDS = {f.name for f in fields(en62552.TestPoint)}
_SETTINGS_VERSION = 2


def _blank_test(
    name: str,
    ambient_c: int,
    vs_target: str,
    test_id: str | None = None,
) -> dict:
    return {
        "id": test_id or f"run-{ambient_c}-{vs_target}",
        "name": name,
        "ambient_c": ambient_c,
        "vs_target": vs_target,
        "power_glob": "",
        "temp_file": "",
        "temp_folder": "",
        "point": None,
        "tcrt_hours": [],
        "elapsed_hours": [],
        "n_defrost_events": 0,
        "defrost_events": [],
        "power_gaps": [],
        "vs_target_warning": None,
        "ss2": None,
    }


def _default_tests() -> list[dict]:
    return [
        _blank_test("16 °C — kouder dan target", 16, "onder"),
        _blank_test("16 °C — warmer dan target", 16, "boven"),
        _blank_test("32 °C — kouder dan target", 32, "onder"),
        _blank_test("32 °C — warmer dan target", 32, "boven"),
    ]


def _as_comp_df(value) -> pd.DataFrame:
    df = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    for col, default in (
        ("name", ""),
        ("table_type", "Fresh food"),
        ("volume", 0.0),
        ("sensors", ""),
    ):
        if col not in df.columns:
            df[col] = default
    return df[["name", "table_type", "volume", "sensors"]].copy()


def _compartment_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for i, row in df.iterrows():
        name = "" if pd.isna(row.get("name")) else str(row.get("name")).strip()
        table_type = row.get("table_type")
        if pd.isna(table_type) or not table_type:
            table_type = "Fresh food"
        table_type = str(table_type)
        volume = row.get("volume")
        volume = 0.0 if pd.isna(volume) else float(volume)
        sensors = "" if pd.isna(row.get("sensors")) else str(row.get("sensors")).strip()
        if not name and not sensors:
            continue
        records.append(
            {
                "name": name or f"compartiment {int(i) + 1}",
                "table_type": table_type
                if table_type in en62552.COMPARTMENT_TARGETS_C
                else "Fresh food",
                "volume": volume,
                "sensors": sensors,
            }
        )
    return records


def _migrate_tests(tests: list[dict]) -> None:
    for t in tests:
        t.setdefault("vs_target", "onder")
        t.setdefault("tcrt_hours", [])
        t.setdefault("elapsed_hours", [])
        t.setdefault("n_defrost_events", 0)
        t.setdefault("defrost_events", [])
        t.setdefault("power_gaps", [])
        t.setdefault("temp_folder", "")
        t.setdefault("vs_target_warning", None)
        t.setdefault("ss2", None)
        t.pop("setpoint", None)


def _init_state() -> None:
    pending = st.session_state.pop("pending_settings", None)
    if pending is not None:
        try:
            _apply_campaign_settings(pending)
        except Exception as exc:
            st.session_state["flash"] = None
            st.session_state["settings_error"] = str(exc)
    if "compartments_df" not in st.session_state:
        st.session_state["compartments_df"] = _as_comp_df(
            st.session_state.get("compartments_ui", DEFAULT_COMPARTMENTS)
        )
    st.session_state["compartments_ui"] = _compartment_records(
        _as_comp_df(st.session_state["compartments_df"])
    )
    st.session_state.setdefault(
        "ambient_sensors", "205 (C), 210 (C), 204 (C)"
    )
    st.session_state.setdefault("gradient_sensors", "110 (C), 206 (C)")
    st.session_state.setdefault("heater_advances_timer", False)
    st.session_state.setdefault("defrost_power_w", 50.0)
    st.session_state.setdefault("tests", _default_tests())
    _migrate_tests(st.session_state["tests"])
    st.session_state.setdefault("active_test", 0)
    st.session_state.setdefault("flash", None)
    st.session_state.setdefault("log", "")
    for key in (
        "lf_meas_data",
        "load_info",
        "df_TCC",
        "df_TCC_w_streaks_and_means",
        "df_meas_data_w_smooth_w_TCC_id",
        "plot_data",
        "ss1_opti",
        "pss",
        "df_all_possible_periods_w_stats",
        "mean_amb_temp",
        "Delta_E_dfj_wh",
        "delta_t_df",
        "ss1_candidates",
        "ss2",
        "df1_details",
        "ss1_diagnostics",
        "ss1_diag_summary",
        "view_df",
        "view_key",
    ):
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("power_folder", "")
    st.session_state.setdefault("power_pool", [])
    _ensure_campaign_power()


def _clear_analysis() -> None:
    for key in (
        "lf_meas_data",
        "load_info",
        "df_TCC",
        "df_TCC_w_streaks_and_means",
        "df_meas_data_w_smooth_w_TCC_id",
        "plot_data",
        "ss1_opti",
        "pss",
        "df_all_possible_periods_w_stats",
        "mean_amb_temp",
        "Delta_E_dfj_wh",
        "delta_t_df",
        "ss1_candidates",
        "ss2",
        "df1_details",
        "ss1_diagnostics",
        "ss1_diag_summary",
    ):
        st.session_state[key] = None


def _run_with_log(fn):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = fn()
        st.session_state["log"] = buf.getvalue()
        return result
    except Exception:
        st.session_state["log"] = buf.getvalue() + "\n" + traceback.format_exc()
        raise


def _show_fig(fig) -> None:
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


def _hours(val) -> float | None:
    if val is None:
        return None
    if hasattr(val, "total_seconds"):
        return float(val.total_seconds()) / 3600.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _ss1_power_spread_limit_pct(hours: float) -> float:
    if hours <= 12:
        return 1.0
    if hours < 36:
        return 1.0 + (hours - 12.0) / 1200.0
    return 3.0


def _thin_df(df: pl.DataFrame, n_max: int = 15000) -> pl.DataFrame:
    if df.height <= n_max:
        return df
    step = max(1, int((df.height + n_max - 1) / n_max))
    return df.gather_every(step)


def _ss1_blocks(ss1: dict, df_tcc: pl.DataFrame) -> list[dict]:
    n = int(ss1.get("N_TCC_per_block") or 0)
    start = int(ss1.get("start_TCC_id") or 0)
    if n <= 0:
        return []
    blocks = []
    for i, label in enumerate(("A", "B", "C")):
        t0 = start + i * n
        t1 = start + (i + 1) * n - 1
        part = df_tcc.filter(
            (pl.col("TCC_id") >= t0) & (pl.col("TCC_id") <= t1)
        ).sort("TCC_id")
        if part.height == 0:
            continue
        blocks.append(
            {
                "blok": label,
                "TCC_van": t0,
                "TCC_tot": t1,
                "n_TCC": part.height,
                "start": part["TCC_start_datetime"].min(),
                "einde": part["TCC_end_datetime"].max(),
            }
        )
    return blocks


def _plotly_power_temp(ss1: dict | None, zoom_ss1: bool) -> go.Figure | None:
    plot_data = st.session_state.get("plot_data") or {}
    df_p = plot_data.get("df_meas_data_w_median")
    df_tcc = st.session_state.get("df_TCC_w_streaks_and_means")
    if df_p is None or df_tcc is None:
        return None

    time_c = analysis.time_col
    p_raw = analysis.val_col
    p_smooth = analysis.val_col_smooth_short
    df_p = _thin_df(df_p.select(
        [c for c in (time_c, p_raw, p_smooth) if c in df_p.columns]
    ).sort(time_c))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Vermogen P", "Temperatuur T (per TCC)"),
        row_heights=[0.48, 0.52],
    )
    x_p = df_p[time_c].to_list()
    if p_raw in df_p.columns:
        fig.add_trace(
            go.Scattergl(
                x=x_p,
                y=df_p[p_raw].to_list(),
                name="P (ruw)",
                mode="lines",
                line=dict(width=0.8, color="#9aa5b1"),
                hovertemplate="%{x}<br>P=%{y:.3f} W<extra>P ruw</extra>",
            ),
            row=1,
            col=1,
        )
    if p_smooth in df_p.columns:
        fig.add_trace(
            go.Scattergl(
                x=x_p,
                y=df_p[p_smooth].to_list(),
                name="P (mediaan 4 s)",
                mode="lines",
                line=dict(width=1.2, color="#c0392b"),
                hovertemplate="%{x}<br>P=%{y:.3f} W<extra>P glad</extra>",
            ),
            row=1,
            col=1,
        )
    threshold = plot_data.get("threshold")
    if threshold is not None:
        fig.add_hline(
            y=float(threshold),
            line_dash="dash",
            line_color="#1e8449",
            annotation_text="drempel",
            row=1,
            col=1,
        )

    temp_cols = [
        c for c in df_tcc.columns if c.startswith("temp_mean_")
    ]
    x_t = df_tcc["TCC_start_datetime"].to_list()
    tcc_ids = df_tcc["TCC_id"].to_list() if "TCC_id" in df_tcc.columns else None
    for col in temp_cols:
        label = col.replace("temp_mean_", "T ")
        fig.add_trace(
            go.Scattergl(
                x=x_t,
                y=df_tcc[col].to_list(),
                name=label,
                mode="lines+markers",
                marker=dict(size=4),
                customdata=tcc_ids,
                hovertemplate="%{x}<br>"
                + label
                + "=%{y:.3f} °C"
                + ("<br>TCC %{customdata}<extra></extra>" if tcc_ids is not None else "<extra></extra>"),
            ),
            row=2,
            col=1,
        )
    if analysis.power_col in df_tcc.columns:
        fig.add_trace(
            go.Scattergl(
                x=x_t,
                y=df_tcc[analysis.power_col].to_list(),
                name="P per TCC",
                mode="markers",
                marker=dict(size=5, color="#1a5276"),
                customdata=tcc_ids,
                hovertemplate="%{x}<br>P TCC=%{y:.3f} W"
                + ("<br>TCC %{customdata}<extra></extra>" if tcc_ids is not None else "<extra></extra>"),
                visible="legendonly",
            ),
            row=1,
            col=1,
        )

    blocks = _ss1_blocks(ss1, st.session_state.get("df_TCC")) if ss1 else []
    block_colors = {"A": "#5dade2", "B": "#58d68d", "C": "#f5b041"}
    for blk in blocks:
        fig.add_vrect(
            x0=blk["start"],
            x1=blk["einde"],
            fillcolor=block_colors.get(blk["blok"], "#aab7b8"),
            opacity=0.18,
            line_width=0,
            annotation_text=f"blok {blk['blok']}",
            annotation_position="top left",
            row=1,
            col=1,
        )
        fig.add_vrect(
            x0=blk["start"],
            x1=blk["einde"],
            fillcolor=block_colors.get(blk["blok"], "#aab7b8"),
            opacity=0.18,
            line_width=0,
            row=2,
            col=1,
        )
    if ss1 and ss1.get("period_start_datetime") is not None:
        t0 = ss1["period_start_datetime"]
        dur = ss1.get("period_duration")
        t1 = t0 + dur if dur is not None else None
        if t1 is not None:
            for row in (1, 2):
                fig.add_vline(x=t0, line_dash="dot", line_color="#1a5276", row=row, col=1)
                fig.add_vline(x=t1, line_dash="dot", line_color="#1a5276", row=row, col=1)
            if zoom_ss1:
                pad = (t1 - t0) * 0.2 if hasattr(t1, "__sub__") else None
                if pad is not None:
                    fig.update_xaxes(range=[t0 - pad, t1 + pad])

    fig.update_yaxes(title_text="W", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_xaxes(title_text="Tijd", row=2, col=1)
    fig.update_layout(
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=40, b=40),
        uirevision="pt-zoom",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=2, col=1)
    return fig


def _logger_temp_columns(df: pl.DataFrame) -> list[str]:
    skip = {
        analysis.time_col,
        analysis.val_col,
        analysis.val_col_smooth_short,
        analysis.val_col_smooth_long,
        analysis.duration_col,
        analysis.power_col,
        "meas_duration",
    }
    cols = []
    for name in df.columns:
        if name in skip:
            continue
        if "(C)" in name or name.lower().startswith("temp"):
            cols.append(name)
    return cols


def _compartment_sensor_names() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for name, spec in analysis.compartments.items():
        mapping[name] = list(spec.get("temp sensors") or [])
    return mapping


def _view_file_key(test: dict) -> tuple | None:
    temp = test.get("temp_file") or ""
    if not temp:
        return None
    temp_path = _resolve_temp_path(temp)
    if not Path(temp_path).is_file():
        return None
    pool = _power_pool(test)
    if not pool:
        return None
    matches = test.get("power_selected") or []
    if not matches:
        _sync_power_selection(test)
        if test.get("power_select_error"):
            return None
        matches = test.get("power_selected") or []
    matches = tuple(p for p in matches if Path(p).is_file())
    if not matches:
        return None
    return (temp_path, matches)


def _aligned_view_df(test: dict) -> pl.DataFrame:
    """Aligned vermogen+temperatuur via load_and_align_data (geen TCC-analyse)."""
    key = _view_file_key(test)
    if key is None:
        raise FileNotFoundError(
            "Geen power- en temperatuurbestanden die bij deze meting passen. "
            "Kies bestanden in het tabblad Meetpunten."
        )
    if st.session_state.get("view_key") == key and st.session_state.get("view_df") is not None:
        return st.session_state["view_df"]
    temp_path, matches = key
    df = analysis.load_and_align_data(list(matches), temp_path).collect()
    st.session_state["view_key"] = key
    st.session_state["view_df"] = df
    return df


def _plotly_view_measurement(
    df: pl.DataFrame,
    extra_channels: list[str],
) -> go.Figure:
    time_c = analysis.time_col
    p_col = analysis.power_col if analysis.power_col in df.columns else analysis.val_col
    temp_cols = _logger_temp_columns(df)
    comp_map = _compartment_sensor_names()
    keep = [time_c]
    if p_col in df.columns:
        keep.append(p_col)
    keep.extend(c for c in temp_cols if c in df.columns)
    plot_df = _thin_df(df.select([c for c in keep if c in df.columns]).sort(time_c))
    x = plot_df[time_c].to_list()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Vermogen P", "Temperatuur T"),
        row_heights=[0.42, 0.58],
    )
    if p_col in plot_df.columns:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=plot_df[p_col].to_list(),
                name="P",
                mode="lines",
                line=dict(width=1.0, color="#c0392b"),
                hovertemplate="%{x}<br>P=%{y:.3f} W<extra>P</extra>",
            ),
            row=1,
            col=1,
        )

    for comp_name, sensors in comp_map.items():
        present = [s for s in sensors if s in plot_df.columns]
        if not present:
            continue
        if len(present) == 1:
            series = plot_df[present[0]]
        else:
            series = plot_df.select(
                pl.mean_horizontal([pl.col(s) for s in present]).alias("mean")
            )["mean"]
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=series.to_list(),
                name=f"T {comp_name}",
                mode="lines",
                line=dict(width=1.4),
                hovertemplate="%{x}<br>T "
                + comp_name
                + "=%{y:.3f} °C<extra>T "
                + comp_name
                + "</extra>",
            ),
            row=2,
            col=1,
        )

    for col in extra_channels:
        if col not in plot_df.columns:
            continue
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=plot_df[col].to_list(),
                name=col,
                mode="lines",
                line=dict(width=1.0, dash="dot"),
                hovertemplate="%{x}<br>" + col + "=%{y:.3f} °C<extra>" + col + "</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="W", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_xaxes(title_text="Tijd", row=2, col=1)
    fig.update_layout(
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=40, b=40),
        uirevision="view-meas-zoom",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=2, col=1)
    return fig


def _render_ss2(test: dict | None) -> None:
    ss2 = (test or {}).get("ss2") or st.session_state.get("ss2")
    n_ev = (test or {}).get("n_defrost_events", 0)
    if n_ev < 2 and not ss2:
        return
    st.markdown("**SS2 — IEC 62552-3 B.4.3**")
    st.caption(
        "PSS2 = (EendY − EendX − ΔEdf) / (tendY − tendX)  (formule 12). "
        "Periode X eindigt bij de start van defrost j, periode Y bij de start van j+1. "
        "ΔEdf, ΔThdf en Δtdr komen uit Annex C (DF1) voor de defrost aan het einde van X. "
        "Daarna B.5 (Tamb-correctie). De hele-cyclus 24·P is alleen ter vergelijking."
    )
    if not ss2:
        st.info(
            "SS2 is niet berekend: er zijn minstens twee ontdooi-events nodig "
            "met een volledig interval ertussen."
        )
        return
    s1, s2, s3, s4 = st.columns(4)
    if ss2.get("b43_pss2_w") is not None:
        s1.metric("PSS2 (12)", f"{ss2['b43_pss2_w']:.3f} W")
        s2.metric("Pss B.5", f"{ss2['b43_pss_w']:.3f} W")
        s3.metric("Edaily 24·Pss", f"{ss2['b43_edaily_kwh']:.4f} kWh/d")
    else:
        s1.metric("P cyclus", f"{ss2['p_mean_w']:.3f} W")
        s2.metric("Pss (cyclus, B.5)", f"{ss2['pss_w']:.3f} W")
        s3.metric("24·P cyclus", f"{ss2['edaily_kwh']:.4f} kWh/d")
    s4.metric(
        "Geldige B.4.3-intervallen",
        f"{ss2.get('b43_n_valid', 0)} / {ss2['n_intervals']}",
    )
    st.caption(
        f"Hele cyclus (incl. defrost, niet formule 12): P = {ss2['p_mean_w']:.3f} W, "
        f"24·P = {ss2['edaily_kwh_measured']:.4f} kWh/d, duur {ss2['duration_h']:.3f} h. "
        f"ΔCOP in B.5 = {ss2.get('delta_cop', 0):.3f} /K (Tabel B.1: 0 bij 16 °C, "
        f"−0,014 bij 32 °C voor meerdere compartimenten)."
    )
    point = (test or {}).get("point")
    if point and ss2.get("b43_edaily_kwh") is not None:
        try:
            edaily_ss1 = en62552.TestPoint(
                **{k: v for k, v in point.items() if k in _TESTPOINT_FIELDS}
            ).edaily_kwh
            delta = ss2["b43_edaily_kwh"] - edaily_ss1
            st.caption(
                f"24·Pss(SS2,B.5) t.o.v. Edaily SS1: {delta:+.4f} kWh/d"
                + (f" ({100.0 * delta / edaily_ss1:+.1f} %)" if edaily_ss1 else "")
            )
        except TypeError:
            pass

    for i, iv in enumerate(ss2.get("b43_intervals") or [], start=1):
        titel = f"Interval {i}: {iv['start']} → {iv['end']}"
        if iv.get("b43_ok"):
            titel += " — B.4.3 geldig"
        else:
            titel += " — B.4.2/C niet volledig geldig"
        with st.expander(titel, expanded=i == 1):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("tendY − tendX", f"{iv['duration_h']:.3f} h")
            m2.metric("EendY − EendX", f"{iv['energy_wh']:.2f} Wh")
            m3.metric(
                "PSS2 (12)",
                f"{iv['PSS2_w']:.3f} W" if iv.get("PSS2_w") is not None else "—",
            )
            m4.metric(
                "Pss B.5",
                f"{iv['pss_w']:.3f} W" if iv.get("pss_w") is not None else "—",
            )
            st.caption(
                "Periode Y grenst aan de volgende defrost."
                if iv.get("y_adjacent_to_next_defrost")
                else "Periode Y is niet aan de volgende defrost grenzend (B.4.1 48 h-uitzondering)."
            )
            x, y = iv.get("period_X"), iv.get("period_Y")
            if x and y:
                st.markdown(
                    f"**Periode X** (einde = start defrost j): TCC {x['tcc_id_start']}–{x['tcc_id_end']}, "
                    f"n = {x['n_tcc']}, {x['start']} → {x['end']}, "
                    f"duur {x['duration_h']:.3f} h, PX = {x['p_mean_w']:.3f} W, "
                    f"CRtX = {x['crt_pct']:.2f} %, E_X = {x['energy_wh']:.2f} Wh, "
                    f"RtX = {x['on_h']:.4f} h."
                )
                st.markdown(
                    f"**Periode Y** (einde = start defrost j+1): TCC {y['tcc_id_start']}–{y['tcc_id_end']}, "
                    f"n = {y['n_tcc']}, {y['start']} → {y['end']}, "
                    f"duur {y['duration_h']:.3f} h, PY = {y['p_mean_w']:.3f} W, "
                    f"CRtY = {y['crt_pct']:.2f} %, E_Y = {y['energy_wh']:.2f} Wh, "
                    f"RtY = {y['on_h']:.4f} h. "
                    f"Max. lengte X/Y volgens B.4.2: {iv.get('max_xy_h', 0):.2f} h "
                    f"(max van 8 h en 50 % van het interval)."
                )
            else:
                st.warning("Geen perioden X/Y met minstens 4 TCC’s voor dit interval.")
            if iv.get("checks"):
                st.markdown("**B.4.2 acceptatie**")
                st.dataframe(
                    [
                        {
                            "Criterium": c["criterium"],
                            "Waarde": c["waarde"],
                            "Limiet": c["limiet"],
                            "OK": "ja" if c["ok"] else "nee",
                        }
                        for c in iv["checks"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
            df1 = iv.get("df1")
            if df1:
                st.markdown("**Annex C DF1 voor de defrost aan het einde van X (C.3.3)**")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("ΔEdfj (19)", f"{df1['Delta_E_dfj_wh']:.3f} Wh")
                d2.metric("Δtdrj (21)", f"{df1['delta_tdr_h']:.4f} h")
                d3.metric("PSS-D", f"{df1['P_SS_D']:.3f} W")
                d4.metric("PSS-F", f"{df1['P_SS_F']:.3f} W")
                st.caption(
                    f"D: TCC {df1['D_start_TCC_id']}, {df1['t_start_D']}, "
                    f"PSS-D = {df1['P_SS_D']:.3f} W, CRtD = {df1['CRt_D_pct']:.2f} %, "
                    f"RtD = {df1['on_D_h']:.4f} h. "
                    f"F: TCC {df1['F_start_TCC_id']}, tot {df1['t_end_F']}, "
                    f"PSS-F = {df1['P_SS_F']:.3f} W, CRtF = {df1['CRt_F_pct']:.2f} %, "
                    f"RtF = {df1['on_F_h']:.4f} h. "
                    f"Venster D→F: {df1['delta_time_h']:.4f} h, "
                    f"E = {df1['E_D_to_F_wh']:.3f} Wh, "
                    f"ΔEdf = {df1['E_D_to_F_wh']:.3f} − {df1['delta_time_h']:.4f}×"
                    f"({df1['P_SS_D']:.3f}+{df1['P_SS_F']:.3f})/2 = {df1['Delta_E_dfj_wh']:.3f} Wh. "
                    f"n = {df1['n_tcc_per_block']} TCC/blok."
                )
            f12 = iv.get("formula_12")
            if f12:
                st.markdown("**Formule (12)**")
                st.code(f12["tekst"])
            f14 = iv.get("formula_14")
            if f14:
                st.markdown("**Formule (14)**")
                st.code(f14["tekst"])
            rows13 = iv.get("formula_13") or []
            if rows13:
                st.markdown("**Formule (13) — TSS2 per compartiment**")
                st.dataframe(
                    [
                        {
                            "Compartiment": r["compartiment"],
                            "TendX °C": round(r["T_endX_C"], 4),
                            "TendY °C": round(r["T_endY_C"], 4),
                            "ΔThdf Kh": round(float(r["dTh_df_Kh"] or 0), 4),
                            "TSS2 °C": round(r["TSS2_C"], 4),
                            "Berekening": r["tekst"],
                        }
                        for r in rows13
                    ],
                    hide_index=True,
                    width="stretch",
                )
            if iv.get("pss_w") is not None and iv.get("PSS2_w") is not None:
                st.markdown("**B.5 Tamb-correctie**")
                st.caption(
                    f"PSSM = PSS2 = {iv['PSS2_w']:.4f} W, Tat = {ss2.get('nominal_ambient')} °C, "
                    f"ΔCOP = {ss2.get('delta_cop')}. "
                    f"Pss = {iv['pss_w']:.4f} W. "
                    f"TSS2: {_temp_summary_line(None, iv.get('temps_ss2'))}."
                )


def _render_ss1_tcc_context() -> None:
    """Als er geen 3-blok-perioden zijn: toon cycli/streaks zodat SS1-falen verklaarbaar is."""
    means = st.session_state.get("df_TCC_w_streaks_and_means")
    tcc = st.session_state.get("df_TCC")
    summary = st.session_state.get("ss1_diag_summary") or {}
    n_tcc = int(summary.get("n_tcc") or (0 if tcc is None else tcc.height))
    n_streaks = int(summary.get("n_streaks") or 0)
    st.markdown(
        f"**Geen 3-blok-perioden gebouwd.** {n_tcc} compressorcyclus/cycli in "
        f"{n_streaks or 'onbekend aantal'} streak(s). Een SS1-periode is drie "
        "opeenvolgende blokken van n cycli; streaks van 1 cyclus (vaak omdat elke "
        f"cyclus als ontdooi telt, drempel P > {float(st.session_state.get('defrost_power_w', 50.0)):g} W) vallen af. Na het weglaten van "
        "de eerste twee en laatste TCC van een streak blijven er minstens 3 cycli nodig."
    )
    if means is None or getattr(means, "height", 0) == 0:
        return
    if "streak_id" in means.columns:
        streak = (
            means.group_by("streak_id")
            .agg(
                n_TCC=pl.len(),
                n_defrost=(
                    pl.col("is_defrost_cycle").sum()
                    if "is_defrost_cycle" in means.columns
                    else pl.lit(None)
                ),
                start=pl.col("TCC_start_datetime").min()
                if "TCC_start_datetime" in means.columns
                else pl.lit(None),
                einde=pl.col("TCC_end_datetime").max()
                if "TCC_end_datetime" in means.columns
                else pl.lit(None),
            )
            .sort("streak_id")
        )
        st.dataframe(streak.to_pandas(), hide_index=True, width="stretch")
    show = [
        c
        for c in (
            "TCC_id",
            "streak_id",
            "TCC_start_datetime",
            "TCC_duration",
            "TCC_on_duration",
            analysis.power_col,
            "is_defrost_cycle",
        )
        if c in means.columns
    ]
    if show:
        st.caption("Alle gedetecteerde TCC’s")
        st.dataframe(means.select(show).to_pandas(), hide_index=True, width="stretch")


def _ss1_diag_display(diag) -> None:
    if diag is None or getattr(diag, "height", 0) == 0:
        _render_ss1_tcc_context()
        return
    summary = st.session_state.get("ss1_diag_summary") or {}
    n = summary.get("n_periods", diag.height)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Kandidaten", str(n))
    c2.metric("Duur ≥ 6 h", str(summary.get("n_ok_duration", "—")))
    c3.metric("Temp-spread OK", str(summary.get("n_ok_temp_spread", "—")))
    c4.metric("|P-helling| OK", str(summary.get("n_ok_power_slope", "—")))
    c5.metric("Start−1/−2 OK", str(summary.get("n_ok_prev_two", "—")))
    pdf = diag.head(80).to_pandas()
    rows = []
    for rec in pdf.to_dict("records"):
        rows.append(
            {
                "Start TCC": rec.get("start_TCC_id"),
                "n / blok": rec.get("N_TCC_per_block"),
                "Aantal TCC": rec.get("n_TCC"),
                "Duur (h)": round(float(rec.get("period_hours") or 0), 3),
                "P gem. (W)": round(float(rec.get("power_avg") or 0), 3),
                "P-spread (%)": round(float(rec.get("power_spread_pct") or 0), 3),
                "P-limiet (%)": round(float(rec.get("power_spread_limit_pct") or 0), 3),
                "P-helling (%/h)": round(float(rec.get("power_slope_pct") or 0), 4),
                "T-spread max (K)": (
                    None
                    if rec.get("max_temp_spread_K") is None
                    else round(float(rec.get("max_temp_spread_K")), 4)
                ),
                "T-helling max (K/h)": (
                    None
                    if rec.get("max_temp_slope_K_per_h") is None
                    else round(float(rec.get("max_temp_slope_K_per_h")), 4)
                ),
                "SS1": "ja" if rec.get("ok_ss1") else "nee",
                "Waarom niet": rec.get("fail_reason") or "",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Eis: duur ≥ 6 h, temp-spread < 0,25 K, |P-helling| < 0,25 %/h, "
        "P-spread volgens duurklasse, en de perioden die 1 en 2 TCC eerder starten "
        "moeten ook aan die eisen voldoen. Maximaal 80 rijen, best-eerst."
    )


def _render_ss1_decision() -> None:
    ss1 = st.session_state.get("ss1_opti")
    if not ss1:
        st.markdown("**Waarom SS1 niet is gehaald**")
        _ss1_diag_display(st.session_state.get("ss1_diagnostics"))
        return
    n = int(ss1.get("N_TCC_per_block") or 0)
    start = int(ss1.get("start_TCC_id") or 0)
    hours = _hours(ss1.get("period_duration")) or 0.0
    sel = list(analysis.compartments_used_for_period_selection)
    st.markdown("**Hoe SS1 is gekozen**")
    st.markdown(
        f"""
Een periode bestaat uit **drie opeenvolgende blokken** van elk **n** compressorcycli (TCC).
Eerst vallen streaks rond ontdooiing af (eerste twee en laatste TCC van een streak).
Daarna blijven alleen perioden over die aan de ISO 16358 / Annex B-filters voldoen.
Van die kandidaten wint de periode die het **vroegst eindigt** (kortste tijd vanaf de start
van de meting). Doel: zien of er lang genoeg gemeten is. Bij gelijke eindtijd: kleinste n,
daarna kortste duur.

Gekozen: **n = {n}** TCC per blok → **3 × {n} = {3 * n}** TCC’s
(`TCC {start}` t/m `TCC {start + 3 * n - 1}`), streak `{ss1.get("streak_id")}`.
Periodeselectie op compartiment(en): **{', '.join(sel) if sel else '—'}**.
        """
    )

    df_tcc = st.session_state.get("df_TCC")
    blocks = _ss1_blocks(ss1, df_tcc) if df_tcc is not None else []
    if blocks:
        st.dataframe(
            [
                {
                    "Blok": b["blok"],
                    "TCC van": b["TCC_van"],
                    "TCC tot": b["TCC_tot"],
                    "Aantal TCC": b["n_TCC"],
                    "Start": str(b["start"]),
                    "Einde": str(b["einde"]),
                }
                for b in blocks
            ],
            hide_index=True,
            width="stretch",
        )

    p_lim = _ss1_power_spread_limit_pct(hours)
    checks = [
        ("Duur ≥ 6 h", f"≥ 6 h", f"{hours:.3f} h", hours >= 6),
        (
            "Vermogensspread",
            f"≤ {p_lim:.3f} %",
            f"{float(ss1.get('power_spread_pct') or 0):.3f} % "
            f"({float(ss1.get('power_spread') or 0):.3f} W)",
            float(ss1.get("power_spread_pct") or 0) <= p_lim + 1e-9,
        ),
        (
            "|Vermogenshelling|",
            "< 0,25 %/h",
            f"{abs(float(ss1.get('power_slope_pct') or 0)):.4f} %/h",
            abs(float(ss1.get("power_slope_pct") or 0)) < 0.25,
        ),
        (
            "Vorige 2 perioden ook geldig",
            "start−1 en start−2 bij dezelfde n",
            "ja" if ss1.get("prev_two_periods_also_valid") else "nee / n.v.t.",
            bool(ss1.get("prev_two_periods_also_valid", True)),
        ),
    ]
    for col in analysis.mean_temp_cols_for_period_selection:
        key = f"temp_spread_{col}"
        name = col.replace("temp_mean_", "")
        val = float(ss1.get(key) or 0)
        checks.append(
            (f"Temp-spread {name}", "< 0,25 K", f"{val:.4f} K", val < 0.25)
        )
    st.dataframe(
        [
            {
                "Criterium": c[0],
                "Eis": c[1],
                "Deze SS1": c[2],
                "OK": "ja" if c[3] else "nee",
            }
            for c in checks
        ],
        hide_index=True,
        width="stretch",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P gemiddeld (SS1)", f"{float(ss1.get('power_avg') or 0):.3f} W")
    k2.metric("CRtSS", f"{float(ss1.get('period_steady_state_compressor_run_time') or 0):.1f} %")
    k3.metric("Pss (na Tamb-correctie)", f"{float(st.session_state.get('pss') or 0):.3f} W")
    k4.metric("Start-TCC", str(start))

    tcc_means = st.session_state.get("df_TCC_w_streaks_and_means")
    if tcc_means is not None and n > 0:
        window = tcc_means.filter(
            (pl.col("TCC_id") >= start) & (pl.col("TCC_id") < start + 3 * n)
        ).sort("TCC_id")
        show_cols = [
            c
            for c in (
                "TCC_id",
                "TCC_start_datetime",
                "TCC_duration",
                "TCC_on_duration",
                analysis.power_col,
                "is_defrost_cycle",
                "streak_id",
                *analysis.mean_temp_cols,
                "temp_mean_amb",
            )
            if c in window.columns
        ]
        st.markdown("**TCC’s in de gekozen SS1**")
        st.dataframe(window.select(show_cols).to_pandas(), hide_index=True, width="stretch")

    cands = st.session_state.get("ss1_candidates")
    all_p = st.session_state.get("df_all_possible_periods_w_stats")
    n_all = all_p.height if all_p is not None else 0
    n_ok = cands.height if cands is not None else 0
    st.caption(
        f"Doorgezocht: {n_all} mogelijke 3-blok-perioden; {n_ok} blijven over na de filters. "
        "De tabel hieronder is die set, gesorteerd zoals de keuze (vroegste einde eerst)."
    )
    if cands is not None and cands.height:
        pdf = cands.head(40).to_pandas()
        keep = [
            c
            for c in pdf.columns
            if c
            in {
                "streak_id",
                "start_TCC_id",
                "N_TCC_per_block",
                "period_duration",
                "period_start_datetime",
                "period_end_datetime",
                "power_avg",
                "power_spread",
                "power_spread_pct",
                "power_slope_pct",
                "period_steady_state_compressor_run_time",
                "prev_two_periods_also_valid",
            }
            or str(c).startswith("temp_spread_")
            or str(c).startswith("temp_mean_")
        ]
        pdf = pdf[keep]
        pdf.insert(0, "gekozen", False)
        if len(pdf):
            pdf.loc[pdf.index[0], "gekozen"] = True
        st.dataframe(pdf, hide_index=True, width="stretch")


def _resolve_temp_path(temp_file: str) -> str:
    path = Path(temp_file)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return str(path)


UPLOAD_ROOT = PROJECT_DIR / "_uploads"


def _is_local_client() -> bool:
    """True als de browser op dezelfde pc draait als de server (native schijfkiezer mag)."""
    if getattr(sys, "frozen", False):
        return True
    try:
        ip = getattr(st.context, "ip_address", None)
        if ip in {"127.0.0.1", "::1", "localhost"}:
            return True
        if ip:
            return False
    except Exception:
        pass
    try:
        host = str(st.context.headers.get("Host") or "")
        hostname = host.split(":")[0].strip().lower()
        return hostname in {"localhost", "127.0.0.1", "::1", "[::1]"}
    except Exception:
        return False


def _safe_id(test_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(test_id))


def _save_uploaded(files, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.iterdir():
        if old.is_file():
            old.unlink()
    saved: list[Path] = []
    for uploaded in files:
        name = Path(uploaded.name).name or "bestand.csv"
        target = dest_dir / name
        target.write_bytes(uploaded.getvalue())
        saved.append(target)
    return saved


def _save_uploaded_power(files, dest_dir: Path) -> list[Path]:
    """CSV’s of één zip van de power-map; .CSV blijft behouden."""
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    zips = [f for f in files if Path(f.name).suffix.lower() == ".zip"]
    if len(zips) == 1 and len(files) == 1:
        for old in dest_dir.iterdir():
            if old.is_file():
                old.unlink()
        with zipfile.ZipFile(io.BytesIO(zips[0].getvalue())) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if name.startswith(".") or not name:
                    continue
                if Path(name).suffix.lower() not in {".csv", ".txt", ".dat"}:
                    continue
                target = dest_dir / name
                target.write_bytes(zf.read(info))
        return list(dest_dir.iterdir())
    return _save_uploaded(files, dest_dir)


def _pick_local_files(*, multiple: bool, title: str) -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        types = [
            ("CSV", "*.csv"),
            ("Text", "*.txt"),
            ("Alle bestanden", "*.*"),
        ]
        if multiple:
            chosen = filedialog.askopenfilenames(title=title, filetypes=types)
            paths = list(chosen) if chosen else []
        else:
            chosen = filedialog.askopenfilename(title=title, filetypes=types)
            paths = [chosen] if chosen else []
        root.destroy()
        return paths
    except Exception:
        st.warning("Bestandskiezer op deze pc is hier niet beschikbaar. Gebruik uploaden.")
        return []


def _pick_local_directory(title: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title=title)
        root.destroy()
        return chosen or None
    except Exception:
        st.warning("Mapkiezer op deze pc is hier niet beschikbaar. Gebruik uploaden.")
        return None


def _pick_local_saveas(title: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Alle bestanden", "*.*")],
        )
        root.destroy()
        return chosen or None
    except Exception:
        st.warning("Opslaan-dialoog is hier niet beschikbaar. Gebruik downloaden.")
        return None


def _list_power_files(paths: list[str] | str) -> list[str]:
    if isinstance(paths, str):
        folder = Path(paths)
        if folder.is_dir():
            found = []
            for pat in ("*.CSV", "*.csv", "*.txt", "*.dat"):
                found.extend(str(p) for p in folder.glob(pat) if p.is_file())
            return sorted(set(found))
        found = glob.glob(paths)
        if not found and not Path(paths).is_absolute():
            found = glob.glob(str(PROJECT_DIR / paths))
        return sorted(found)
    files: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(_list_power_files(str(path)))
        elif path.is_file():
            files.append(str(path))
    return sorted(set(files))


def _default_power_folder() -> str:
    cand = PROJECT_DIR / "power_meetdata_map"
    if cand.is_dir() and _list_power_files(str(cand)):
        return str(cand)
    return ""


def _ensure_campaign_power() -> None:
    pool = [p for p in (st.session_state.get("power_pool") or []) if Path(p).is_file()]
    folder = st.session_state.get("power_folder") or ""
    if pool:
        st.session_state["power_pool"] = pool
        if not folder:
            st.session_state["power_folder"] = str(Path(pool[0]).parent)
        return
    if folder:
        files = _list_power_files(folder)
        if files:
            st.session_state["power_pool"] = files
            return
    for t in st.session_state.get("tests") or []:
        legacy = t.get("power_pool") or t.get("power_paths")
        if legacy:
            files = [p for p in legacy if Path(p).is_file()]
            if files:
                st.session_state["power_pool"] = files
                st.session_state["power_folder"] = str(Path(files[0]).parent)
                return
        pattern = t.get("power_glob") or ""
        if pattern:
            files = _list_power_files(pattern)
            if files:
                st.session_state["power_pool"] = files
                st.session_state["power_folder"] = str(Path(files[0]).parent)
                return
    default = _default_power_folder()
    if default:
        st.session_state["power_folder"] = default
        st.session_state["power_pool"] = _list_power_files(default)


def _campaign_power_pool() -> list[str]:
    pool = [p for p in (st.session_state.get("power_pool") or []) if Path(p).is_file()]
    if pool:
        return pool
    folder = st.session_state.get("power_folder") or ""
    if folder:
        return _list_power_files(folder)
    return []


def _power_pool(test: dict) -> list[str]:
    camp = _campaign_power_pool()
    if camp:
        return camp
    pool = test.get("power_pool")
    if pool:
        return [p for p in pool if Path(p).is_file()]
    paths = test.get("power_paths")
    if paths:
        return [p for p in paths if Path(p).is_file()]
    pattern = test.get("power_glob") or ""
    if not pattern:
        return []
    return _list_power_files(pattern)


def _power_matches(test: dict) -> list[str]:
    return _power_pool(test)


def _power_arg(test: dict) -> str | list[str]:
    matches = test.get("power_selected") or _power_matches(test)
    matches = [p for p in matches if Path(p).is_file()]
    return matches if matches else (st.session_state.get("power_folder") or "")


def _sync_power_selection(test: dict) -> None:
    test.pop("power_select_error", None)
    pool = _power_pool(test)
    temp = test.get("temp_file") or ""
    temp_path = _resolve_temp_path(temp) if temp else ""
    if not pool or not temp_path or not Path(temp_path).is_file():
        test["power_selected"] = []
        return
    try:
        chosen = analysis.select_power_files_for_temp(pool, temp_path)
    except Exception as exc:
        test["power_selected"] = []
        test["power_select_error"] = str(exc)
        return
    test["power_selected"] = chosen
    tmin, tmax = analysis.temp_file_span(temp_path)
    test["temp_span"] = (str(tmin), str(tmax))


def _apply_campaign_power(paths: list[str] | str) -> None:
    files = _list_power_files(paths)
    st.session_state["power_pool"] = files
    if isinstance(paths, str) and Path(paths).is_dir():
        st.session_state["power_folder"] = paths
    elif files:
        st.session_state["power_folder"] = str(Path(files[0]).parent)
    elif isinstance(paths, str):
        st.session_state["power_folder"] = paths
    glob = (
        str(Path(st.session_state["power_folder"]) / "*")
        if st.session_state.get("power_folder")
        else ""
    )
    if "power_folder_text" in st.session_state:
        st.session_state["power_folder_text"] = st.session_state.get("power_folder") or ""
    for t in st.session_state.get("tests") or []:
        t["power_pool"] = files
        t["power_paths"] = files
        t["power_glob"] = glob
        t.pop("_power_upload_sig", None)
        _sync_power_selection(t)


def _apply_temp_folder(test: dict, folder: str, index: int | None = None) -> None:
    test["temp_folder"] = folder or ""
    files = _list_power_files(folder) if folder else []
    test["temp_candidates"] = files
    if len(files) == 1:
        test["temp_file"] = files[0]
    elif test.get("temp_file") and files and test["temp_file"] not in files:
        resolved = _resolve_temp_path(test["temp_file"])
        match = next((p for p in files if Path(p).name == Path(resolved).name), None)
        test["temp_file"] = match or ""
    elif not files:
        if not (test.get("temp_file") and Path(_resolve_temp_path(test["temp_file"])).is_file()):
            test["temp_file"] = ""
    if index is not None:
        if f"tf_{index}" in st.session_state:
            st.session_state[f"tf_{index}"] = folder
        if f"t_{index}" in st.session_state:
            st.session_state[f"t_{index}"] = test.get("temp_file") or ""
    _sync_power_selection(test)


def _campaign_settings() -> dict:
    return {
        "version": _SETTINGS_VERSION,
        "power_folder": st.session_state.get("power_folder") or "",
        "compartments": _compartment_records(_as_comp_df(st.session_state["compartments_df"])),
        "ambient_sensors": st.session_state.get("ambient_sensors"),
        "gradient_sensors": st.session_state.get("gradient_sensors"),
        "defrost_power_w": st.session_state.get("defrost_power_w"),
        "heater_advances_timer": st.session_state.get("heater_advances_timer"),
        "tests": [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "ambient_c": t.get("ambient_c"),
                "vs_target": t.get("vs_target"),
                "temp_folder": t.get("temp_folder") or "",
                "temp_file": t.get("temp_file") or "",
                "point": t.get("point"),
                "tcrt_hours": t.get("tcrt_hours") or [],
                "elapsed_hours": t.get("elapsed_hours") or [],
                "n_defrost_events": t.get("n_defrost_events") or 0,
                "defrost_events": t.get("defrost_events") or [],
                "power_gaps": t.get("power_gaps") or [],
                "vs_target_warning": t.get("vs_target_warning"),
            }
            for t in st.session_state.get("tests") or []
        ],
    }


def _apply_campaign_settings(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("Ongeldig instellingenbestand.")
    comps = data.get("compartments")
    if comps is not None:
        st.session_state["compartments_df"] = _as_comp_df(comps)
        st.session_state["compartments_ui"] = _compartment_records(_as_comp_df(comps))
    for key in ("ambient_sensors", "gradient_sensors"):
        if data.get(key) is not None:
            st.session_state[key] = data[key]
    if data.get("defrost_power_w") is not None:
        st.session_state["defrost_power_w"] = float(data["defrost_power_w"])
    if "heater_advances_timer" in data:
        st.session_state["heater_advances_timer"] = bool(data["heater_advances_timer"])
    folder = data.get("power_folder") or ""
    tests_in = data.get("tests")
    if tests_in:
        restored = []
        for item in tests_in:
            t = _blank_test(
                item.get("name") or "Meting",
                int(item.get("ambient_c") or 32),
                item.get("vs_target") or "onder",
                test_id=item.get("id"),
            )
            t["temp_folder"] = item.get("temp_folder") or ""
            t["temp_file"] = item.get("temp_file") or ""
            t["point"] = item.get("point")
            t["tcrt_hours"] = item.get("tcrt_hours") or []
            t["elapsed_hours"] = item.get("elapsed_hours") or []
            t["n_defrost_events"] = item.get("n_defrost_events") or 0
            t["defrost_events"] = item.get("defrost_events") or []
            t["power_gaps"] = item.get("power_gaps") or []
            t["vs_target_warning"] = item.get("vs_target_warning")
            restored.append(t)
        st.session_state["tests"] = restored
        _migrate_tests(st.session_state["tests"])
    if folder:
        _apply_campaign_power(folder)
    else:
        for t in st.session_state.get("tests") or []:
            if t.get("temp_folder"):
                _apply_temp_folder(t, t["temp_folder"])
            elif t.get("temp_file"):
                _sync_power_selection(t)
    if tests_in:
        for i, t in enumerate(st.session_state.get("tests") or []):
            if t.get("temp_folder"):
                _apply_temp_folder(t, t["temp_folder"])
            elif t.get("temp_file"):
                _sync_power_selection(t)
            st.session_state[f"n_{i}"] = t.get("name") or ""
            st.session_state[f"a_{i}"] = int(t.get("ambient_c") or 32)
            st.session_state[f"vs_{i}"] = t.get("vs_target") or "onder"
            st.session_state[f"tf_{i}"] = t.get("temp_folder") or ""
            st.session_state[f"t_{i}"] = t.get("temp_file") or ""
    if folder:
        st.session_state["power_folder_text"] = folder


def _render_campaign_power() -> None:
    st.markdown("**Powermetingen (één map voor alle meetpunten)**")
    st.caption(
        "Zet alle wattmeter-CSV’s in één map. Per temperatuurmeting zoekt de app "
        "zelf de powerbestanden waarvan de tijdspanne overlap heeft."
    )
    local = _is_local_client()
    if local:
        mode = st.radio(
            "Power-map aanleveren",
            options=["schijf", "upload"],
            format_func=lambda v: (
                "Selecteren op deze pc" if v == "schijf" else "Uploaden (browser)"
            ),
            horizontal=True,
            key="power_folder_mode",
        )
    else:
        mode = "upload"
        st.caption("Deze sessie is op afstand: upload de power-CSV’s of een zip.")

    if mode == "upload":
        st.caption(
            "Yokogawa-files heten `.CSV` (hoofdletters). Alle extensies zijn toegestaan. "
            "Liever één zip van de hele power-map."
        )
        power_up = st.file_uploader(
            "Power-CSV’s of één zip",
            type=None,
            accept_multiple_files=True,
            key="power_folder_up",
            help="Alle bestanden, ook .CSV. Of één .zip van de power-map.",
        )
        if power_up:
            sig = tuple((f.name, f.size) for f in power_up)
            if st.session_state.get("_power_folder_upload_sig") != sig:
                dest = UPLOAD_ROOT / "_campaign" / "power"
                saved = _save_uploaded_power(power_up, dest)
                _apply_campaign_power(str(dest) if dest.is_dir() else [str(p) for p in saved])
                st.session_state["_power_folder_upload_sig"] = sig
    else:
        if st.button("Power-map kiezen…", key="pick_campaign_power"):
            folder = _pick_local_directory("Map met alle power-bestanden")
            if folder:
                files = _list_power_files(folder)
                if files:
                    _apply_campaign_power(folder)
                    st.rerun()
                else:
                    st.warning("In die map staan geen CSV/TXT/DAT-bestanden.")

    folder_key = "power_folder_text"
    if folder_key not in st.session_state:
        st.session_state[folder_key] = st.session_state.get("power_folder") or ""
    new_folder = st.text_input("Pad van de power-map", key=folder_key)
    if new_folder != (st.session_state.get("power_folder") or ""):
        if not new_folder or Path(new_folder).is_dir():
            _apply_campaign_power(new_folder)

    pool = _campaign_power_pool()
    folder = st.session_state.get("power_folder") or ""
    if pool:
        st.caption(f"{len(pool)} power-bestand(en) in {folder or Path(pool[0]).parent}")
    else:
        st.caption("Nog geen power-map gekozen.")


def _render_settings_io() -> None:
    st.markdown("**Instellingen opslaan en openen**")
    st.caption(
        "Bestandslocaties, gekozen temperatuurbestanden en compartimentgegevens. "
        "Lokaal: opslaan/openen op schijf. Overal: downloaden of uploaden als JSON."
    )
    local = _is_local_client()
    payload = json.dumps(_campaign_settings(), indent=2, default=str)
    cols = st.columns(4 if local else 2)
    if local:
        if cols[0].button("Opslaan op deze pc…", key="save_settings_local"):
            path = _pick_local_saveas("Campagne-instellingen opslaan")
            if path:
                Path(path).write_text(payload, encoding="utf-8")
                st.session_state["flash"] = f"Instellingen opgeslagen: {path}"
                st.rerun()
        if cols[1].button("Openen van deze pc…", key="open_settings_local"):
            chosen = _pick_local_files(multiple=False, title="Campagne-instellingen (JSON)")
            if chosen:
                data = json.loads(Path(chosen[0]).read_text(encoding="utf-8"))
                st.session_state["pending_settings"] = data
                st.session_state["flash"] = f"Instellingen geladen: {chosen[0]}"
                st.rerun()
        cols[2].download_button(
            "Downloaden (JSON)",
            data=payload,
            file_name="campagne_en62552.json",
            mime="application/json",
            key="dl_settings",
        )
        up_col = cols[3]
    else:
        cols[0].download_button(
            "Downloaden (JSON)",
            data=payload,
            file_name="campagne_en62552.json",
            mime="application/json",
            key="dl_settings",
        )
        up_col = cols[1]
    uploaded = up_col.file_uploader(
        "Uploaden (JSON)",
        type=["json"],
        accept_multiple_files=False,
        key="settings_up",
    )
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.get("_settings_upload_sig") != sig:
            data = json.loads(uploaded.getvalue().decode("utf-8"))
            st.session_state["_settings_upload_sig"] = sig
            st.session_state["pending_settings"] = data
            st.session_state["flash"] = f"Instellingen geladen: {uploaded.name}"
            st.rerun()


def _render_temp_files(test: dict, index: int) -> None:
    local = _is_local_client()
    if local:
        mode = st.radio(
            "Temperatuurmeting",
            options=["schijf", "upload"],
            format_func=lambda v: (
                "Selecteren op deze pc" if v == "schijf" else "Uploaden (browser)"
            ),
            horizontal=True,
            key=f"filemode_{index}",
            help="Kies een map. Staat er maar één CSV in, dan wordt die automatisch gebruikt.",
        )
    else:
        mode = "upload"
        st.caption("Deze sessie is op afstand: upload de temperatuur-CSV.")

    if mode == "upload":
        temp_up = st.file_uploader(
            "Temperatuur-CSV(s) of zip van de map",
            type=None,
            accept_multiple_files=True,
            key=f"temp_up_{index}",
            help="Eén bestand, meerdere bestanden, of een zip. Bij één bestand in de map wordt dat automatisch gekozen.",
        )
        if temp_up:
            sig = tuple((f.name, f.size) for f in temp_up)
            if test.get("_temp_upload_sig") != sig:
                dest = UPLOAD_ROOT / _safe_id(test["id"]) / "temp"
                saved = _save_uploaded_power(temp_up, dest)
                _apply_temp_folder(test, str(dest), index)
                if len(saved) == 1:
                    test["temp_file"] = str(saved[0])
                    _sync_power_selection(test)
                test["_temp_upload_sig"] = sig
    else:
        c1, c2 = st.columns(2)
        if c1.button("Temperatuurmap kiezen…", key=f"pick_td_{index}"):
            folder = _pick_local_directory("Map met temperatuurmeting")
            if folder:
                files = _list_power_files(folder)
                if files:
                    _apply_temp_folder(test, folder, index)
                    test.pop("_temp_upload_sig", None)
                    st.rerun()
                else:
                    st.warning("In die map staan geen CSV/TXT/DAT-bestanden.")
        if c2.button("Temperatuur-CSV kiezen…", key=f"pick_t_{index}"):
            chosen = _pick_local_files(multiple=False, title="Temperatuur-CSV")
            if chosen:
                test["temp_file"] = chosen[0]
                test["temp_folder"] = str(Path(chosen[0]).parent)
                test.pop("_temp_upload_sig", None)
                _sync_power_selection(test)
                st.rerun()

    tf_key, t_key = f"tf_{index}", f"t_{index}"
    if tf_key not in st.session_state:
        st.session_state[tf_key] = test.get("temp_folder") or ""
    if t_key not in st.session_state:
        st.session_state[t_key] = test.get("temp_file") or ""
    new_folder = st.text_input("Pad van de temperatuurmap", key=tf_key)
    if new_folder != (test.get("temp_folder") or ""):
        if not new_folder or Path(new_folder).is_dir():
            _apply_temp_folder(test, new_folder, index)
    else:
        test["temp_folder"] = new_folder

    candidates = test.get("temp_candidates")
    if candidates is None and test.get("temp_folder"):
        candidates = _list_power_files(test["temp_folder"])
        test["temp_candidates"] = candidates
    candidates = candidates or []
    if len(candidates) > 1:
        names = [Path(p).name for p in candidates]
        current = test.get("temp_file") or ""
        try:
            idx_cur = next(i for i, p in enumerate(candidates) if p == current)
        except StopIteration:
            idx_cur = 0
            test["temp_file"] = candidates[0]
            _sync_power_selection(test)
        pick = st.selectbox(
            "Temperatuurbestand in deze map",
            options=list(range(len(candidates))),
            index=idx_cur,
            format_func=lambda i: names[i],
            key=f"temp_choice_{index}",
        )
        if candidates[pick] != test.get("temp_file"):
            test["temp_file"] = candidates[pick]
            st.session_state[t_key] = candidates[pick]
            _sync_power_selection(test)
    elif len(candidates) == 1:
        st.caption(f"Automatisch gekozen: {Path(candidates[0]).name}")
    else:
        new_temp = st.text_input("Temperatuur-CSV (bestand)", key=t_key)
        if new_temp != (test.get("temp_file") or ""):
            test["temp_file"] = new_temp
            _sync_power_selection(test)
        else:
            test["temp_file"] = new_temp

    selected = [p for p in (test.get("power_selected") or []) if Path(p).is_file()]
    pool = _power_pool(test)
    if selected:
        st.caption(
            f"Power automatisch: {len(selected)} van {len(pool)} bestand(en) — "
            + ", ".join(Path(p).name for p in selected[:8])
            + ("…" if len(selected) > 8 else "")
        )
    elif pool and test.get("temp_file"):
        st.caption(f"{len(pool)} power-bestand(en) in de map, nog geen overlap met deze temperatuurmeting.")
    elif not pool:
        st.caption("Kies eerst de power-map bovenaan dit tabblad.")
    if test.get("power_select_error"):
        st.warning(test["power_select_error"])
    temp_ok = (
        Path(_resolve_temp_path(test.get("temp_file") or "")).is_file()
        if test.get("temp_file")
        else False
    )
    if temp_ok:
        span = test.get("temp_span")
        extra = f" ({span[0]} – {span[1]})" if span else ""
        st.caption(f"Temperatuur: {Path(test['temp_file']).name}{extra}")
    else:
        st.caption("Nog geen temperatuurbestand gekozen.")


def apply_appliance_config() -> dict:
    if not hasattr(analysis, "configure_appliance"):
        import importlib

        importlib.reload(analysis)
    rows = _compartment_records(_as_comp_df(st.session_state["compartments_df"]))
    if not rows:
        raise ValueError("Voeg minstens één compartiment toe in de tabel.")
    new = {}
    for row in rows:
        sensors = [s.strip() for s in str(row["sensors"]).split(",") if s.strip()]
        if not sensors:
            raise ValueError(f"Compartiment '{row['name']}' heeft geen sensors.")
        table_type = row["table_type"]
        if table_type not in en62552.COMPARTMENT_TARGETS_C:
            raise ValueError(f"Onbekend compartimenttype: {table_type}")
        new[row["name"]] = {
            "temp sensors": sensors,
            "volume": float(row["volume"]),
            "compartment type": en62552.frozen_unfrozen(table_type),
            "target_temp": en62552.COMPARTMENT_TARGETS_C[table_type],
            "table_type": table_type,
        }
    amb = [s.strip() for s in st.session_state["ambient_sensors"].split(",") if s.strip()]
    grad = [s.strip() for s in st.session_state["gradient_sensors"].split(",") if s.strip()]
    analysis.configure_appliance(
        new,
        ambient_sensors=amb,
        gradient_sensors=grad,
        defrost_power_w=float(st.session_state.get("defrost_power_w", 50.0)),
    )
    st.session_state["compartments_ui"] = rows
    return new


def targets_from_config() -> dict[str, float]:
    return {
        row["name"]: en62552.COMPARTMENT_TARGETS_C[row["table_type"]]
        for row in _compartment_records(_as_comp_df(st.session_state["compartments_df"]))
        if row["table_type"] in en62552.COMPARTMENT_TARGETS_C
    }


def _extract_defrost_intervals(df_means: pl.DataFrame) -> tuple[list[float], list[float], int]:
    needed = {"is_defrost_cycle", "TCC_id", "TCC_start_datetime", "TCC_on_duration"}
    if not needed.issubset(set(df_means.columns)):
        return [], [], 0
    df = df_means.sort("TCC_id")
    is_defrost = [bool(x) for x in df["is_defrost_cycle"].to_list()]
    n_events = 0
    in_event = False
    for flag in is_defrost:
        if flag and not in_event:
            n_events += 1
            in_event = True
        elif not flag:
            in_event = False
    on_col = df["TCC_on_duration"]
    try:
        on_hours = on_col.dt.total_hours(fractional=True).to_list()
    except Exception:
        on_hours = []
        for v in on_col.to_list():
            if v is None:
                on_hours.append(0.0)
            elif hasattr(v, "total_seconds"):
                on_hours.append(v.total_seconds() / 3600.0)
            else:
                on_hours.append(float(v))
    tcrt, elapsed = en62552.defrost_intervals_from_cycles(
        is_defrost,
        df["TCC_id"].to_list(),
        df["TCC_start_datetime"].to_list(),
        on_hours,
    )
    return tcrt, elapsed, n_events


def _defrost_event_records(
    df_means: pl.DataFrame,
    df1_details: list[dict] | None = None,
) -> list[dict]:
    needed = {"is_defrost_cycle", "TCC_id", "TCC_start_datetime"}
    if df_means is None or not needed.issubset(set(df_means.columns)):
        return []
    df = df_means.sort("TCC_id")
    is_defrost = [bool(x) for x in df["is_defrost_cycle"].to_list()]
    starts = df["TCC_start_datetime"].to_list()
    ends = (
        df["TCC_end_datetime"].to_list()
        if "TCC_end_datetime" in df.columns
        else starts
    )
    ids = df["TCC_id"].to_list()
    durs = (
        _duration_hours(df["TCC_duration"])
        if "TCC_duration" in df.columns
        else [0.0] * df.height
    )
    pcol = analysis.power_col
    pw = (
        [float(x) if x is not None else 0.0 for x in df[pcol].to_list()]
        if pcol in df.columns
        else [0.0] * df.height
    )
    event_idx = en62552.defrost_event_start_indices(is_defrost)
    tcrt, elapsed, _ = _extract_defrost_intervals(df)
    rows = []
    for i, a in enumerate(event_idx):
        b = event_idx[i + 1] if i + 1 < len(event_idx) else len(is_defrost)
        last = a
        for k in range(a, b):
            if is_defrost[k]:
                last = k
            else:
                break
        energy = sum(pw[k] * durs[k] for k in range(a, last + 1))
        heater_h = sum(durs[k] for k in range(a, last + 1))
        df1 = _match_df1(df1_details or [], str(starts[a]))
        rows.append(
            {
                "event": i + 1,
                "tcc_start": int(ids[a]),
                "tcc_end": int(ids[last]),
                "start": str(starts[a]),
                "end": str(ends[last]),
                "heater_h": heater_h,
                "energy_wh": energy,
                "delta_edf_wh": None if df1 is None else df1.get("Delta_E_dfj_wh"),
                "interval_to_next_h": elapsed[i] if i < len(elapsed) else None,
                "tcrt_to_next_h": tcrt[i] if i < len(tcrt) else None,
            }
        )
    return rows


def _gap_row(gap: dict) -> dict:
    duur = float(gap.get("duur_s") or 0)
    hours, rem = divmod(duur, 3600.0)
    minutes, seconds = divmod(rem, 60.0)
    if hours >= 1:
        duur_txt = f"{int(hours)} h {int(minutes):02d} min"
    elif minutes >= 1:
        duur_txt = f"{int(minutes)} min {seconds:.0f} s"
    else:
        duur_txt = f"{seconds:.1f} s"
    row = {
        "Bron": gap.get("bron") or "",
        "Einde": str(gap.get("einde")),
        "Start": str(gap.get("start")),
        "Duur": duur_txt,
        "Duur (s)": round(duur, 1),
    }
    if gap.get("na_bestand"):
        row["Na bestand"] = gap["na_bestand"]
    if gap.get("voor_bestand"):
        row["Voor bestand"] = gap["voor_bestand"]
    return row


def _power_gaps_for_files(paths: list[str], df=None) -> list[dict]:
    files = [p for p in paths if Path(p).is_file()]
    gaps: list[dict] = []
    if files:
        try:
            gaps.extend(analysis.power_file_gaps(files))
        except Exception:
            pass
    if df is not None and analysis.time_col in df.columns:
        try:
            times = df[analysis.time_col].sort().to_list()
            gaps.extend(analysis.datetime_series_gaps(times))
        except Exception:
            pass
    return gaps


def _duration_hours(col: pl.Series) -> list[float]:
    try:
        return [float(x) if x is not None else 0.0 for x in col.dt.total_hours(fractional=True).to_list()]
    except Exception:
        hours: list[float] = []
        for v in col.to_list():
            if v is None:
                hours.append(0.0)
            elif hasattr(v, "total_seconds"):
                hours.append(v.total_seconds() / 3600.0)
            else:
                hours.append(float(v))
        return hours


def _correct_pss(
    pss1: float,
    mean_amb: float | None,
    temps: dict[str, float],
    nominal: float,
) -> float:
    """ISO 16358 / Annex B Tamb-correctie, zelfde als analyze_steady_state."""
    comps = analysis.compartments
    if not comps or mean_amb is None:
        return pss1
    teller = 0.0
    noemer = 0.0
    for name, spec in comps.items():
        den = analysis.c1 * (18 + spec["target_temp"]) + analysis.c2
        teller += spec["volume"] / den
        if name in temps:
            noemer += spec["volume"] * (mean_amb - temps[name]) / den
    if abs(noemer) < 1e-12:
        return pss1
    return pss1 * (
        1
        + (
            (nominal - mean_amb)
            / (1 + (nominal - mean_amb) * analysis.delta_cop)
        )
        * teller
        / noemer
    )


def _tcc_stats(df: pl.DataFrame) -> dict | None:
    if df is None or df.height == 0:
        return None
    durs = _duration_hours(df["TCC_duration"])
    ons = (
        _duration_hours(df["TCC_on_duration"])
        if "TCC_on_duration" in df.columns
        else [0.0] * df.height
    )
    pw = [float(x) if x is not None else 0.0 for x in df[analysis.power_col].to_list()]
    dur = sum(durs)
    energy = sum(p * dt for p, dt in zip(pw, durs))
    on_h = sum(ons)
    temps: dict[str, float] = {}
    for col in ("temp_mean_amb", *analysis.mean_temp_cols):
        if col not in df.columns or dur <= 0:
            continue
        vals = df[col].to_list()
        acc = 0.0
        w = 0.0
        for v, dt in zip(vals, durs):
            if v is None:
                continue
            acc += float(v) * dt
            w += dt
        if w:
            temps[col] = acc / w
    ids = [int(x) for x in df["TCC_id"].to_list()]
    t0 = df["TCC_start_datetime"][0]
    if "TCC_end_datetime" in df.columns:
        t1 = df["TCC_end_datetime"][-1]
    else:
        t1 = df["TCC_start_datetime"][-1]
    return {
        "n_tcc": df.height,
        "tcc_id_start": ids[0],
        "tcc_id_end": ids[-1],
        "start": str(t0),
        "end": str(t1),
        "duration_h": dur,
        "energy_wh": energy,
        "p_mean_w": energy / dur if dur else None,
        "on_h": on_h,
        "crt_pct": 100.0 * on_h / dur if dur else None,
        "temps": temps,
        "mean_amb_c": temps.get("temp_mean_amb"),
    }


def _b42_checks(x: dict, y: dict) -> list[dict]:
    px = float(x["p_mean_w"] or 0)
    py = float(y["p_mean_w"] or 0)
    p_avg = (px + py) / 2.0
    spread_w = abs(px - py)
    spread_pct = 100.0 * spread_w / p_avg if p_avg else None
    limit_w = en62552.ss2_power_spread_limit_w(p_avg)
    ratio = x["duration_h"] / y["duration_h"] if y["duration_h"] else None
    checks = [
        {
            "criterium": "B.4.2 n TCC (X = Y, n ≥ 4)",
            "waarde": f"nX={x['n_tcc']}, nY={y['n_tcc']}",
            "limiet": "nX = nY ≥ 4",
            "ok": x["n_tcc"] == y["n_tcc"] and x["n_tcc"] >= 4,
        },
        {
            "criterium": "B.4.2 duur X en Y ≥ 4 h",
            "waarde": f"X={x['duration_h']:.3f} h, Y={y['duration_h']:.3f} h",
            "limiet": "≥ 4 h",
            "ok": x["duration_h"] >= 4 and y["duration_h"] >= 4,
        },
        {
            "criterium": "B.4.2 duur X/Y",
            "waarde": f"{ratio:.3f}" if ratio is not None else "—",
            "limiet": "0,80 … 1,25",
            "ok": ratio is not None and 0.8 <= ratio <= 1.25,
        },
        {
            "criterium": "B.4.2 vermogensspread (11)",
            "waarde": (
                f"{spread_w:.3f} W"
                + (f" ({spread_pct:.2f} %)" if spread_pct is not None else "")
            ),
            "limiet": f"< max(1 W, 2 %) = {limit_w:.3f} W",
            "ok": spread_w < limit_w,
        },
    ]
    for name in analysis.compartments:
        col = f"temp_mean_{name}"
        tx, ty = x["temps"].get(col), y["temps"].get(col)
        if tx is None or ty is None:
            checks.append(
                {
                    "criterium": f"B.4.2 tempspread {name} (9)",
                    "waarde": "ontbreekt",
                    "limiet": "< 0,5 K",
                    "ok": False,
                }
            )
            continue
        spread_t = abs(tx - ty)
        checks.append(
            {
                "criterium": f"B.4.2 tempspread {name} (9)",
                "waarde": f"{spread_t:.3f} K  (X={tx:.3f} °C, Y={ty:.3f} °C)",
                "limiet": "< 0,5 K",
                "ok": spread_t < 0.5,
            }
        )
    return checks


def _select_ss2_xy(df: pl.DataFrame, event_id, next_id, last_defrost_id, interval_h: float):
    max_xy_h = max(8.0, 0.5 * interval_h)
    pool_x = df.filter(
        (pl.col("TCC_id") < event_id) & (~pl.col("is_defrost_cycle"))
    ).sort("TCC_id")
    pool_y = df.filter(
        (pl.col("TCC_id") < next_id)
        & (pl.col("TCC_id") > last_defrost_id)
        & (~pl.col("is_defrost_cycle"))
    ).sort("TCC_id")
    last_try = None
    n_max = min(pool_x.height, pool_y.height)
    for n in range(4, n_max + 1):
        x = _tcc_stats(pool_x.tail(n))
        y = _tcc_stats(pool_y.tail(n))
        if not x or not y:
            continue
        if x["duration_h"] > max_xy_h or y["duration_h"] > max_xy_h:
            break
        checks = _b42_checks(x, y)
        last_try = (x, y, checks, n, max_xy_h)
        core_ok = all(c["ok"] for c in checks)
        if core_ok:
            return x, y, checks, n, max_xy_h, True
    if last_try:
        x, y, checks, n, max_xy_h = last_try
        return x, y, checks, n, max_xy_h, False
    return None, None, [], None, max_xy_h, False


def _match_df1(details: list[dict], start_str: str) -> dict | None:
    if not details:
        return None
    for d in details:
        if str(d.get("defrost_start")) == str(start_str):
            return d
    start_prefix = str(start_str)[:19]
    for d in details:
        if str(d.get("defrost_start", ""))[:19] == start_prefix:
            return d
    return None


def _compute_ss2(
    df_means: pl.DataFrame,
    nominal_ambient: float,
    df1_details: list[dict] | None = None,
    delta_edf_wh: float = 0.0,
) -> dict | None:
    """SS2 volgens B.4.3 plus de hele-cycluscontrole."""
    needed = {
        "is_defrost_cycle",
        "TCC_id",
        "TCC_start_datetime",
        "TCC_duration",
        analysis.power_col,
    }
    if not needed.issubset(set(df_means.columns)):
        return None
    df = df_means.sort("TCC_id")
    extra_cols = [
        c
        for c in ("temp_mean_amb", *analysis.mean_temp_cols)
        if c in df.columns
    ]
    extra_means = {c: df[c].to_list() for c in extra_cols}
    on_hours = (
        _duration_hours(df["TCC_on_duration"])
        if "TCC_on_duration" in df.columns
        else [0.0] * df.height
    )
    raw = en62552.compute_ss2_cycle_energy(
        [bool(x) for x in df["is_defrost_cycle"].to_list()],
        df["TCC_id"].to_list(),
        df["TCC_start_datetime"].to_list(),
        _duration_hours(df["TCC_duration"]),
        [float(x) if x is not None else 0.0 for x in df[analysis.power_col].to_list()],
        extra_means,
    )
    if not raw:
        return None

    is_defrost = [bool(x) for x in df["is_defrost_cycle"].to_list()]
    tcc_ids = df["TCC_id"].to_list()
    events = en62552.defrost_event_start_indices(is_defrost)
    last_defrost_ids = []
    for ev_i, start_i in enumerate(events):
        end_i = events[ev_i + 1] if ev_i + 1 < len(events) else len(is_defrost)
        last_i = start_i
        for k in range(start_i, end_i):
            if is_defrost[k]:
                last_i = k
            else:
                break
        last_defrost_ids.append(tcc_ids[last_i])

    b43_intervals = []
    for iv, (a, b) in enumerate(zip(events, events[1:])):
        energy = raw["intervals"][iv]["energy_wh"]
        dt_h = raw["intervals"][iv]["duration_h"]
        id0, id1 = tcc_ids[a], tcc_ids[b]
        on_xy = sum(
            float(h)
            for tid, h in zip(tcc_ids, on_hours)
            if tid is not None and id0 <= tid < id1
        )
        x, y, checks, n_xy, max_xy_h, xy_ok = _select_ss2_xy(
            df, id0, id1, last_defrost_ids[iv], dt_h
        )
        df1 = _match_df1(df1_details or [], raw["intervals"][iv]["start"])
        d_edf = float(df1["Delta_E_dfj_wh"]) if df1 else float(delta_edf_wh or 0)
        d_tdr = float(df1["delta_tdr_h"]) if df1 else 0.0
        df1_ok = df1 is not None
        checks = list(checks)
        checks.append(
            {
                "criterium": "B.4.2 begin-defrost geldig DF1 (Annex C)",
                "waarde": "ja" if df1_ok else "nee",
                "limiet": "geldige D/F + ΔEdf",
                "ok": df1_ok,
            }
        )
        pss2 = None
        crt_ss2 = None
        temps_ss2: dict[str, float] = {}
        formula12 = None
        formula14 = None
        formula13 = []
        if dt_h > 0:
            pss2 = en62552.ss2_pss2_w(energy, d_edf, dt_h)
            crt_ss2 = en62552.ss2_crt_pct(on_xy, d_tdr, dt_h)
            formula12 = {
                "E_endY_minus_E_endX_wh": energy,
                "delta_edf_wh": d_edf,
                "t_endY_minus_t_endX_h": dt_h,
                "PSS2_w": pss2,
                "tekst": (
                    f"PSS2 = ({energy:.4f} − {d_edf:.4f}) / {dt_h:.4f} "
                    f"= {pss2:.4f} W"
                ),
            }
            formula14 = {
                "Rt_endY_minus_Rt_endX_h": on_xy,
                "delta_tdr_h": d_tdr,
                "t_endY_minus_t_endX_h": dt_h,
                "CRtSS2_pct": crt_ss2,
                "tekst": (
                    f"CRtSS2 = 100 × ({on_xy:.4f} − {d_tdr:.4f}) / {dt_h:.4f} "
                    f"= {crt_ss2:.2f} %"
                ),
            }
            for name in analysis.compartments:
                col = f"temp_mean_{name}"
                tx = (x or {}).get("temps", {}).get(col)
                ty = (y or {}).get("temps", {}).get(col)
                dth = (df1 or {}).get("dth_kh", {}).get(col, 0.0) if df1 else 0.0
                if tx is None or ty is None:
                    continue
                tss = en62552.ss2_temp_c(tx, ty, float(dth or 0), dt_h)
                temps_ss2[name] = tss
                formula13.append(
                    {
                        "compartiment": name,
                        "T_endX_C": tx,
                        "T_endY_C": ty,
                        "dTh_df_Kh": dth,
                        "TSS2_C": tss,
                        "tekst": (
                            f"TSS2,{name} = ({tx:.4f} + {ty:.4f})/2 − "
                            f"{float(dth or 0):.4f}/{dt_h:.4f} = {tss:.4f} °C"
                        ),
                    }
                )
        tamb = None
        if x and y:
            ax_ = x.get("mean_amb_c")
            ay_ = y.get("mean_amb_c")
            if ax_ is not None and ay_ is not None:
                tamb = (ax_ + ay_) / 2.0
        elif raw["intervals"][iv].get("mean_amb_c") is not None:
            tamb = raw["intervals"][iv]["mean_amb_c"]
        pss_b5 = None
        if pss2 is not None:
            pss_b5 = _correct_pss(pss2, tamb, temps_ss2, nominal_ambient)
        b43_ok = xy_ok and df1_ok and pss2 is not None
        y_adjacent = True
        b43_intervals.append(
            {
                **raw["intervals"][iv],
                "on_h": on_xy,
                "period_X": x,
                "period_Y": y,
                "n_xy": n_xy,
                "max_xy_h": max_xy_h,
                "checks": checks,
                "xy_ok": xy_ok,
                "df1_ok": df1_ok,
                "df1": df1,
                "b43_ok": b43_ok,
                "y_adjacent_to_next_defrost": y_adjacent,
                "PSS2_w": pss2,
                "pss_w": pss_b5,
                "CRtSS2_pct": crt_ss2,
                "temps_ss2": temps_ss2,
                "formula_12": formula12,
                "formula_13": formula13,
                "formula_14": formula14,
            }
        )

    temps_named = {
        name: raw["temps"][f"temp_mean_{name}"]
        for name in analysis.compartments
        if raw["temps"].get(f"temp_mean_{name}") is not None
    }
    pss_cycle = _correct_pss(
        raw["p_mean_w"],
        raw.get("mean_amb_c"),
        temps_named,
        nominal_ambient,
    )
    raw["pss_w"] = pss_cycle
    raw["edaily_kwh"] = 24.0 * pss_cycle / 1000.0
    raw["temps_c"] = temps_named
    valid_b43 = [iv for iv in b43_intervals if iv.get("b43_ok") and iv.get("pss_w") is not None]
    if valid_b43:
        tot_t = sum(iv["duration_h"] for iv in valid_b43)
        pss2_mean = sum(iv["PSS2_w"] * iv["duration_h"] for iv in valid_b43) / tot_t
        pss_b5_mean = sum(iv["pss_w"] * iv["duration_h"] for iv in valid_b43) / tot_t
        raw["b43_pss2_w"] = pss2_mean
        raw["b43_pss_w"] = pss_b5_mean
        raw["b43_edaily_kwh"] = 24.0 * pss_b5_mean / 1000.0
    raw["b43_intervals"] = b43_intervals
    raw["b43_n_valid"] = len(valid_b43)
    raw["nominal_ambient"] = nominal_ambient
    raw["delta_cop"] = analysis.delta_cop
    return raw


def _temp_summary_line(tamb: float | None, temps: dict | None) -> str:
    parts: list[str] = []
    if tamb is not None:
        parts.append(f"Tamb={float(tamb):.2f} °C")
    for name, val in (temps or {}).items():
        if val is None:
            continue
        parts.append(f"{name}={float(val):.2f} °C")
    return " · ".join(parts)


def campaign_defrost_classification() -> en62552.DefrostClassification:
    tcrt: list[float] = []
    elapsed: list[float] = []
    for test in st.session_state["tests"]:
        tcrt.extend(test.get("tcrt_hours") or [])
        elapsed.extend(test.get("elapsed_hours") or [])
    return en62552.classify_defrost_controller(tcrt, elapsed)


def _tdf_for_point(crt_ss_pct: float, delta_edf_wh: float, clf: en62552.DefrostClassification) -> float | None:
    if not delta_edf_wh:
        return None
    if clf.kind == "elapsed_time" and clf.mean_elapsed_h:
        return round(float(clf.mean_elapsed_h), 1)
    if clf.mean_trt_h:
        try:
            return en62552.defrost_interval_crt(float(clf.mean_trt_h), crt_ss_pct)
        except ValueError:
            return None
    return None


def collected_points() -> list[en62552.TestPoint]:
    clf = campaign_defrost_classification()
    points = []
    for test in st.session_state["tests"]:
        p = test.get("point")
        if p is None:
            continue
        data = {k: v for k, v in p.items() if k in _TESTPOINT_FIELDS}
        data["vs_target"] = test.get("vs_target") or data.get("vs_target") or "onder"
        point = en62552.TestPoint(**data)
        point = replace(
            point,
            delta_tdf_h=_tdf_for_point(point.crt_ss_pct, point.delta_edf_wh, clf),
        )
        points.append(point)
    return points


def run_full_test(test: dict) -> None:
    _clear_analysis()
    apply_appliance_config()
    temp_path = _resolve_temp_path(test.get("temp_file") or "")
    if not Path(temp_path).is_file():
        raise FileNotFoundError(f"Temperatuurbestand niet gevonden: {temp_path}")
    pool = _power_pool(test)
    if not pool:
        raise FileNotFoundError(
            "Geen power-bestanden gevonden. Zet alle powermetingen in één map en kies die map."
        )
    _sync_power_selection(test)
    if test.get("power_select_error"):
        raise ValueError(test["power_select_error"])
    matches = test.get("power_selected") or []
    if not matches:
        raise FileNotFoundError(
            "Geen power-bestanden die bij de tijdspanne van de temperatuurmeting passen."
        )
    power_arg = matches

    lf = _run_with_log(lambda: analysis.load_and_align_data(power_arg, temp_path))
    df = lf.collect()
    st.session_state["lf_meas_data"] = df.lazy()
    st.session_state["load_info"] = {
        "n_rows": df.height,
        "tmin": df[analysis.time_col].min(),
        "tmax": df[analysis.time_col].max(),
        "preview": df.head(25).to_pandas(),
    }

    df_TCC, df_means, df_meas, plot_data = _run_with_log(
        lambda: analysis.detect_tcc_cycles(
            df.lazy(),
            make_plots=False,
            defrost_power_w=float(st.session_state.get("defrost_power_w", 50.0)),
        )
    )
    st.session_state["df_TCC"] = df_TCC
    st.session_state["df_TCC_w_streaks_and_means"] = df_means
    st.session_state["df_meas_data_w_smooth_w_TCC_id"] = df_meas
    st.session_state["plot_data"] = plot_data

    try:
        ss1_result = _run_with_log(
            lambda: analysis.analyze_steady_state(
                df_means, nominal_ambient=float(test["ambient_c"])
            )
        )
    except analysis.SS1NotFoundError as exc:
        st.session_state["ss1_diagnostics"] = exc.diagnostics
        st.session_state["ss1_diag_summary"] = exc.summary
        st.session_state["df_all_possible_periods_w_stats"] = exc.diagnostics
        tcrt, elapsed, n_events = _extract_defrost_intervals(df_means)
        test["tcrt_hours"] = tcrt
        test["elapsed_hours"] = elapsed
        test["n_defrost_events"] = n_events
        test["defrost_events"] = _defrost_event_records(df_means)
        test["power_gaps"] = _power_gaps_for_files(matches, df)
        raise
    if len(ss1_result) == 5:
        ss1_opti, pss, df_periods, mean_amb_temp, ss1_candidates = ss1_result
    else:
        ss1_opti, pss, df_periods, mean_amb_temp = ss1_result
        ss1_candidates = None
    st.session_state["ss1_opti"] = ss1_opti
    st.session_state["pss"] = pss
    st.session_state["df_all_possible_periods_w_stats"] = df_periods
    st.session_state["mean_amb_temp"] = mean_amb_temp
    st.session_state["ss1_candidates"] = ss1_candidates
    try:
        st.session_state["ss1_diagnostics"] = analysis.annotate_ss1_periods(df_periods)
        st.session_state["ss1_diag_summary"] = analysis.ss1_diagnostics_summary(
            st.session_state["ss1_diagnostics"]
        )
    except Exception:
        st.session_state["ss1_diagnostics"] = None

    defrost_result = _run_with_log(
        lambda: analysis.analyze_defrost(df_means, df_periods, df_meas)
    )
    if len(defrost_result) == 3:
        delta_e, delta_t_df, df1_details = defrost_result
    else:
        delta_e, delta_t_df = defrost_result
        df1_details = []
    st.session_state["Delta_E_dfj_wh"] = delta_e
    st.session_state["delta_t_df"] = delta_t_df
    st.session_state["df1_details"] = df1_details

    temps = {
        name: float(ss1_opti[f"temp_mean_{name}"])
        for name in analysis.compartments.keys()
    }
    crt_ss = float(ss1_opti.get("period_steady_state_compressor_run_time") or 0.0)
    if crt_ss <= 0:
        crt_ss = (
            df_means.select(pl.col("TCC_on_duration") / pl.col("TCC_duration"))
            .mean()
            .item()
            * 100
        )

    tcrt, elapsed, n_events = _extract_defrost_intervals(df_means)
    test["tcrt_hours"] = tcrt
    test["elapsed_hours"] = elapsed
    test["n_defrost_events"] = n_events
    test["defrost_events"] = _defrost_event_records(df_means, df1_details)
    test["power_gaps"] = _power_gaps_for_files(matches, df)
    ss2 = (
        _compute_ss2(
            df_means,
            float(test["ambient_c"]),
            df1_details=df1_details,
            delta_edf_wh=float(delta_e or 0),
        )
        if n_events >= 2
        else None
    )
    test["ss2"] = ss2
    st.session_state["ss2"] = ss2

    tdh = 0.0
    tdr = 0.0
    local_clf = en62552.classify_defrost_controller(tcrt, elapsed)
    tdf = _tdf_for_point(float(crt_ss), float(delta_e or 0), local_clf)
    if tdf is None or not delta_e:
        tdf_used = None
        delta_e_used = 0.0
    else:
        tdf_used = tdf
        delta_e_used = float(delta_e)

    vs_target = test.get("vs_target") or "onder"
    warning = en62552.check_vs_target_label(temps, targets_from_config(), vs_target)
    test["vs_target_warning"] = warning

    point = en62552.TestPoint(
        name=test["name"],
        ambient_c=float(test["ambient_c"]),
        vs_target=vs_target,
        pss_w=float(pss),
        temps_c=temps,
        crt_ss_pct=float(crt_ss),
        delta_edf_wh=delta_e_used,
        delta_tdf_h=tdf_used,
        delta_tdr_h=tdr,
        mean_amb_c=float(mean_amb_temp),
    )
    test["point"] = {
        "name": point.name,
        "ambient_c": point.ambient_c,
        "vs_target": point.vs_target,
        "pss_w": point.pss_w,
        "temps_c": point.temps_c,
        "crt_ss_pct": point.crt_ss_pct,
        "delta_edf_wh": point.delta_edf_wh,
        "delta_tdf_h": point.delta_tdf_h,
        "delta_tdr_h": point.delta_tdr_h,
        "mean_amb_c": point.mean_amb_c,
    }


st.set_page_config(page_title="Koelkastanalyse EN 62552-3", layout="wide")
_init_state()

st.title("Koelkastanalyse — EN 62552-3")
st.caption(
    "Zonder ijsbereiding. Edaily bij target via interpolatie van een meting kouder én "
    "een meting warmer dan de doeltemperatuur. Ontdooi-interval uit de metingen (CRT). "
    "Volumes voer je zelf in. Elke meting mag op een eigen locatie staan."
)

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)
settings_error = st.session_state.pop("settings_error", None)
if settings_error:
    st.error(f"Instellingen laden mislukt: {settings_error}")

tab_need, tab_appliance, tab_tests, tab_view, tab_analyse, tab_defrost, tab_combine, tab_log = st.tabs(
    [
        "Wat is nodig",
        "Toestel",
        "Meetpunten",
        "Bekijk meting",
        "Analyse één meting",
        "Defrost",
        "Interpolatie & jaarverbruik",
        "Log",
    ]
)

with tab_need:
    st.subheader("Stappen volgens EN 62552-3")
    st.markdown(
        """
1. **Toestel vastleggen** — aantal en soort compartimenten in de tabel (type Tabel 1, **volume**, sensoren). Rijen toevoegen of wijzigen mag.
2. **Twee omgevingen meten** — 16 °C én 32 °C (geen ijsbereiding, geen load-processing).
3. **Per omgeving twee instellingen** — de thermostaat is vaak een draaiknop zonder schaal.
   Kies alleen of de knop **kouder** of **warmer** staat dan de doeltemperatuur.
   De energie bij target volgt altijd uit **interpolatie** tussen die twee metingen (Annex E.3; geen extrapolatie, ΔT ≤ 4 K).
4. **Per meting de energiecomponenten bepalen**:
   - steady-state vermogen **Pss** (Annex B / SS1)
   - extra ontdooi-energie **ΔEdf** (Annex C)
   - compressor-run-time tussen ontdooiingen **Δtrt uit de metingen** (geen handmatige invoer)
   - ontdooi-interval **Δtdf** (Annex D.3, formule 26) als het een CRT-regelaar lijkt
   - dagverbruik **Edaily = 24·Pss + 24·ΔEdf/Δtdf** (formule 2)
5. **Per omgeving interpoleren** naar de targets → Edaily,16 en Edaily,32.
6. **Jaarverbruik (Europa)**  
   E_total = 182,5 × Edaily,16 + 182,5 × Edaily,32  (kWh/jaar, zonder Eaux).
        """
    )
    st.subheader("Status van deze campagne")
    try:
        apply_appliance_config()
    except Exception as exc:
        st.error(f"{exc} — {getattr(analysis, '__file__', analysis)}")
    points = collected_points()
    targets = targets_from_config()
    for item in en62552.campaign_status(points, targets):
        if item["ok"]:
            st.success(item["text"])
        else:
            st.warning(f"{item['text']}" + (f" — {item['detail']}" if item["detail"] else ""))

    st.info(
        "Annex E.4 (triangulatie bij twee onafhankelijke thermostaten) en Annex G "
        "(load processing) zitten nog niet in deze versie."
    )

with tab_appliance:
    st.subheader("Compartimenten")
    st.caption(
        "Wijzig naam, type en volume, of voeg rijen toe. Type volgt Tabel 1 van de norm. "
        "Volume is invoer (Annex H meet je niet hier)."
    )
    add_col, del_col, _ = st.columns([1, 1, 2])
    if add_col.button("Compartiment toevoegen"):
        df = _as_comp_df(st.session_state["compartments_df"])
        extra = pd.DataFrame(
            [
                {
                    "name": f"compartiment {len(df) + 1}",
                    "table_type": "Fresh food",
                    "volume": 0.0,
                    "sensors": "",
                }
            ]
        )
        st.session_state["compartments_df"] = pd.concat([df, extra], ignore_index=True)
        st.rerun()
    if del_col.button("Laatste rij verwijderen"):
        df = _as_comp_df(st.session_state["compartments_df"])
        if len(df) > 1:
            st.session_state["compartments_df"] = df.iloc[:-1].reset_index(drop=True)
            st.rerun()
        else:
            st.warning("Er moet minstens één compartiment blijven.")

    edited = st.data_editor(
        _as_comp_df(st.session_state["compartments_df"]),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Naam", required=True, width="medium"),
            "table_type": st.column_config.SelectboxColumn(
                "Soort (Tabel 1)",
                options=list(en62552.COMPARTMENT_TARGETS_C.keys()),
                required=True,
                width="medium",
            ),
            "volume": st.column_config.NumberColumn(
                "Volume (l)", min_value=0.0, step=1.0, format="%.1f"
            ),
            "sensors": st.column_config.TextColumn("Sensors (komma-gescheiden)", width="large"),
        },
    )
    st.session_state["compartments_df"] = edited
    st.session_state["compartments_ui"] = _compartment_records(edited)
    st.caption(
        f"{len(st.session_state['compartments_ui'])} compartiment(en). "
        "Gebruik + onder de tabel of de knop hierboven om een rij toe te voegen."
    )

    st.text_input("Omgevingssensoren", key="ambient_sensors")
    st.text_input("Gradiëntsensoren", key="gradient_sensors")

    st.subheader("Ontdooiregelaar")
    st.info(
        "De compressor-run-time tot de volgende ontdooiing (Δtrt) wordt uit de metingen "
        "bepaald: som van de compressor-aan-tijd tussen opeenvolgende ontdooi-events. "
        "Als de CV van Δtrt ≥ 10 % waarschuwt het programma dat het geen CRT-regelaar lijkt "
        "(elapsed-time of variable/demand-defrost)."
    )
    st.number_input(
        "Ontdooi als vermogen P groter is dan (W)",
        min_value=0.0,
        max_value=5000.0,
        step=5.0,
        format="%.1f",
        key="defrost_power_w",
        help="Een TCC telt als ontdooi-cyclus als er ergens in die cyclus P boven deze drempel komt. Standaard 50 W. Zet hoger als normale compressor-aan al boven 50 W zit.",
    )
    st.checkbox(
        "Timer loopt door tijdens ontdooiverwarming (Δtdh > 0) — nog niet gemeten, alleen ter info",
        key="heater_advances_timer",
    )
    if st.button("Toestelconfiguratie toepassen"):
        try:
            apply_appliance_config()
            st.session_state["flash"] = "Toestelconfiguratie toegepast."
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

with tab_tests:
    st.subheader("Meetpunten")
    st.caption(
        "Alle powermetingen staan in één map (bovenaan). Per meetpunt kies je alleen de "
        "temperatuurmeting — de app zoekt de bijbehorende powerdata. "
        "De exacte instelwaarde hoeft niet bekend te zijn: kouder of warmer dan target."
    )
    _render_campaign_power()
    st.divider()
    _render_settings_io()
    st.divider()
    for i, test in enumerate(st.session_state["tests"]):
        side = VS_TARGET_OPTIONS.get(test.get("vs_target", "onder"), test.get("vs_target"))
        with st.expander(
            f"{test['name']}  ({test['ambient_c']} °C, {side})",
            expanded=i == 0,
        ):
            c1, c2 = st.columns(2)
            test["name"] = c1.text_input("Naam", test["name"], key=f"n_{i}")
            test["ambient_c"] = c2.selectbox(
                "Omgeving Tamb",
                options=[16, 32],
                index=0 if int(test["ambient_c"]) == 16 else 1,
                key=f"a_{i}",
            )
            current_vs = test.get("vs_target", "onder")
            test["vs_target"] = st.radio(
                "Thermostaat t.o.v. doeltemperatuur",
                options=list(VS_TARGET_OPTIONS.keys()),
                format_func=lambda v: VS_TARGET_OPTIONS[v],
                index=0 if current_vs == "onder" else 1,
                horizontal=True,
                key=f"vs_{i}",
                help="Draaiknop zonder schaal: kouder of warmer dan target is voldoende.",
            )
            _render_temp_files(test, i)
            if test.get("vs_target_warning"):
                st.warning(test["vs_target_warning"])
            if test.get("point"):
                p = test["point"]
                try:
                    edaily = en62552.TestPoint(**{k: v for k, v in p.items() if k in _TESTPOINT_FIELDS}).edaily_kwh
                except TypeError:
                    edaily = float("nan")
                success_lines = [
                    f"**SS1:** Pss={p['pss_w']:.3f} W, Edaily={edaily:.4f} kWh/d"
                ]
                temps_line = _temp_summary_line(p.get("mean_amb_c"), p.get("temps_c"))
                if temps_line:
                    success_lines.append(f"**Gemeten:** {temps_line}")
                ss2 = test.get("ss2")
                if ss2:
                    if ss2.get("b43_pss_w") is not None:
                        success_lines.append(
                            f"**SS2 B.4.3:** PSS2={ss2['b43_pss2_w']:.3f} W, "
                            f"Pss={ss2['b43_pss_w']:.3f} W, "
                            f"24·Pss={ss2['b43_edaily_kwh']:.4f} kWh/d "
                            f"({ss2.get('b43_n_valid', 0)} geldig interval"
                            f"{'' if ss2.get('b43_n_valid') == 1 else 'len'})"
                        )
                    else:
                        success_lines.append(
                            f"**SS2:** Pss={ss2['pss_w']:.3f} W, "
                            f"Edaily={ss2['edaily_kwh']:.4f} kWh/d "
                            f"({ss2['n_intervals']} volledige ontdooicyclus"
                            f"{'' if ss2['n_intervals'] == 1 else 'sen'})"
                        )
                st.success("  \n".join(success_lines))
                n_ev = test.get("n_defrost_events", 0)
                n_int = len(test.get("tcrt_hours") or [])
                if n_ev:
                    mean_trt = (
                        sum(test["tcrt_hours"]) / n_int if n_int else None
                    )
                    st.caption(
                        f"Ontdooi-events: {n_ev}, intervallen: {n_int}"
                        + (f", Δtrt uit meting ≈ {mean_trt:.2f} h" if mean_trt else "")
                    )
            cols = st.columns(2)
            if cols[0].button("Dit meetpunt analyseren", key=f"run_{i}"):
                st.session_state["active_test"] = i
                try:
                    with st.spinner(f"Analyse van {test['name']}…"):
                        run_full_test(test)
                    st.session_state["flash"] = f"Meetpunt '{test['name']}' geanalyseerd."
                    st.rerun()
                except analysis.SS1NotFoundError as exc:
                    st.error(str(exc))
                    st.markdown("### Validatie stabiele periode (SS1)")
                    _render_ss1_decision()
                except Exception as exc:
                    st.error(str(exc))
            if cols[1].button("Verwijderen", key=f"del_{i}"):
                st.session_state["tests"].pop(i)
                st.rerun()

    if st.button("Meetpunt toevoegen"):
        n = len(st.session_state["tests"]) + 1
        st.session_state["tests"].append(
            _blank_test(f"Meting {n}", 32, "onder", test_id=f"run-{n}")
        )
        st.rerun()

with tab_view:
    st.subheader("Bekijk meting")
    st.caption(
        "Vermogen en temperaturen van de gekozen meting op de uitgelijnde tijdreeks "
        "(dezelfde samples als na load_and_align_data). Niet alleen TCC-startpunten — "
        "daardoor blijft T zichtbaar tot het einde van de overlap, ook na de laatste complete cyclus."
    )
    tests = st.session_state["tests"]
    if not tests:
        st.info("Voeg eerst een meetpunt toe in het tabblad Meetpunten.")
    else:
        idx_default = st.session_state.get("active_test", 0)
        if not (0 <= idx_default < len(tests)):
            idx_default = 0
        chosen = st.selectbox(
            "Gekozen meting",
            options=list(range(len(tests))),
            index=idx_default,
            format_func=lambda i: (
                f"{tests[i]['name']}  ({tests[i]['ambient_c']} °C, "
                f"{VS_TARGET_OPTIONS.get(tests[i].get('vs_target', 'onder'), tests[i].get('vs_target'))})"
            ),
            key="view_test_select",
        )
        st.session_state["active_test"] = chosen
        test = tests[chosen]
        try:
            apply_appliance_config()
        except Exception as exc:
            st.warning(f"Toestelconfiguratie: {exc}")

        if _view_file_key(test) is None:
            err = test.get("power_select_error")
            if err:
                st.error(err)
            else:
                st.info(
                    "Kies de power-map en een temperatuurmeting voor dit meetpunt in het tabblad Meetpunten."
                )
        else:
            try:
                with st.spinner("Meting laden en uitlijnen…"):
                    df_view = _aligned_view_df(test)
            except Exception as exc:
                st.error(str(exc))
            else:
                tmin = df_view[analysis.time_col].min()
                tmax = df_view[analysis.time_col].max()
                m1, m2, m3 = st.columns(3)
                m1.metric("Rijen (uitgelijnd)", f"{df_view.height:,}".replace(",", "."))
                m2.metric("Start", str(tmin))
                m3.metric("Einde", str(tmax))

                logger_cols = _logger_temp_columns(df_view)
                extra_cols = list(logger_cols)
                missing_comp = [
                    name
                    for name, sensors in _compartment_sensor_names().items()
                    if not any(s in df_view.columns for s in sensors)
                ]
                if missing_comp:
                    st.warning(
                        "Geen sensoren in de meting voor: " + ", ".join(missing_comp)
                    )
                st.markdown("**Compartimenttemperaturen (standaard)**")
                shown = []
                for name, sensors in _compartment_sensor_names().items():
                    present = [s for s in sensors if s in df_view.columns]
                    if present:
                        shown.append(f"{name}: {', '.join(present)}")
                st.caption(
                    "Gemiddelde van de compartimentsensoren uit de toestelconfiguratie. "
                    + (" · ".join(shown) if shown else "Geen compartimentsensoren gevonden.")
                )

                extra_selected: list[str] = []
                if extra_cols:
                    st.markdown("**Extra temperatuurkanalen**")
                    st.caption(
                        "Standaard uit. Vink individuele loggerkanalen aan "
                        "(compartiment, omgeving, gradiënt) om alle temperaturen te zien."
                    )
                    show_all_extra = st.checkbox(
                        "Toon alle extra kanalen",
                        value=False,
                        key="view_all_extra_temps",
                    )
                    if show_all_extra:
                        extra_selected = list(extra_cols)
                        st.caption(", ".join(extra_cols))
                    else:
                        boxes = st.columns(3)
                        for i, col in enumerate(extra_cols):
                            if boxes[i % 3].checkbox(col, value=False, key=f"view_temp_{col}"):
                                extra_selected.append(col)
                else:
                    st.caption("Geen extra loggerkanalen naast de compartimentsensoren.")

                fig_view = _plotly_view_measurement(df_view, extra_selected)
                st.plotly_chart(
                    fig_view,
                    width="stretch",
                    config={
                        "scrollZoom": True,
                        "displaylogo": False,
                        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                    },
                )

with tab_analyse:
    st.subheader("Laatst geanalyseerde meting")
    info = st.session_state["load_info"]
    if info is None:
        st.info("Analyseer een meetpunt via het tabblad Meetpunten.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Rijen", f"{info['n_rows']:,}".replace(",", "."))
        m2.metric("Start", str(info["tmin"]))
        m3.metric("Einde", str(info["tmax"]))
        if st.session_state["pss"] is not None:
            ss1 = st.session_state["ss1_opti"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Pss", f"{st.session_state['pss']:.3f} W")
            r2.metric("T_amb gemeten", f"{st.session_state['mean_amb_temp']:.2f} °C")
            r3.metric("SS1-duur", str(ss1["period_duration"]))
            r4.metric("ΔEdf", f"{st.session_state['Delta_E_dfj_wh']:.2f} Wh")
            tests = st.session_state["tests"]
            idx = st.session_state.get("active_test", 0)
            if 0 <= idx < len(tests):
                tcrt = tests[idx].get("tcrt_hours") or []
                elapsed = tests[idx].get("elapsed_hours") or []
                n_ev = tests[idx].get("n_defrost_events", 0)
                if tests[idx].get("vs_target_warning"):
                    st.warning(tests[idx]["vs_target_warning"])
                if n_ev == 0:
                    st.info("Geen ontdooi-cycli in deze meting; Edaily = 24·Pss.")
                else:
                    clf = en62552.classify_defrost_controller(tcrt, elapsed)
                    st.markdown("**Ontdooi-interval uit deze meting**")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Ontdooi-events", n_ev)
                    d2.metric(
                        "Δtrt (compressor)",
                        f"{clf.mean_trt_h:.2f} h" if clf.mean_trt_h else "—",
                    )
                    d3.metric(
                        "Kloktijd tussen defrost",
                        f"{clf.mean_elapsed_h:.2f} h" if clf.mean_elapsed_h else "—",
                    )
                    if clf.cv_trt is not None:
                        st.caption(
                            f"CV compressor-run-time = {clf.cv_trt:.1%} "
                            f"(limiet CRT: < {en62552.CRT_CV_LIMIT:.0%})"
                        )
                    if clf.warning:
                        st.warning(clf.warning)

            _render_ss2(tests[idx] if 0 <= idx < len(tests) else None)

        st.markdown("### Vermogen en temperatuur")
        st.caption(
            "Scroll om in te zoomen, sleep om te panen, dubbelklik om te resetten. "
            "De gekleurde banden zijn blok A/B/C van SS1. Hover toont het tijdstip en de waarde."
        )
        zoom_ss1 = st.checkbox("Start ingezoomd op SS1", value=True, key="zoom_ss1")
        fig_pt = _plotly_power_temp(st.session_state.get("ss1_opti"), zoom_ss1)
        if fig_pt is not None:
            st.plotly_chart(
                fig_pt,
                width="stretch",
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                },
            )
        else:
            st.info("Nog geen plotdata. Analyseer eerst een meetpunt.")

        st.markdown("### Validatie stabiele periode (SS1)")
        _render_ss1_decision()

        if st.session_state["plot_data"] is not None:
            with st.expander("Oude matplotlib-grafieken (histogram / cycli)"):
                fig = analysis.plot_power_and_histogram(
                    st.session_state["plot_data"]["df_meas_data_w_median"],
                    st.session_state["plot_data"]["histogram"],
                    st.session_state["plot_data"]["threshold"],
                    st.session_state["plot_data"]["base_level"],
                    st.session_state["plot_data"]["peak_level"],
                )
                _show_fig(fig)
                fig = analysis.plot_detected_cycles(
                    st.session_state["df_TCC"],
                    st.session_state["plot_data"]["df_meas_data_w_median"],
                    st.session_state["plot_data"]["threshold"],
                    ss1_opti=st.session_state["ss1_opti"],
                )
                _show_fig(fig)

with tab_defrost:
    st.subheader("Ontdooi-events")
    st.caption(
        "Overzicht van alle gedetecteerde defrosts na analyse van een meetpunt: "
        "energie tijdens de heater-TCC’s, ΔEdf (Annex C) als die is berekend, "
        "intervallen tot de volgende defrost, en het type regelaar. "
        "Onderbrekingen in de powermeting (gaten tussen bestanden of samples) staan eronder."
    )
    tests = st.session_state["tests"]
    clf = campaign_defrost_classification()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Type (campagne)", _KIND_NL.get(clf.kind, clf.kind))
    k2.metric("Δtrt gemeten", f"{clf.mean_trt_h:.2f} h" if clf.mean_trt_h else "—")
    k3.metric("Kloktijd", f"{clf.mean_elapsed_h:.2f} h" if clf.mean_elapsed_h else "—")
    k4.metric("Intervallen", str(clf.n))
    if clf.cv_trt is not None:
        st.caption(
            f"CV Δtrt = {clf.cv_trt:.1%}"
            + (f", CV kloktijd = {clf.cv_elapsed:.1%}" if clf.cv_elapsed is not None else "")
            + f"  (CRT-limiet < {en62552.CRT_CV_LIMIT:.0%})"
        )
    if clf.warning:
        st.warning(clf.warning)

    any_events = False
    campaign_gaps: list[dict] = []
    pool = _campaign_power_pool()
    if pool:
        campaign_gaps = _power_gaps_for_files(pool)

    for i, test in enumerate(tests):
        events = test.get("defrost_events") or []
        n_ev = test.get("n_defrost_events") or len(events)
        local_clf = en62552.classify_defrost_controller(
            test.get("tcrt_hours") or [],
            test.get("elapsed_hours") or [],
        )
        title = (
            f"{test['name']}  ({test['ambient_c']} °C, "
            f"{VS_TARGET_OPTIONS.get(test.get('vs_target', 'onder'), test.get('vs_target'))})"
            f" — {n_ev} ontdooiing{'en' if n_ev != 1 else ''}"
        )
        with st.expander(title, expanded=bool(events) or i == 0):
            if not events and not test.get("temp_file"):
                st.info("Nog geen temperatuurmeting gekozen.")
                continue
            if not events:
                st.info("Nog geen analyse, of geen ontdooi-events in deze meting.")
                continue
            any_events = True
            st.caption(
                f"Type in deze meting: **{_KIND_NL.get(local_clf.kind, local_clf.kind)}**"
                + (
                    f" · Δtrt ≈ {local_clf.mean_trt_h:.2f} h"
                    if local_clf.mean_trt_h
                    else ""
                )
                + (
                    f" · kloktijd ≈ {local_clf.mean_elapsed_h:.2f} h"
                    if local_clf.mean_elapsed_h
                    else ""
                )
            )
            rows = []
            soort = _KIND_NL.get(local_clf.kind, local_clf.kind)
            for ev in events:
                rows.append(
                    {
                        "Event": ev.get("event"),
                        "Start": ev.get("start"),
                        "Einde": ev.get("end"),
                        "TCC": f"{ev.get('tcc_start')}–{ev.get('tcc_end')}",
                        "Heater (h)": None
                        if ev.get("heater_h") is None
                        else round(float(ev["heater_h"]), 4),
                        "E heater (Wh)": None
                        if ev.get("energy_wh") is None
                        else round(float(ev["energy_wh"]), 2),
                        "ΔEdf (Wh)": None
                        if ev.get("delta_edf_wh") is None
                        else round(float(ev["delta_edf_wh"]), 3),
                        "Δt tot volgende (h)": None
                        if ev.get("interval_to_next_h") is None
                        else round(float(ev["interval_to_next_h"]), 3),
                        "Δtrt tot volgende (h)": None
                        if ev.get("tcrt_to_next_h") is None
                        else round(float(ev["tcrt_to_next_h"]), 3),
                        "Soort": soort
                        if ev.get("interval_to_next_h") is not None
                        else "geen volgend interval",
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
            gaps = test.get("power_gaps") or []
            file_gaps = [g for g in gaps if g.get("bron") == "bestanden"]
            sample_gaps = [g for g in gaps if g.get("bron") == "samples"]
            if file_gaps or sample_gaps:
                st.warning(
                    f"{len(file_gaps)} onderbreking(en) tussen powerbestanden, "
                    f"{len(sample_gaps)} gat(en) in de uitgelijnde samples."
                )
                st.dataframe(
                    [_gap_row(g) for g in gaps],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.success("Geen onderbrekingen in de powermeting van dit meetpunt.")

    if not any_events:
        st.info("Analyseer eerst minstens één meetpunt om defrost-events te zien.")

    st.markdown("**Onderbrekingen in de hele power-map**")
    if not pool:
        st.caption("Nog geen power-map gekozen.")
    elif not campaign_gaps:
        st.success("Geen gaten tussen de powerbestanden in de map (drempel 30 s).")
    else:
        st.warning(
            f"{len(campaign_gaps)} onderbreking(en) tussen opeenvolgende powerbestanden."
        )
        st.dataframe(
            [_gap_row(g) for g in campaign_gaps],
            hide_index=True,
            width="stretch",
        )

with tab_combine:
    st.subheader("Interpolatie per omgeving en jaarverbruik")
    points = collected_points()
    targets = targets_from_config()
    clf = campaign_defrost_classification()
    kind_nl = {
        "compressor_run_time": "compressor-run-time (Annex D.3)",
        "elapsed_time": "elapsed-time (Annex D.2) — afwijkend van CRT",
        "variable": "variable/demand-defrost — afwijkend van CRT",
        "onvoldoende_data": "nog onvoldoende intervallen om het type te toetsen",
    }
    st.markdown("**Ontdooitype uit alle metingen**")
    k1, k2, k3 = st.columns(3)
    k1.metric("Type", kind_nl.get(clf.kind, clf.kind))
    k2.metric("Δtrt gemeten", f"{clf.mean_trt_h:.2f} h" if clf.mean_trt_h else "—")
    k3.metric("Aantal intervallen", clf.n)
    if clf.cv_trt is not None:
        st.caption(
            f"CV Δtrt = {clf.cv_trt:.1%}"
            + (f", CV kloktijd = {clf.cv_elapsed:.1%}" if clf.cv_elapsed is not None else "")
        )
    if clf.warning:
        st.warning(clf.warning)
    if clf.kind == "elapsed_time":
        st.info(
            "Δtdf wordt voorlopig gelijk gesteld aan de gemeten kloktijd tussen ontdooiingen, "
            "niet aan de CRT-formule."
        )

    if not points:
        st.info("Analyseer eerst minstens één meetpunt.")
    else:
        rows = []
        for p in points:
            row = {
                "Meting": p.name,
                "Tamb °C": p.ambient_c,
                "t.o.v. target": VS_TARGET_OPTIONS.get(p.vs_target, p.vs_target),
                "Pss W": round(p.pss_w, 3),
                "Edaily kWh/d": round(p.edaily_kwh, 4),
                "CRtSS %": round(p.crt_ss_pct, 1),
                "ΔEdf Wh": round(p.delta_edf_wh, 2),
                "Δtdf h": p.delta_tdf_h,
            }
            for name, t in p.temps_c.items():
                target = targets.get(name)
                row[f"T {name}"] = round(t, 2)
                if target is not None:
                    row[f"Δ {name}"] = round(t - target, 2)
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)

        r16 = en62552.resolve_ambient_edaily(points, targets, 16)
        r32 = en62552.resolve_ambient_edaily(points, targets, 32)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**16 °C**")
            if r16.valid:
                st.metric("Edaily,16", f"{r16.edaily_wh/1000:.4f} kWh/d")
                st.caption(f"{r16.method}" + (
                    f" — interpolatie op {r16.interpolation_compartment}"
                    if r16.interpolation_compartment else ""
                ))
            else:
                st.error("; ".join(r16.messages))
            for msg in r16.messages:
                st.caption(msg)
        with c2:
            st.markdown("**32 °C**")
            if r32.valid:
                st.metric("Edaily,32", f"{r32.edaily_wh/1000:.4f} kWh/d")
                st.caption(f"{r32.method}" + (
                    f" — interpolatie op {r32.interpolation_compartment}"
                    if r32.interpolation_compartment else ""
                ))
            else:
                st.error("; ".join(r32.messages))
            for msg in r32.messages:
                st.caption(msg)

        if r16.valid and r32.valid:
            e_year = en62552.annual_energy_europe_kwh(
                r16.edaily_wh / 1000.0, r32.edaily_wh / 1000.0
            )
            st.metric("Jaarverbruik E_total (EU, zonder Eaux)", f"{e_year:.2f} kWh/jaar")
            st.caption("E_total = 182,5 × Edaily,16 + 182,5 × Edaily,32")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                ["16 °C", "32 °C"],
                [r16.edaily_wh / 1000.0, r32.edaily_wh / 1000.0],
                color=["steelblue", "darkorange"],
            )
            ax.set_ylabel("Edaily (kWh/d)")
            ax.set_title("Dagelijks energieverbruik bij de twee testomgevingen")
            _show_fig(fig)

with tab_log:
    if st.session_state["log"]:
        st.code(st.session_state["log"], language=None)
    else:
        st.caption("Nog geen log.")
