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

# Streamlit herlaadt app.py, maar niet altijd lokale imports.
# In een gebundelde exe is reload niet nodig en kan het stukgaan.
if not getattr(sys, "frozen", False):
    en62552 = reload(en62552)


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
_TESTPOINT_FIELDS = {f.name for f in fields(en62552.TestPoint)}


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
        "power_glob": analysis.power_files,
        "temp_file": analysis.temp_file,
        "point": None,
        "tcrt_hours": [],
        "elapsed_hours": [],
        "n_defrost_events": 0,
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
        t.setdefault("vs_target_warning", None)
        t.setdefault("ss2", None)
        t.pop("setpoint", None)


def _init_state() -> None:
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
    ):
        st.session_state.setdefault(key, None)


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


def _render_ss2(test: dict | None) -> None:
    ss2 = (test or {}).get("ss2") or st.session_state.get("ss2")
    n_ev = (test or {}).get("n_defrost_events", 0)
    if n_ev < 2 and not ss2:
        return
    st.markdown("**SS2 — volledige ontdooicyclus**")
    st.caption(
        "Gemiddeld vermogen van de start van ontdooi-event j tot de start van event j+1. "
        "Edaily = 24·P (ontdooi-energie al inbegrepen). Extra controle naast SS1: "
        "24·Pss + 24·ΔEdf/Δtdf."
    )
    if not ss2:
        st.info(
            "SS2 is niet berekend: er zijn minstens twee ontdooi-events nodig "
            "met een volledig interval ertussen."
        )
        return
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("P gemeten", f"{ss2['p_mean_w']:.3f} W")
    s2.metric("Pss (Tamb-correctie)", f"{ss2['pss_w']:.3f} W")
    s3.metric("Edaily SS2", f"{ss2['edaily_kwh']:.4f} kWh/d")
    s4.metric("Intervallen", ss2["n_intervals"])
    temps_line = _temp_summary_line(ss2.get("mean_amb_c"), ss2.get("temps_c"))
    if temps_line:
        st.caption(f"SS2-gemiddelden: {temps_line} · duur {ss2['duration_h']:.2f} h")
    point = (test or {}).get("point")
    if point:
        try:
            edaily_ss1 = en62552.TestPoint(
                **{k: v for k, v in point.items() if k in _TESTPOINT_FIELDS}
            ).edaily_kwh
            delta = ss2["edaily_kwh"] - edaily_ss1
            st.caption(
                f"Verschil t.o.v. SS1: {delta:+.4f} kWh/d "
                f"({100.0 * delta / edaily_ss1:+.1f} %)"
                if edaily_ss1
                else f"Verschil t.o.v. SS1: {delta:+.4f} kWh/d"
            )
        except TypeError:
            pass
    rows = []
    for i, iv in enumerate(ss2.get("intervals") or [], start=1):
        rows.append(
            {
                "Interval": i,
                "Start": iv["start"],
                "Einde": iv["end"],
                "Duur h": round(iv["duration_h"], 3),
                "E Wh": round(iv["energy_wh"], 2),
                "P W": round(iv["p_mean_w"], 3),
                "Edaily kWh/d": round(iv["edaily_kwh"], 4),
                "Tamb °C": (
                    round(iv["mean_amb_c"], 2) if iv.get("mean_amb_c") is not None else None
                ),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")


def _render_ss1_decision() -> None:
    ss1 = st.session_state.get("ss1_opti")
    if not ss1:
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
Van die kandidaten wint de **kleinste n**, daarna de kortste duur, daarna de vroegste start.

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
        "De tabel hieronder is die set, gesorteerd zoals de keuze (kleinste n eerst)."
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
        target.write_bytes(uploaded.getbuffer())
        saved.append(target)
    return saved


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


def _power_matches(test: dict) -> list[str]:
    paths = test.get("power_paths")
    if paths:
        return [p for p in paths if Path(p).is_file()]
    pattern = test.get("power_glob") or ""
    if not pattern:
        return []
    found = glob.glob(pattern)
    if not found and not Path(pattern).is_absolute():
        found = glob.glob(str(PROJECT_DIR / pattern))
    return found


def _power_arg(test: dict) -> str | list[str]:
    matches = _power_matches(test)
    return matches if matches else (test.get("power_glob") or "")


def _apply_power_paths(test: dict, paths: list[str], index: int | None = None) -> None:
    files = [str(Path(p)) for p in paths if Path(p).is_file()]
    test["power_paths"] = files
    if files:
        folder = Path(files[0]).parent
        test["power_glob"] = str(folder / "*")
        if index is not None:
            st.session_state[f"p_{index}"] = test["power_glob"]


def _render_test_files(test: dict, index: int) -> None:
    local = _is_local_client()
    if local:
        mode = st.radio(
            "Bestanden aanleveren",
            options=["schijf", "upload"],
            format_func=lambda v: (
                "Selecteren op deze pc" if v == "schijf" else "Uploaden (browser)"
            ),
            horizontal=True,
            key=f"filemode_{index}",
            help="Op een webserver: uploaden. Lokaal: direct een pad op je schijf kiezen, zonder kopiëren.",
        )
    else:
        mode = "upload"
        st.caption("Deze sessie is op afstand: kies de CSV’s via uploaden.")

    if mode == "upload":
        power_up = st.file_uploader(
            "Power-CSV’s",
            type=["csv", "txt", "dat"],
            accept_multiple_files=True,
            key=f"power_up_{index}",
            help="Eén of meer vermogensbestanden van de logger.",
        )
        if power_up:
            sig = tuple((f.name, f.size) for f in power_up)
            if test.get("_power_upload_sig") != sig:
                saved = _save_uploaded(
                    power_up, UPLOAD_ROOT / _safe_id(test["id"]) / "power"
                )
                _apply_power_paths(test, [str(p) for p in saved], index)
                test["_power_upload_sig"] = sig
        temp_up = st.file_uploader(
            "Temperatuur-CSV",
            type=["csv", "txt", "dat"],
            accept_multiple_files=False,
            key=f"temp_up_{index}",
        )
        if temp_up is not None:
            sig = (temp_up.name, temp_up.size)
            if test.get("_temp_upload_sig") != sig:
                saved = _save_uploaded(
                    [temp_up], UPLOAD_ROOT / _safe_id(test["id"]) / "temp"
                )
                if saved:
                    test["temp_file"] = str(saved[0])
                    st.session_state[f"t_{index}"] = test["temp_file"]
                    test["_temp_upload_sig"] = sig
    else:
        p1, p2 = st.columns(2)
        if p1.button("Power-bestanden kiezen…", key=f"pick_p_{index}"):
            chosen = _pick_local_files(
                multiple=True, title="Power-bestanden (CSV)"
            )
            if chosen:
                _apply_power_paths(test, chosen, index)
                test.pop("_power_upload_sig", None)
                st.rerun()
        if p2.button("Power-map kiezen…", key=f"pick_pd_{index}"):
            folder = _pick_local_directory("Map met power-bestanden")
            if folder:
                csvs = sorted(Path(folder).glob("*.csv"))
                files = [str(p) for p in csvs] or [
                    str(p) for p in Path(folder).iterdir() if p.is_file()
                ]
                if files:
                    _apply_power_paths(test, files, index)
                    test.pop("_power_upload_sig", None)
                    st.rerun()
                else:
                    st.warning("In die map staan geen bestanden.")
        if st.button("Temperatuur-CSV kiezen…", key=f"pick_t_{index}"):
            chosen = _pick_local_files(
                multiple=False, title="Temperatuur-CSV"
            )
            if chosen:
                test["temp_file"] = chosen[0]
                st.session_state[f"t_{index}"] = chosen[0]
                test.pop("_temp_upload_sig", None)
                st.rerun()

    matches = _power_matches(test)
    if matches:
        st.caption(f"{len(matches)} power-bestand(en): " + ", ".join(Path(p).name for p in matches[:8])
                   + ("…" if len(matches) > 8 else ""))
    else:
        st.caption("Nog geen power-bestanden gekozen.")
    temp_ok = Path(_resolve_temp_path(test.get("temp_file") or "")).is_file() if test.get("temp_file") else False
    if temp_ok:
        st.caption(f"Temperatuur: {Path(test['temp_file']).name}")
    else:
        st.caption("Nog geen temperatuurbestand gekozen.")

    with st.expander("Handmatig pad (optioneel)"):
        new_glob = st.text_input(
            "Power-bestanden (glob of map\\*)",
            test.get("power_glob") or "",
            key=f"p_{index}",
        )
        if new_glob != (test.get("power_glob") or ""):
            test.pop("power_paths", None)
        test["power_glob"] = new_glob
        test["temp_file"] = st.text_input(
            "Temperatuur-CSV",
            test.get("temp_file") or "",
            key=f"t_{index}",
        )


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
    analysis.configure_appliance(new, ambient_sensors=amb, gradient_sensors=grad)
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


def _compute_ss2(df_means: pl.DataFrame, nominal_ambient: float) -> dict | None:
    """SS2 uit complete ontdooi-tot-ontdooi-intervallen (minstens twee events)."""
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
    temps_named = {
        name: raw["temps"][f"temp_mean_{name}"]
        for name in analysis.compartments
        if raw["temps"].get(f"temp_mean_{name}") is not None
    }
    pss = _correct_pss(
        raw["p_mean_w"],
        raw.get("mean_amb_c"),
        temps_named,
        nominal_ambient,
    )
    raw["pss_w"] = pss
    raw["edaily_kwh"] = 24.0 * pss / 1000.0
    raw["temps_c"] = temps_named
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
    power_arg = _power_arg(test)
    temp_path = _resolve_temp_path(test.get("temp_file") or "")
    if not Path(temp_path).is_file():
        raise FileNotFoundError(f"Temperatuurbestand niet gevonden: {temp_path}")
    matches = power_arg if isinstance(power_arg, list) else glob.glob(str(power_arg))
    if not matches:
        raise FileNotFoundError("Geen power-bestanden gevonden. Upload of selecteer CSV’s.")

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
        lambda: analysis.detect_tcc_cycles(df.lazy(), make_plots=False)
    )
    st.session_state["df_TCC"] = df_TCC
    st.session_state["df_TCC_w_streaks_and_means"] = df_means
    st.session_state["df_meas_data_w_smooth_w_TCC_id"] = df_meas
    st.session_state["plot_data"] = plot_data

    ss1_result = _run_with_log(
        lambda: analysis.analyze_steady_state(
            df_means, nominal_ambient=float(test["ambient_c"])
        )
    )
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

    delta_e, delta_t_df = _run_with_log(
        lambda: analysis.analyze_defrost(df_means, df_periods, df_meas)
    )
    st.session_state["Delta_E_dfj_wh"] = delta_e
    st.session_state["delta_t_df"] = delta_t_df

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
    ss2 = _compute_ss2(df_means, float(test["ambient_c"])) if n_events >= 2 else None
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

tab_need, tab_appliance, tab_tests, tab_analyse, tab_combine, tab_log = st.tabs(
    [
        "Wat is nodig",
        "Toestel",
        "Meetpunten",
        "Analyse één meting",
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
    st.subheader("Meetpunten (elke meting eigen bestanden)")
    st.caption(
        "De exacte instelwaarde hoeft niet bekend te zijn. Kies of de knop kouder of warmer "
        "staat dan de doeltemperatuur. Per omgevingstemperatuur zijn beide nodig voor interpolatie. "
        "Meetbestanden: uploaden in de browser, of lokaal een bestand/map op je schijf kiezen."
    )
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
            _render_test_files(test, i)
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

    st.download_button(
        "Campagne exporteren (JSON)",
        data=json.dumps(
            {
                "compartments": st.session_state["compartments_ui"],
                "tests": [
                    {k: v for k, v in t.items() if k != "point"} | {"point": t.get("point")}
                    for t in st.session_state["tests"]
                ],
            },
            indent=2,
            default=str,
        ),
        file_name="campagne_en62552.json",
        mime="application/json",
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
