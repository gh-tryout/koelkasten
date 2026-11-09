"""
D.3 Compartment average air temperatures determination
    D.3.1 General
    For each compartment, the average air temperature at each sensor location is determined.
    These temperatures are then combined to determine the compartment temperature.

    D.3.2 Determination of the average temperature of a sensor over a period
    The average temperature of a sensor over a period shall be determined by integration. More
    frequent sampling for selected periods during the test may be used. For example, higher
    frequency sampling may be useful for short duration events. If it is necessary to combine data
    with dissimilar sampling rates, each data point shall be weighted in proportion to its relevant
    sampling period.

    D.3.3 Determination of the temperature of a compartment
    The temperature of a compartment shall be determined over the applicable temperature
    determination period.

    D.3.4 Calculation of temperature average
    The compartment temperature shall be the arithmetic mean of the temperatures for all
    temperature sensors in the compartment.
"""

"""
Dit programma verwerkt ruwe meetdata van vermogensmeters en temperatuursensoren om de
efficiëntie van een koelapparaat te kwantificeren volgens specifieke teststandaarden.

KERNBEGRIPPEN:
1. Cycles (TCC - Temperature Control Cycles):
   Dit zijn de individuele aan/uit-cycli van de compressor.
   - 'On-duration': De tijd dat de compressor actief koelt.
   - 'TCC_duration': De totale tijd van het begin van één koelactie tot de volgende.
   Het script identificeert deze cycli door een drempelwaarde (threshold) toe te passen
   op het (gefilterde) vermogenssignaal.

2. Periods (Analyseperioden):
   Een 'Period' is een aaneengesloten reeks van meerdere TCC's (bijv. 3 blokken van 'k' cycli).
   Deze worden gebruikt om te controleren op stabiliteit.
   - Steady-State (SS): Een periode waarin de gemiddelde temperatuur en het vermogen
     binnen strikte toleranties blijven (bijv. < 0.25K drift).
   - D- en F-perioden: Specifieke stabiele perioden direct vóór (D - Departure) en
     na (F - Finish) een ontdooicyclus om de herstelenergie te berekenen.

HOOFDPROCEDURE:
- Data Ingestie: Laden van vermogensdata (Power) en temperatuurdata (CSV/UTF-16).
- Filtering: Toepassen van median-filters om ruis te verwijderen en piekniveaus
  (compressor aan) versus basisniveaus (compressor uit) te bepalen via histogrammen.
- TCC Segmentatie: Het opsplitsen van de tijdslijn in individuele compressor-acties.
- Steady-State Detectie: Het algoritmisch zoeken naar de meest stabiele 6- tot 12-uurs
  vensters waarin het apparaat thermisch in evenwicht is.
- Ontdooi-analyse: Het detecteren van 'Defrost' pieken en het berekenen van de
  bijbehorende energie-impact (Delta_E_dfj) door het verbruik rondom de ontdooiing
  te vergelijken met de stabiele perioden (D en F).
- Calculatie: Berekening van het uiteindelijke dagelijkse energieverbruik (pss)
  gecorrigeerd naar een standaard omgevingstemperatuur.
"""

from time import time

start_time = time()
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
import re
import numpy as np
import polars as pl
import polars.selectors as cs
import sys
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

end_time = time()
print(f"loading imports took {end_time - start_time:.5f} seconds")

DEBUG = True
DEBUG = False
DEBUG3 = True
DEBUG3 = False


class SS1NotFoundError(ValueError):
    """Geen geldige SS1, met tabel van alle kandidaat-perioden en waarom ze afvallen."""

    def __init__(self, message: str, diagnostics=None, summary=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.summary = summary or {}


def format_dates(fig, ax):
    fig.autofmt_xdate()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(
        axis="x", which="minor", length=6, labelsize=8, labelbottom=True, pad=6
    )
    ax.tick_params(axis="x", which="major", length=10, labelsize=9, pad=10)
    fig.autofmt_xdate(rotation=30)
    ax.set_axisbelow(True)
    ax.xaxis.grid(
        True, which="major", linestyle="-", linewidth=0.8, color="gray", alpha=0.6
    )
    ax.xaxis.grid(
        True, which="minor", linestyle="--", linewidth=0.6, color="gray", alpha=0.4
    )


def plot_power_and_histogram(
    df_meas_data_w_median: pl.DataFrame,
    histogram: pl.DataFrame,
    threshold: float,
    base_level: float,
    peak_level: float,
):
    fig, (ax1, ax2) = plt.subplots(
        figsize=(14, 6), ncols=2, sharey=True, gridspec_kw={"width_ratios": [5, 1]}
    )

    x = df_meas_data_w_median[time_col].to_list()
    y = df_meas_data_w_median[val_col].to_list()
    y_smooth = df_meas_data_w_median[val_col_smooth_long].to_list()

    ax1.axhline(threshold, color="green", alpha=0.8, label="threshold")
    ax1.plot(x, y, label=val_col)
    ax1.plot(x, y_smooth, label=val_col_smooth_long)
    ax1.set_title("power consumption over time with base and peak levels")
    ax1.legend(title="Power Data")
    ax1.set_xlim([min(x), max(x)])  # type: ignore
    format_dates(fig, ax1)
    ax1.axhline(
        base_level,
        color="red",
        linestyle="dashdot",
        label=f"Base Level: {base_level:.5f} W",
    )
    ax1.axhspan(
        base_level - ROUNDING_INTERVAL / 2,
        base_level + ROUNDING_INTERVAL / 2,
        color="black",
        alpha=0.1,
    )
    ax1.axhline(
        peak_level,
        color="black",
        linestyle="solid",
        label=f"Peak Level: {peak_level:.5f} W",
    )
    ax1.axhspan(
        peak_level - ROUNDING_INTERVAL / 2,
        peak_level + ROUNDING_INTERVAL / 2,
        color="black",
        alpha=0.1,
    )
    ax1.set_ylabel("Power (W)")
    ax1.set_xlabel("Time")
    ax1.legend()

    ax2.barh(
        y=histogram["bin_center"].to_list(),
        width=histogram["count"].to_list(),
        height=ROUNDING_INTERVAL * 0.9,
        align="center",
        left=0,
    )
    ax2.set_xlabel("Frequency")
    ax2.set_title("Histogram")
    ax2.set_xscale("log")
    ax2.axhline(
        base_level,
        color="red",
        linestyle="dashdot",
        label=f"Base Level: {base_level:.5f} W",
    )
    ax2.axhline(
        peak_level,
        color="black",
        linestyle="solid",
        label=f"Peak Level: {peak_level:.5f} W",
    )
    plt.subplots_adjust(wspace=0.05)
    return fig


def plot_detected_cycles(
    df_TCC: pl.DataFrame,
    df_meas_data_w_median: pl.DataFrame,
    threshold: float,
    ss1_opti: dict | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("Pastel2")  # pyright: ignore[reportAttributeAccessIssue]
    for cycle in df_TCC.iter_rows(named=True):
        ax.axvspan(
            cycle["TCC_start_datetime"],
            cycle["TCC_end_datetime"],
            color=cmap(cycle["TCC_id"] % cmap.N),
            alpha=0.5,
        )

    x = df_meas_data_w_median[time_col].to_list()
    y = df_meas_data_w_median[val_col].to_list()
    y_smooth = df_meas_data_w_median[val_col_smooth_short].to_list()

    ax.axhline(threshold, color="green", alpha=0.8, label="threshold")
    ax.plot(x, y, label=val_col, color="red", alpha=0.5)
    ax.plot(x, y_smooth, label=f"{val_col_smooth_short}", color="k", alpha=0.5)
    if ss1_opti is not None:
        t0 = ss1_opti["period_start_datetime"]
        t1 = t0 + ss1_opti["period_duration"]
        ax.axvspan(t0, t1, color="tab:blue", alpha=0.25, label="SS1")
    ax.set_title("Power Data with Detected Cycles")
    ax.set_xlim([min(x), max(x)])  # type: ignore
    format_dates(fig, ax)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time")
    ax.legend()
    return fig


def plot_compartment_temperatures(df_TCC_w_streaks_and_means: pl.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = df_TCC_w_streaks_and_means["TCC_start_datetime"].to_list()
    for col in mean_temp_cols + ["temp_mean_amb"]:
        if col in df_TCC_w_streaks_and_means.columns:
            ax.plot(x, df_TCC_w_streaks_and_means[col].to_list(), label=col)
    ax.set_title("Time-weighted mean temperatures per TCC")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Time")
    format_dates(fig, ax)
    ax.legend()
    return fig


def compute_daily_energy_kwh(
    pss: float,
    delta_e_dfj_wh: float,
    delta_t_df: pl.DataFrame,
) -> tuple[float, float | None]:
    delta_t_mean = delta_t_df["delta_t_defrost_h"].mean() if delta_t_df.height else None
    if delta_t_mean is None:
        return 0.001 * 24 * pss, None
    return 0.001 * 24 * (pss + delta_e_dfj_wh / delta_t_mean), float(delta_t_mean)


compartments = {
    "fridge": {
        "temp sensors": ["105 (C)", "202 (C)", "209 (C)"],
        "volume": 233,
        "compartment type": "unfrozen",
        "target_temp": 4,
    },
    "freezer 3 star": {
        "temp sensors": ["102 (C)", "103 (C)", "201 (C)", "203 (C)", "107 (C)"],
        "volume": 107,
        "compartment type": "frozen",
        "target_temp": -18,
    },
}

# check if there are both frozen and unfrozen compartments
N_compartment_types = len(
    {compartment["compartment type"] for compartment in compartments.values()}
)
N_compartments = len(compartments.keys())

# achterhaal welke van de compartments gebruikt moeten worden voor de periode selectie
name_vol_type = [
    {
        "name": compartment_name,
        "volume": compartment_specs["volume"],
        "type": compartment_specs["compartment type"],
    }
    for compartment_name, compartment_specs in compartments.items()
]
# lists of compartments that are frozen or unfrozen sorted by volume
frozen_compartment_sorted_by_volume = [
    z["name"]
    for z in sorted(
        filter(lambda x: x["type"] == "frozen", name_vol_type),
        key=lambda y: y["volume"],
        reverse=True,
    )
]
unfrozen_compartment_sorted_by_volume = [
    z["name"]
    for z in sorted(
        filter(lambda x: x["type"] == "unfrozen", name_vol_type),
        key=lambda y: y["volume"],
        reverse=True,
    )
]

if (
    len(frozen_compartment_sorted_by_volume) > 0
    and len(unfrozen_compartment_sorted_by_volume) > 0
):
    compartments_used_for_period_selection = [
        frozen_compartment_sorted_by_volume[0],
        unfrozen_compartment_sorted_by_volume[0],
    ]
elif len(frozen_compartment_sorted_by_volume) > 0:
    compartments_used_for_period_selection = frozen_compartment_sorted_by_volume[:2]
elif len(unfrozen_compartment_sorted_by_volume) > 0:
    compartments_used_for_period_selection = unfrozen_compartment_sorted_by_volume[:2]
else:
    raise ValueError("No compartments defined")


gradient_temp_sensors = ["110 (C)", "206 (C)"]
ambient_temp_sensors = ["205 (C)", "210 (C)", "204 (C)"]
# ["110 (C)", "210 (C)", "206 (C)", "204 (C)"]
all_temp_sensors = (
    [*[x for y in [x["temp sensors"] for x in compartments.values()] for x in y]]
    + gradient_temp_sensors
    + ambient_temp_sensors
)


power_files = "power_meetdata_map/Auto_20260416143058_*"
# temp_file = "temp_data_utf8.csv"
# temp_file = "meting eind dec 2025/Data*.csv"
temp_file = "temp_meetdata_map/Data 4_16_2026 14_31_15 1.csv"

time_col: str = "Datetime"
val_col: str = "P-1"
val_col_smooth_long: str = "P-1_smooth_30_sec"
val_col_smooth_short: str = "P-1_smooth_4_sec"
duration_col: str = "duration_t"
power_col: str = "P-1"
defrost_power_threshold_w: float = 50.0

ROUNDING_INTERVAL = 2.0  # watts for histogram binning when determining base/peak levels

time_weighted_means_sensors = all_temp_sensors + [power_col]
mean_temp_cols = [f"temp_mean_{comp}" for comp in compartments.keys()]
mean_temp_cols_for_period_selection = [
    f"temp_mean_{comp}" for comp in compartments_used_for_period_selection
]

# ISO 16358 ambient-temperature correction constants
target_ambient_temp = 16
delta_cop = 0.00  # per degree C; Annex B ΔCOP (0 bij 16 °C, zie Tabel B.1)
c1 = 0.011364
c2 = 1.25


def configure_appliance(
    new_compartments: dict,
    ambient_sensors: list[str] | None = None,
    gradient_sensors: list[str] | None = None,
    defrost_power_w: float | None = None,
) -> None:
    """Zet compartimenten/volumes/sensoren voor een nieuwe toestelconfiguratie."""
    global compartments, compartments_used_for_period_selection
    global all_temp_sensors, time_weighted_means_sensors
    global mean_temp_cols, mean_temp_cols_for_period_selection
    global ambient_temp_sensors, gradient_temp_sensors
    global N_compartments, N_compartment_types
    global defrost_power_threshold_w

    compartments = new_compartments
    if ambient_sensors is not None:
        ambient_temp_sensors = list(ambient_sensors)
    if gradient_sensors is not None:
        gradient_temp_sensors = list(gradient_sensors)
    if defrost_power_w is not None:
        defrost_power_threshold_w = float(defrost_power_w)

    N_compartment_types = len(
        {compartment["compartment type"] for compartment in compartments.values()}
    )
    N_compartments = len(compartments)

    name_vol_type = [
        {
            "name": compartment_name,
            "volume": compartment_specs["volume"],
            "type": compartment_specs["compartment type"],
        }
        for compartment_name, compartment_specs in compartments.items()
    ]
    frozen_sorted = [
        z["name"]
        for z in sorted(
            filter(lambda x: x["type"] == "frozen", name_vol_type),
            key=lambda y: y["volume"],
            reverse=True,
        )
    ]
    unfrozen_sorted = [
        z["name"]
        for z in sorted(
            filter(lambda x: x["type"] == "unfrozen", name_vol_type),
            key=lambda y: y["volume"],
            reverse=True,
        )
    ]
    if frozen_sorted and unfrozen_sorted:
        compartments_used_for_period_selection = [frozen_sorted[0], unfrozen_sorted[0]]
    elif frozen_sorted:
        compartments_used_for_period_selection = frozen_sorted[:2]
    elif unfrozen_sorted:
        compartments_used_for_period_selection = unfrozen_sorted[:2]
    else:
        raise ValueError("No compartments defined")

    all_temp_sensors = (
        [s for spec in compartments.values() for s in spec["temp sensors"]]
        + gradient_temp_sensors
        + ambient_temp_sensors
    )
    time_weighted_means_sensors = all_temp_sensors + [power_col]
    mean_temp_cols = [f"temp_mean_{comp}" for comp in compartments.keys()]
    mean_temp_cols_for_period_selection = [
        f"temp_mean_{comp}" for comp in compartments_used_for_period_selection
    ]


def weighted_mean_expr(col_name: str, weight_col: str) -> pl.Expr:
    return (pl.col(col_name) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()


def weighted_time_accumulation_expr_hours(value_col: str, time_col: str) -> pl.Expr:
    return (pl.col(value_col) * pl.col(time_col).dt.total_hours(fractional=True)).sum()


def get_period_stats(block_a_df, block_b_df, block_c_df):
    """ "Compute statistics for a given period defined by three blocks (A, B, C).
    Each block represents a sequence of TCCs, and the function calculates various statistics
    such as period duration, average temperatures, temperature spreads, temperature slopes,
    power spread, power slope, power average, and steady-state compressor run time for the period.
    The function assumes that the input dataframes contain the necessary columns for these
    calculations and that they are structured in a way that allows for the computations to be
    performed as described.

    First, the function concatenates the three blocks of TCCs into a single dataframe and calculates
    the total duration of the period by summing the durations of the individual blocks. It also calculates
    the time duration between the midpoints of block A and block C, which is used for slope calculations.

    Then it computes the time-weighted mean temperatures for each sensor across the entire period using
    the provided weighted mean expression. It also calculates the temperature spread for the sensors
    used in period selection by finding the difference between the maximum and minimum values across
    the blocks. The temperature slopes are calculated by taking the difference in mean temperatures
    between block C and block A and dividing by the time duration between their midpoints.

    For power, it calculates the spread as the peak-to-peak value across the blocks, the slope as the
    difference in mean power between block C and block A divided by the time duration, and the average
    power across the period. It also computes the percentage spread and slope relative to the average power.
    """
    period_start_datetime = block_a_df["block_TCC_start_datetime"].item()

    df_blocks = pl.concat([block_a_df, block_b_df, block_c_df])
    df_blocks.write_parquet("df_blocks.parquet")

    period_duration = df_blocks["block_duration"].sum()
    a_c_time_duration = (
        block_c_df["block_time_midpoint"] - block_a_df["block_time_midpoint"]
    ).item().total_seconds() / 3600

    temp_means = df_blocks.select(
        [weighted_mean_expr(col, "block_duration") for col in mean_temp_cols]
    )

    temp_spreads = df_blocks.select(mean_temp_cols_for_period_selection).select(
        (cs.numeric().max() - cs.numeric().min())
    )
    temp_slopes = (
        block_c_df[mean_temp_cols_for_period_selection]
        - block_a_df[mean_temp_cols_for_period_selection]
    ) / a_c_time_duration

    power_spread = np.ptp(df_blocks[power_col].to_numpy())
    power_slope = (
        block_c_df[power_col] - block_a_df[power_col]
    ).item() / a_c_time_duration

    power_avg = df_blocks.select(weighted_mean_expr(power_col, "block_duration")).item()
    power_spread_pct = (power_spread / power_avg) * 100
    power_slope_pct = (power_slope / power_avg) * 100

    # steady state compressor run time the percentage of time that the compressor is on during the total time
    period_compressor_on_duration = df_blocks["block_total_on_duration"].sum()
    period_steady_state_compressor_run_time = (
        period_compressor_on_duration / period_duration * 100
    )

    return {
        "period_duration": period_duration,
        **{col: val for col, val in zip(temp_means.columns, temp_means.row())},
        **{
            f"temp_spread_{col}": val
            for col, val in zip(temp_spreads.columns, temp_spreads.row())
        },
        **{
            f"temp_slope_{col}": val
            for col, val in zip(temp_slopes.columns, temp_slopes.row())
        },
        "power_avg": power_avg,
        "power_spread": power_spread,
        "power_spread_pct": power_spread_pct,
        "power_slope_pct": power_slope_pct,
        "period_steady_state_compressor_run_time": period_steady_state_compressor_run_time,
        "period_start_datetime": period_start_datetime,
    }


def _detect_text_encoding(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read(4)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "UTF-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _first_existing_path(files: str | list[str]) -> str:
    if isinstance(files, (list, tuple)):
        if not files:
            raise FileNotFoundError("Geen power-bestanden opgegeven.")
        return str(files[0])
    matches = glob(files)
    if matches:
        return matches[0]
    if Path(files).is_file():
        return files
    raise FileNotFoundError(f"Geen power-bestand gevonden: {files}")


def _detect_csv_separator(path: str) -> str:
    """Yokogawa-exports gebruiken ';', oudere dumps soms ','."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    return ";" if header.count(";") > header.count(",") else ","


POWER_TIME_SLACK = timedelta(hours=6)
POWER_GAP_LIMIT = timedelta(seconds=30)


def _decode_text_sample(raw: bytes, encoding: str | None = None) -> str:
    enc = encoding or _detect_text_encoding_from_bytes(raw[:4])
    return raw.decode(enc, errors="replace")


def _detect_text_encoding_from_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "UTF-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _read_last_nonempty_line(path: str, encoding: str) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 131072))
        chunk = fh.read()
    if encoding.upper().startswith("UTF-16"):
        text = chunk.decode(encoding, errors="replace")
    else:
        text = chunk.decode(encoding, errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _power_datetime_from_row(header: str, line: str, sep: str) -> datetime | None:
    cols = [c.strip() for c in header.strip().split(sep)]
    vals = [v.strip() for v in line.strip().split(sep)]
    if len(vals) < 3 or len(cols) < 3:
        return None
    row = dict(zip(cols, vals))
    date = row.get("Date")
    time_v = row.get("Time")
    if not date or not time_v:
        return None
    ms = str(row.get("Millisecond") or "0").strip() or "0"
    ms = ms.split(".")[0].zfill(3)[:3]
    try:
        return datetime.strptime(f"{date} {time_v}.{ms}", "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(f"{date} {time_v}", "%Y/%m/%d %H:%M:%S")
        except ValueError:
            return None


def power_file_span(path: str) -> tuple[datetime, datetime] | None:
    """Eerste en laatste timestamp van een wattmeter-CSV, zonder het hele bestand te lezen."""
    encoding = _detect_text_encoding(path)
    sep = _detect_csv_separator(path)
    with open(path, encoding=encoding, errors="replace") as fh:
        header = fh.readline()
        first = ""
        for line in fh:
            if line.strip():
                first = line
                break
    if not header or not first:
        return None
    last = _read_last_nonempty_line(path, encoding)
    t0 = _power_datetime_from_row(header, first, sep)
    t1 = _power_datetime_from_row(header, last, sep) or t0
    if t0 is None:
        return None
    if t1 < t0:
        t0, t1 = t1, t0
    return t0, t1


def temp_file_span(path: str) -> tuple[datetime, datetime]:
    encoding = _detect_text_encoding(path)
    skip = _temp_csv_header_skip(path, encoding)
    df = pl.read_csv(
        path,
        skip_rows=skip,
        encoding=encoding,
        truncate_ragged_lines=True,
        columns=["Time"],
    )
    if "Time" not in df.columns or df.height == 0:
        raise ValueError(f"Geen tijdkolom in temperatuur-CSV: {path}")
    times = (
        df.lazy()
        .select(pl.col("Time").str.to_datetime("%d-%m-%Y %H:%M:%S:%3f", strict=False))
        .drop_nulls()
        .collect()
    )
    if times.height == 0:
        raise ValueError(f"Geen geldige tijden in temperatuur-CSV: {path}")
    tmin = times["Time"].min()
    tmax = times["Time"].max()
    if hasattr(tmin, "to_pydatetime"):
        tmin = tmin.to_pydatetime()
        tmax = tmax.to_pydatetime()
    return tmin, tmax


def select_power_files_for_temp(
    power_files: str | list[str],
    temp_file: str,
    slack: timedelta | None = None,
) -> list[str]:
    """Kies power-CSV’s waarvan de tijdspanne de temperatuurmeting raakt (met marge)."""
    slack = slack if slack is not None else POWER_TIME_SLACK
    if isinstance(power_files, (str, Path)):
        raw = glob(str(power_files))
        if not raw and Path(power_files).is_dir():
            raw = [
                str(p)
                for p in Path(power_files).iterdir()
                if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".dat"}
            ]
        paths = raw
    else:
        paths = [str(p) for p in power_files]
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        raise FileNotFoundError("Geen power-bestanden om uit te kiezen.")
    tmin, tmax = temp_file_span(temp_file)
    window_lo = tmin - slack
    window_hi = tmax + slack
    spans: list[tuple[str, datetime, datetime]] = []
    for path in paths:
        span = power_file_span(path)
        if span is None:
            continue
        spans.append((path, span[0], span[1]))
    if not spans:
        raise ValueError("Kon in geen enkel power-bestand een tijdspanne lezen.")
    chosen = [
        (p, a, b) for p, a, b in spans if b >= window_lo and a <= window_hi
    ]
    if not chosen:
        def _gap(a: datetime, b: datetime) -> float:
            if b < tmin:
                return (tmin - b).total_seconds()
            if a > tmax:
                return (a - tmax).total_seconds()
            return 0.0

        nearest = min(spans, key=lambda x: _gap(x[1], x[2]))
        nearest_gap = _gap(nearest[1], nearest[2])
        if nearest_gap > timedelta(hours=24).total_seconds():
            listing = "; ".join(
                f"{Path(p).name}: {a} – {b}" for p, a, b in sorted(spans, key=lambda x: x[1])[:8]
            )
            raise ValueError(
                "Geen power-bestanden bij de temperatuurmeting. "
                f"Temperatuur: {tmin} – {tmax}. Power: {listing}"
            )
        cluster_lo = nearest[1] - slack
        cluster_hi = nearest[2] + slack
        chosen = [(p, a, b) for p, a, b in spans if b >= cluster_lo and a <= cluster_hi]
    cmin = min(a for _, a, _ in chosen)
    cmax = max(b for _, _, b in chosen)
    filled = [(p, a, b) for p, a, b in spans if b >= cmin and a <= cmax]
    filled.sort(key=lambda x: x[1])
    return [p for p, _, _ in filled]


def power_file_gaps(
    paths: list[str],
    min_gap: timedelta | None = None,
) -> list[dict]:
    """Gaten tussen opeenvolgende power-CSV’s (einde bestand i tot start bestand i+1)."""
    min_gap = min_gap if min_gap is not None else POWER_GAP_LIMIT
    spans: list[tuple[str, datetime, datetime]] = []
    for path in paths:
        span = power_file_span(path)
        if span is None:
            continue
        spans.append((path, span[0], span[1]))
    spans.sort(key=lambda x: x[1])
    gaps: list[dict] = []
    for (p0, _a0, b0), (p1, a1, _b1) in zip(spans, spans[1:]):
        gap = a1 - b0
        if gap > min_gap:
            gaps.append(
                {
                    "bron": "bestanden",
                    "na_bestand": Path(p0).name,
                    "voor_bestand": Path(p1).name,
                    "einde": b0,
                    "start": a1,
                    "duur_s": gap.total_seconds(),
                }
            )
    return gaps


def datetime_series_gaps(
    times,
    min_gap: timedelta | None = None,
    sample_limit: timedelta | None = None,
) -> list[dict]:
    """Gaten in een gesorteerde datetime-reeks (uitgelijnde powermeting)."""
    min_gap = min_gap if min_gap is not None else POWER_GAP_LIMIT
    seq = list(times)
    if len(seq) < 2:
        return []
    diffs: list[float] = []
    for a, b in zip(seq, seq[1:]):
        if a is None or b is None:
            continue
        diffs.append((b - a).total_seconds())
    if not diffs:
        return []
    diffs_sorted = sorted(diffs)
    median_s = diffs_sorted[len(diffs_sorted) // 2]
    limit_s = min_gap.total_seconds()
    if sample_limit is not None:
        limit_s = max(limit_s, sample_limit.total_seconds())
    elif median_s > 0:
        limit_s = max(limit_s, 5.0 * median_s)
    gaps: list[dict] = []
    for a, b, dt_s in zip(seq, seq[1:], diffs):
        if dt_s > limit_s:
            gaps.append(
                {
                    "bron": "samples",
                    "einde": a,
                    "start": b,
                    "duur_s": dt_s,
                }
            )
    return gaps


_WATTMETER_PREFIX = re.compile(r"^WT\d+:")


def _cast_numeric_csv_col(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .replace(["NAN", "Error", "ERROR", "---", ""], None)
        .cast(pl.Float64, strict=False)
    )


def _ensure_power_column(lf: pl.LazyFrame, names: list[str]) -> pl.LazyFrame:
    """Zorg voor een numerieke P-1-kolom; ontbreekt die, dan U-1 × I-1."""
    if power_col in names:
        return lf.with_columns(_cast_numeric_csv_col(power_col))
    if "U-1" in names and "I-1" in names:
        lf = lf.with_columns(
            _cast_numeric_csv_col("U-1"),
            _cast_numeric_csv_col("I-1"),
        ).with_columns((pl.col("U-1") * pl.col("I-1")).alias(power_col))
        print("vermogen: P-1 = U-1 × I-1 (geen P-kanaal in de CSV)")
        return lf
    raise ValueError(
        f"Geen vermogenskolom '{power_col}' gevonden. Kolommen: {names}"
    )


def _normalize_wattmeter_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Zet WT1:P-1 / WT2:P-1 (enz.) om naar P-1, U-1, … zoals de rest van de code verwacht."""
    names = lf.collect_schema().names()
    wt_cols = [c for c in names if _WATTMETER_PREFIX.match(c)]
    if not wt_cols:
        return _ensure_power_column(lf, names)

    lf = lf.with_columns(
        [
            pl.col(c)
            .str.strip_chars()
            .replace(["NAN", "Error", "ERROR", "---", ""], None)
            .cast(pl.Float64, strict=False)
            for c in wt_cols
        ]
    )
    # WT1 wint bij gelijke meetgrootheid als beide kanalen in het bestand zitten.
    wt_cols_sorted = sorted(
        wt_cols, key=lambda c: (0 if c.startswith("WT1:") else 1, c)
    )
    taken = {n for n in names if n not in wt_cols}
    rename_map: dict[str, str] = {}
    for src in wt_cols_sorted:
        dst = _WATTMETER_PREFIX.sub("", src)
        if dst in taken:
            continue
        rename_map[src] = dst
        taken.add(dst)
    if rename_map:
        lf = lf.rename(rename_map)

    schema = lf.collect_schema().names()
    if power_col not in schema:
        candidates = [c for c in schema if c.endswith("P-1")]
        if not candidates:
            raise ValueError(
                f"Geen kolom '{power_col}' na hernoemen van wattmeterkanalen. "
                f"Kolommen: {schema}"
            )
        lf = lf.rename({candidates[0]: power_col})
        print(f"vermogen: '{candidates[0]}' gebruikt als {power_col}")
    else:
        src = next(
            (c for c in wt_cols_sorted if _WATTMETER_PREFIX.sub("", c) == power_col),
            power_col,
        )
        print(f"vermogen: {src} -> {power_col}")
    return lf


def _temp_csv_header_skip(path: str, encoding: str) -> int:
    """Aantal rijen vóór de datakop 'Scan,Time,...' (hangt af van het aantal kanalen)."""
    with open(path, encoding=encoding, errors="replace") as fh:
        for i, line in enumerate(fh):
            first = line.split(",")[0].strip().strip('"')
            if first == "Scan":
                return i
    raise ValueError(
        f"Geen kopregel 'Scan,Time,...' gevonden in {path}. "
        "Controleer of dit een temperatuur-CSV van de datalogger is."
    )


def load_power_data(power_files: str | list[str]) -> pl.LazyFrame:
    """Lees wattmeter-CSV’s als LazyFrame met Datetime en P-1, zonder temperatuur."""
    sep = _detect_csv_separator(_first_existing_path(power_files))
    lf_power = pl.scan_csv(power_files, infer_schema=False, separator=sep)
    names = lf_power.collect_schema().names()
    if "StoreNo" in names and "Store No." not in names:
        lf_power = lf_power.rename({"StoreNo": "Store No."})
        names = lf_power.collect_schema().names()
    if "Date" not in names:
        raise ValueError(
            "Power-CSV heeft geen kolom 'Date'. "
            f"Scheidingsteken={sep!r}. Kolommen: {names}"
        )
    drop_power = [c for c in ("Date", "Time", "Millisecond", "Store No.") if c in names]
    lf_power = (
        lf_power.select(~cs.contains("duplicated"))
        .with_columns(
            pl.format(
                "{} {}.{}",
                pl.col("Date"),
                pl.col("Time"),
                pl.col("Millisecond").cast(pl.String).str.pad_start(3, "0"),
            )
            .str.to_datetime("%Y/%m/%d %H:%M:%S.%3f")
            .alias(time_col)
        )
        .drop(drop_power)
    )
    lf_power = _normalize_wattmeter_columns(lf_power)
    return (
        lf_power.drop_nulls(subset=[time_col, power_col])
        .select([time_col, cs.all().exclude(time_col)])
        .sort(time_col)
    )


def load_and_align_data(
    power_files: str | list[str],
    temp_file: str,
) -> pl.LazyFrame:
    """Lees stroom- en temperatuurdata in, align op tijd en interpolatie ontbrekende waarden.

    De CSV-scans, de join en de interpolatie blijven een Polars LazyFrame tot de caller
    materialiseert. Temperatuurdata wordt eager gelezen vanwege UTF-16, daarna direct lazy.
    """
    lf_power = load_power_data(power_files)
    power_span = lf_power.select(
        pl.col(time_col).min().alias("tmin"),
        pl.col(time_col).max().alias("tmax"),
        pl.len().alias("n"),
    ).collect().row(0, named=True)

    # Temp pipeline: UTF-16 is niet betrouwbaar via scan_csv, daarom read_csv + lazy().
    # Het aantal kopregels hangt af van het aantal kanalen; zoek de regel Scan,Time.
    temp_encoding = _detect_text_encoding(temp_file)
    temp_skip = _temp_csv_header_skip(temp_file, temp_encoding)
    print(f"temperatuur-CSV: encoding={temp_encoding}, skip_rows={temp_skip}")
    df_temp = pl.read_csv(
        temp_file,
        skip_rows=temp_skip,
        encoding=temp_encoding,
        truncate_ragged_lines=True,
    )
    if "Time" not in df_temp.columns:
        raise ValueError(
            "Temperatuur-CSV heeft geen kolom 'Time' na de kopregel. "
            f"Gevonden kolommen: {df_temp.columns[:8]}"
        )
    drop_cols = [c for c in ("Scan",) if c in df_temp.columns]
    lf_temp = (
        df_temp.lazy()
        .drop(drop_cols)
        .select(~cs.starts_with("Alarm"))
        .with_columns(pl.col("Time").str.to_datetime("%d-%m-%Y %H:%M:%S:%3f"))
        .rename({"Time": time_col})
        .select([time_col, cs.all().exclude(time_col)])
        .sort(time_col)
    )
    temp_span = lf_temp.select(
        pl.col(time_col).min().alias("tmin"),
        pl.col(time_col).max().alias("tmax"),
        pl.len().alias("n"),
    ).collect().row(0, named=True)
    if (
        power_span["tmin"] is None
        or temp_span["tmin"] is None
        or power_span["tmax"] < temp_span["tmin"]
        or temp_span["tmax"] < power_span["tmin"]
    ):
        raise ValueError(
            "Power- en temperatuurdata overlappen niet in de tijd. "
            f"Power: {power_span['tmin']} – {power_span['tmax']} "
            f"({power_span['n']} rijen). "
            f"Temperatuur: {temp_span['tmin']} – {temp_span['tmax']} "
            f"({temp_span['n']} rijen). "
            "Kies bestanden van dezelfde meetperiode."
        )

    # Join and interpolate. Semi-join houdt alleen timestamps die in de power-data zitten.
    lf_meas_data = (
        lf_power.join(lf_temp, on=time_col, how="full", coalesce=True)
        .sort(time_col)
        # interpolate using time column
        .with_columns(cs.numeric().exclude(time_col).interpolate_by(pl.col(time_col)))
        .with_columns(meas_duration=pl.col("Datetime").diff().shift(-1))
        .drop_nulls(subset=[time_col, power_col])
        # Alleen het overlappende deel: buiten de temp-reeks blijft interpolatie leeg.
        .filter(
            pl.any_horizontal(
                pl.col(c).is_not_null()
                for c in lf_temp.collect_schema().names()
                if c != time_col
            )
        )
        # Semi-join acts as a filter to keep only timestamps present in power
        .join(lf_power.select(time_col), on=time_col, how="semi")
        .with_columns(pl.col(time_col).diff().shift(-1).alias(duration_col))
        .filter(pl.col(duration_col).is_not_null())
    )
    return lf_meas_data


def detect_tcc_cycles(
    lf_meas_data: pl.LazyFrame,
    *,
    make_plots: bool | None = None,
    defrost_power_w: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Filter het vermogenssignaal en knip het op in compressor-cycli (TCC's).

    Median-filter, histogram/threshold, TCC-segmentatie en time-weighted means per cyclus
    lopen via LazyFrames tot een collect nodig is (numpy-histogramdrempels, plots, parquet).
    """
    if make_plots is None:
        make_plots = DEBUG
    thresh_w = float(
        defrost_power_threshold_w if defrost_power_w is None else defrost_power_w
    )
    print(f"ontdooi-drempel: P > {thresh_w:g} W")
    start_time = time()
    df_meas_data = lf_meas_data.collect()
    # df_meas_data.write_parquet("df_meas_data.parquet")
    # sys.exit(0)
    print(f"everything before smoothing filter took {time() - start_time:.5f} seconds")
    if df_meas_data.height < 2:
        raise ValueError(
            f"Te weinig meetpunten na uitlijnen ({df_meas_data.height}). "
            "Power- en temperatuurreeksen overlappen waarschijnlijk niet."
        )

    mean_dur = (
        df_meas_data.lazy()
        .select(pl.col(duration_col).mean())
        .collect()
        .item()
    )
    if mean_dur is None:
        raise ValueError(
            "Geen geldige tijdsintervallen in de uitgelijnde meetdata "
            f"(kolom {duration_col})."
        )
    mean_sampling_interval_seconds = mean_dur.total_seconds()
    if mean_sampling_interval_seconds <= 0:
        raise ValueError(
            f"Ongeldig sampling-interval: {mean_sampling_interval_seconds} s."
        )
    sigma_seconds_long = 30  # seconds
    sigma_samples_long = max(3, int(sigma_seconds_long / mean_sampling_interval_seconds))
    sigma_seconds_short = 4  # seconds
    sigma_samples_short = max(3, int(sigma_seconds_short / mean_sampling_interval_seconds))

    start_time = time()
    df_meas_data_w_median = (
        df_meas_data.lazy()
        .with_columns(
            pl.col(power_col)
            .rolling_median(window_size=sigma_samples_short, center=True)
            .alias(val_col_smooth_short),
            pl.col(power_col)
            .rolling_median(window_size=sigma_samples_long, center=True)
            .alias(val_col_smooth_long),
        )
        .slice(sigma_samples_long, df_meas_data.height - 2 * sigma_samples_long)
        .collect()
    )
    print(f"median filter took {time() - start_time:.5f} seconds")

    # determine base and peak levels by making a histogram of the smoothed power data
    # and taking the two most common bins using a bin width of ROUNDING_INTERVAL
    # NOTE a more rigid approach could be to use the pybaselines package
    # NOTE maybe replace with CUSUM or PELT https://www.researchgate.net/figure/Pruned-exact-linear-time-PELT-algorithm-overview_fig1_317273871
    # https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F317273871%2Ffigure%2Ffig3%2FAS%3A614128532090880%401523431060027%2FOutlier-detection-by-pruned-exact-linear-time-time-series-Kalman-filter-PELT-TSKF_Q640.jpg&f=1&nofb=1&ipt=21834f5ab10471f42864a2244c4a40dfcde9c17e4a6f8036ed8a5eff98ef36ae
    # NOTE See plateau_detection.ipynb
    histogram = (
        df_meas_data_w_median.lazy()
        .select(
            (
                (pl.col(val_col_smooth_long) / ROUNDING_INTERVAL).floor()
                * ROUNDING_INTERVAL
            ).alias("rounded")
        )  # compute rounded bin values and create "rounded" column
        .group_by("rounded")  # group by rounded bin values
        .len(name="count")  # count occurrences in each bin
        # add half bin width to get bin centers
        .with_columns(
            (pl.col("rounded").cast(pl.Float32) + (ROUNDING_INTERVAL / 2)).alias(
                "bin_center"
            )
        )
        .collect()
    )
    # histogram.write_parquet("histogram.parquet")

    # get the two most common bins
    base_level, peak_level = histogram.top_k(2, by="count")["bin_center"]
    # TODO remove magic numbers of 0.5 watt and 75 %
    # the 0.5 watt is to ensure that we are really looking at the "off" state for the base level and not a bin that includes some "on" values, and the 75% is to ensure that we are really looking at the "on" state for the peak level and not a bin that includes some "off" values
    # the 75% is somewhat arbitrary but it is to ensure that we are looking at a bin that is mostly "on" values for the peak level, and not a bin that includes a significant amount of "off" values
    activation = base_level + 0.75 * np.ptp(
        df_meas_data_w_median.lazy()
        .select(val_col_smooth_long)
        .filter(abs(pl.col(val_col_smooth_long).round(2) - base_level) < 0.5)
        .collect()
    )

    # threshold = (base_level + peak_level) / 2
    threshold = base_level + 0.05 * (base_level + peak_level)

    if make_plots:
        fig = plot_power_and_histogram(
            df_meas_data_w_median, histogram, threshold, base_level, peak_level
        )
        fig.savefig("grafiek_vermogen_histogram.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # with the threshold, segment the time series into temperature control cycles TCC
    # (a cycle of the compressor being on and off)
    # and filter out cycles that are shorter than 5 minutes to remove noise
    # first create a column that indicates when the compressor is on or off based on the threshold,
    # then find the transitions from off to on and from on to off,
    # and then filter out cycles that are shorter than 5 minutes
    # give each cycle a unique id and compute the start and end time of each cycle and the duration of each cycle
    # NOTE this approach assumes that the compressor is either fully on or fully off,
    # and that there are no intermediate states. If there are intermediate states,
    # a more complex approach may be needed to segment the cycles accurately.
    df_TCC = (
        df_meas_data_w_median.lazy()
        .select(
            [
                pl.col(time_col),
                (pl.col(val_col_smooth_short) >= activation).cast(pl.Int32).alias("is_on"),
            ]
        )
        .with_columns(pulse_transitions_pos_neg=pl.col("is_on").diff(-1))
        .filter(pl.col("pulse_transitions_pos_neg") != 0)
        .sort("pulse_transitions_pos_neg")
        .with_columns(
            time_diff_btwn_pulses_of_same_orientation=pl.col("Datetime").diff(1).shift(-1)
        )
        .filter(pl.col("time_diff_btwn_pulses_of_same_orientation") > timedelta(minutes=5))
        .drop("time_diff_btwn_pulses_of_same_orientation", "pulse_transitions_pos_neg")
        .sort("Datetime")
        .filter(pl.col("is_on").diff(1) != 0)
        .with_columns(pl.col("Datetime").shift(-1).alias("pulse_end_datetime"))
        .rename({"Datetime": "TCC_start_datetime"})
        .filter(pl.col("is_on") == 1)
        .drop("is_on")
        .with_row_index("TCC_id")
        .with_columns(
            TCC_duration=pl.col("TCC_start_datetime").diff().shift(-1),
            TCC_end_datetime=pl.col("TCC_start_datetime").shift(-1),
        )
        .with_columns(
            TCC_on_duration=pl.col("TCC_duration")
            - (pl.col("pulse_end_datetime") - pl.col("TCC_start_datetime")),
        )
        .drop_nulls()
        .collect()
    )
    # df_TCC.write_parquet("df_TCC.parquet")

    # now let's join the TCC with the original data to get the temperature data for each cycle, and compute time-weighted means for each sensor for each cycle
    lf_meas_data_w_smooth_pre_TCC_join = df_meas_data_w_median.lazy().with_columns(
        pl.col("Datetime").alias("bucket")
    )
    lf_TCC_pre_meas_data_join = df_TCC.lazy().with_columns(
        pl.col("TCC_start_datetime").alias("bucket")
    )

    start_time = time()
    # first we do a cross join to get all combinations of measurements and TCCs,
    # then we filter to keep only the measurements that fall within the start and end time of each TCC,
    # and then we compute the time-weighted means for each sensor for each TCC
    df_meas_data_w_smooth_w_TCC_id = (
        lf_meas_data_w_smooth_pre_TCC_join.join(
            lf_TCC_pre_meas_data_join, how="cross"
        )  # small-ish cross within bucket
        .filter(
            pl.col("Datetime").is_between(
                pl.col("TCC_start_datetime"), pl.col("TCC_end_datetime"), closed="both"
            )
        )
        .select(
            [
                "Datetime",
                "meas_duration",
                "TCC_id",
                "TCC_duration",
                *time_weighted_means_sensors,
            ]
        )
        .sort("Datetime")
        .collect()
    )
    # df_meas_data_w_smooth_w_TCC_id.write_parquet("df_meas_data_w_smooth_w_TCC_id.parquet")

    df_TCC_w_streaks_and_means = (
        df_TCC.lazy()
        .join(
            df_meas_data_w_smooth_w_TCC_id.lazy()
            .with_columns(is_defrost_cycle=pl.col(power_col) > thresh_w)
            .group_by("TCC_id")
            .agg(
                [
                    pl.any("is_defrost_cycle"),
                    *[
                        weighted_mean_expr(s, "meas_duration")
                        for s in time_weighted_means_sensors
                    ],
                ]
            )
            .sort("TCC_id")
            .with_columns(pl.col("is_defrost_cycle").cum_sum().alias("streak_id")),
            on="TCC_id",
        )
        .sort("TCC_id")
        # ref A
        .with_columns(
            [
                pl.mean_horizontal(
                    [pl.col(s) for s in comp_details["temp sensors"]]
                ).alias(f"temp_mean_{comp}")
                for comp, comp_details in [
                    *compartments.items(),
                    ("amb", {"temp sensors": ambient_temp_sensors}),
                ]
            ]
        )
        .collect()
    )
    df_TCC_w_streaks_and_means.write_parquet("df_TCC_w_streaks_and_means.parquet")

    print(
        f"TCC joining and weighted means computation took {time() - start_time:.5f} seconds"
    )

    if make_plots:
        fig = plot_detected_cycles(df_TCC, df_meas_data_w_median, threshold)
        fig.savefig("grafiek_tcc_cycli.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    plot_data = {
        "df_meas_data_w_median": df_meas_data_w_median,
        "histogram": histogram,
        "threshold": threshold,
        "base_level": base_level,
        "peak_level": peak_level,
    }
    return df_TCC, df_TCC_w_streaks_and_means, df_meas_data_w_smooth_w_TCC_id, plot_data


def _ss1_power_spread_limit_expr() -> pl.Expr:
    hours = pl.col("period_duration").dt.total_seconds() / 3600
    return (
        pl.when(pl.col("period_duration") <= timedelta(hours=12))
        .then(1.0)
        .when(pl.col("period_duration") < timedelta(hours=36))
        .then(1.0 + (hours - 12.0) / 1200.0)
        .otherwise(3.0)
    )


def annotate_ss1_periods(df: pl.DataFrame) -> pl.DataFrame:
    """Voeg ISO 16358 / Annex B-vinkjes en faalredenen toe aan alle 3-blok-perioden."""
    if df.height == 0:
        return df
    spread_cols = [c for c in df.columns if str(c).startswith("temp_spread_")]
    slope_cols = [c for c in df.columns if str(c).startswith("temp_slope_")]
    lf = df.lazy().with_columns(
        period_hours=pl.col("period_duration").dt.total_seconds() / 3600.0,
        n_TCC=pl.col("N_TCC_per_block") * 3,
        power_spread_limit_pct=_ss1_power_spread_limit_expr(),
    )
    lf = lf.with_columns(
        ok_duration=pl.col("period_duration") >= timedelta(hours=6),
        ok_temp_spread=(
            pl.all_horizontal(pl.col(c) < 0.25 for c in spread_cols)
            if spread_cols
            else pl.lit(True)
        ),
        ok_power_spread=pl.col("power_spread_pct") <= pl.col("power_spread_limit_pct"),
        ok_power_slope=pl.col("power_slope_pct").abs() < 0.25,
        max_temp_spread_K=(
            pl.max_horizontal(pl.col(c) for c in spread_cols) if spread_cols else pl.lit(None)
        ),
        max_temp_slope_K_per_h=(
            pl.max_horizontal(pl.col(c).abs() for c in slope_cols)
            if slope_cols
            else pl.lit(None)
        ),
    )
    lf = (
        lf.sort(["streak_id", "N_TCC_per_block", "start_TCC_id"])
        .with_columns(
            prev_two_periods_also_valid=(
                pl.col("start_TCC_id")
                == (
                    pl.col("start_TCC_id")
                    .shift(1)
                    .over("N_TCC_per_block", "streak_id")
                    + 1
                )
            )
            & (
                pl.col("start_TCC_id")
                == (
                    pl.col("start_TCC_id")
                    .shift(2)
                    .over("N_TCC_per_block", "streak_id")
                    + 2
                )
            )
        )
        .with_columns(
            pl.col("prev_two_periods_also_valid").fill_null(False),
        )
        .with_columns(
            ok_ss1=(
                pl.col("ok_duration")
                & pl.col("ok_temp_spread")
                & pl.col("ok_power_spread")
                & pl.col("ok_power_slope")
                & pl.col("prev_two_periods_also_valid")
            )
        )
        .with_columns(
            fail_reason=pl.concat_str(
                [
                    pl.when(~pl.col("ok_duration"))
                    .then(
                        pl.format(
                            "duur {} h < 6 h",
                            pl.col("period_hours").round(2),
                        )
                    )
                    .otherwise(pl.lit("")),
                    pl.when(~pl.col("ok_temp_spread"))
                    .then(
                        pl.format(
                            "temp-spread {} K ≥ 0,25 K",
                            pl.col("max_temp_spread_K").round(3),
                        )
                    )
                    .otherwise(pl.lit("")),
                    pl.when(~pl.col("ok_power_spread"))
                    .then(
                        pl.format(
                            "P-spread {} % > limiet {} %",
                            pl.col("power_spread_pct").round(3),
                            pl.col("power_spread_limit_pct").round(3),
                        )
                    )
                    .otherwise(pl.lit("")),
                    pl.when(~pl.col("ok_power_slope"))
                    .then(
                        pl.format(
                            "|P-helling| {} %/h ≥ 0,25 %/h",
                            pl.col("power_slope_pct").abs().round(4),
                        )
                    )
                    .otherwise(pl.lit("")),
                    pl.when(~pl.col("prev_two_periods_also_valid"))
                    .then(pl.lit("start−1 en start−2 niet beide geldig"))
                    .otherwise(pl.lit("")),
                ],
                separator="; ",
            )
        )
        .with_columns(
            fail_reason=pl.col("fail_reason").str.replace_all(r"(^; )|(; $)|(; ){2,}", "; ").str.strip_chars("; ")
        )
        .with_columns(
            fail_reason=pl.when(pl.col("ok_ss1"))
            .then(pl.lit("voldoet"))
            .otherwise(pl.col("fail_reason"))
        )
        .sort(
            [
                pl.col("ok_ss1"),
                pl.col("ok_duration"),
                pl.col("ok_power_slope"),
                pl.col("ok_temp_spread"),
                pl.col("period_hours"),
            ],
            descending=[True, True, True, True, True],
        )
    )
    return lf.collect()


def ss1_diagnostics_summary(diag: pl.DataFrame) -> dict:
    n = diag.height
    if n == 0:
        return {"n_periods": 0}
    return {
        "n_periods": n,
        "n_ok_ss1": int(diag["ok_ss1"].sum()),
        "n_ok_duration": int(diag["ok_duration"].sum()),
        "n_ok_temp_spread": int(diag["ok_temp_spread"].sum()),
        "n_ok_power_spread": int(diag["ok_power_spread"].sum()),
        "n_ok_power_slope": int(diag["ok_power_slope"].sum()),
        "n_ok_prev_two": int(diag["prev_two_periods_also_valid"].sum()),
        "max_hours": float(diag["period_hours"].max() or 0.0),
    }


def analyze_steady_state(
    df_TCC_w_streaks_and_means: pl.DataFrame,
    nominal_ambient: float | None = None,
) -> tuple[dict, float, pl.DataFrame, float, pl.DataFrame]:
    """Bereken alle kandidaat-perioden en selecteer de eerst afgeronde geldige SS1 plus pss.

    Een periode is drie opeenvolgende blokken van n TCC's. Van de kandidaten die aan
    de stabiliteitscriteria voldoen (inclusief de twee perioden die 1 en 2 TCC's
    eerder starten) kiezen we de periode die het vroegst eindigt: de kortste tijd
    vanaf de start van de meting tot een geldige SS1. Zo zie je of er lang genoeg
    gemeten is. De Python-lus over blokken blijft eager (caching per (start, n));
    de SS1-filters lopen daarna via een LazyFrame.
    """
    # now a period selecection algorithm that loops through all possible periods defined by three consecutive blocks of TCCs,
    #  where each block consists of k TCCs, and computes the statistics for each period using the get_period_stats function
    # defined above. The results are stored in a list of dictionaries, which is then converted to a dataframe for further analysis.
    # I think the ISO 16358-1 standard suggests to just loop through the periods and find the first one you find that meets the criteria for a steady state period, and then use that period for the rest of the analysis,
    # but since the data is already loaded and we have the get_period_stats function defined, it is not much more computationally expensive to compute the statistics for all possible periods and then filter for the desired periods based on the computed statistics,

    # the following list of references to columns is used in the loop below to compute the statistics
    # for each period defined by three consecutive blocks of TCCs, where each block consists of k TCCs.
    # By defining this list of expressions once, we can reuse it for each block without having to
    # rewrite the same expressions multiple times, which makes the code cleaner and more efficient.
    _relevant_cols = [
        *[weighted_mean_expr(col, "TCC_duration") for col in mean_temp_cols],
        weighted_mean_expr(power_col, "TCC_duration"),
        pl.col("TCC_duration").sum().alias("block_duration"),
        (
            pl.col("TCC_start_datetime").min()
            + (pl.col("TCC_end_datetime").max() - pl.col("TCC_start_datetime").min()) / 2
        ).alias("block_time_midpoint"),
        pl.col("TCC_on_duration").sum().alias("block_total_on_duration"),
        pl.col("TCC_start_datetime").min().alias("block_TCC_start_datetime"),
        pl.col("TCC_end_datetime").max().alias("block_TCC_end_datetime"),
    ]

    start_time = time()
    all_stats_list = []
    for streak_id in (
        df_TCC_w_streaks_and_means.lazy()
        .unique("streak_id")
        .select("streak_id")
        .collect()["streak_id"]
        .to_list()
    ):
        print(f"now doing {streak_id=}")
        df_TCC_w_sam_slice = (
            df_TCC_w_streaks_and_means.lazy()
            .filter(pl.col("streak_id") == streak_id)
            .collect()
        )
        # remove the first two and last TCC since they are likely part of a defrost cycle
        # a lone defrost TCC is its own streak and cannot form a 3-block period
        if df_TCC_w_sam_slice.height < 4:
            print(f"skipping {streak_id=}: only {df_TCC_w_sam_slice.height} TCC(s)")
            continue
        df_TCC_w_sam_slice = df_TCC_w_sam_slice.sort("TCC_id").slice(
            2, df_TCC_w_sam_slice.height - 3
        )

        # the stats_cache is a dictionary that stores the computed statistics for each block of TCCs,
        #  so that we don't have to recompute the same statistics multiple times when they are used in multiple periods.
        #  The key of the dictionary is a tuple of (start_TCC_id, k),
        #  where start_TCC_id is the id of the first TCC in the block and k is the number of TCCs in the block.
        #  The value of the dictionary is the dataframe with the computed statistics for that block of TCCs.
        stats_cache = {}

        min_TCC_id, max_TCC_id = (
            df_TCC_w_sam_slice["TCC_id"].min(),
            df_TCC_w_sam_slice["TCC_id"].max(),
        )
        for start_TCC_id in range(int(min_TCC_id), int(max_TCC_id) + 1):
            n_tcc = int(max_TCC_id) - start_TCC_id + 1
            max_k = n_tcc // 3
            if max_k < 1:
                break
            for k in range(1, max_k + 1):
                s: int = start_TCC_id - min_TCC_id  # type: ignore

                if (start_TCC_id, k) not in stats_cache:
                    stats_cache[(start_TCC_id, k)] = df_TCC_w_sam_slice.slice(s, k).select(
                        _relevant_cols
                    )
                if (start_TCC_id + k, k) not in stats_cache:
                    stats_cache[(start_TCC_id + k, k)] = df_TCC_w_sam_slice.slice(
                        s + k, k
                    ).select(_relevant_cols)
                if (start_TCC_id + 2 * k, k) not in stats_cache:
                    stats_cache[(start_TCC_id + 2 * k, k)] = df_TCC_w_sam_slice.slice(
                        s + 2 * k, k
                    ).select(_relevant_cols)

                # Compute stats for primary period and for the previous periods
                blocks_in_curr_period = [
                    stats_cache[(start_TCC_id, k)],
                    stats_cache[(start_TCC_id + k, k)],
                    stats_cache[(start_TCC_id + 2 * k, k)],
                ]
                stats = get_period_stats(*blocks_in_curr_period)
                stats["streak_id"] = streak_id
                stats["id"] = (start_TCC_id, k)
                stats["start_TCC_id"] = start_TCC_id
                stats["N_TCC_per_block"] = k
                all_stats_list.append(stats)

    n_tcc = df_TCC_w_streaks_and_means.height
    n_streaks = df_TCC_w_streaks_and_means["streak_id"].n_unique()
    if not all_stats_list:
        raise SS1NotFoundError(
            "Geen SS1-perioden kunnen worden gevormd. "
            f"{n_tcc} compressor-cyclus/cycli in {n_streaks} streak(s). "
            "Een periode is 3 blokken van n cycli; na het weglaten van defrost-randen "
            "zijn er te weinig aaneengesloten cycli (minstens 3).",
            diagnostics=pl.DataFrame(),
            summary={"n_periods": 0, "n_tcc": n_tcc, "n_streaks": n_streaks},
        )

    # convert the all_stats_list to a dataframe
    df_all_possible_periods_w_stats = pl.from_dicts(
        all_stats_list,
        infer_schema_length=1,
        schema_overrides={
            "period_duration": pl.Duration("ms"),
            "period_start_datetime": pl.Datetime("ms"),
        },
    )
    print(f"Period stat computations took {time() - start_time:.5f} seconds")
    # print(df_all_possible_periods_w_stats.columns)

    start_time = time()
    # filter for ss1 using criteria from ISO 16358 standard
    ss1_candidates = (
        df_all_possible_periods_w_stats.lazy()
        .filter(pl.col("period_duration") >= timedelta(hours=6))
        .filter(pl.all_horizontal(cs.starts_with("temp_spread") < 0.25))
        .filter(
            (
                pl.when(pl.col("period_duration") <= timedelta(hours=12))
                .then(pl.col("power_spread_pct") <= 1)
                .when(
                    (pl.col("period_duration") > timedelta(hours=12))
                    & (pl.col("period_duration") < timedelta(hours=36))
                )
                .then(
                    pl.col("power_spread_pct")
                    <= (
                        1
                        + (
                            (pl.col("period_duration").dt.total_seconds() / 3600 - 12)
                            / 1200
                        )
                    )
                )
                .otherwise(pl.col("power_spread_pct") <= 3)
            )
        )
        .filter(pl.col("power_slope_pct").abs() < 0.25)
        # determine if periods that start 1 tcc and 2 tcc earlier are also valid
        .sort(["N_TCC_per_block", "start_TCC_id"])
        .with_columns(
            # shift the previous two rows of s and compare if they are equal to the s of the current row minus 1 and minus 2
            prev_two_periods_also_valid=(
                pl.col("start_TCC_id")
                == (
                    pl.col("start_TCC_id")
                    .shift(1)
                    .over("N_TCC_per_block")
                    .over("streak_id")
                    + 1
                )
            )
            & (
                pl.col("start_TCC_id")
                == (
                    pl.col("start_TCC_id")
                    .shift(2)
                    .over("N_TCC_per_block")
                    .over("streak_id")
                    + 2
                )
            )
        )
        .filter(pl.col("prev_two_periods_also_valid"))
        .with_columns(
            period_end_datetime=pl.col("period_start_datetime") + pl.col("period_duration")
        )
        # Eerste geldige SS1: vroegste einde (kortste tijd vanaf de start van de meting).
        # Daarna kleinste n, daarna kortste duur. power_spread alleen als tiebreaker.
        .sort(
            [
                "period_end_datetime",
                "N_TCC_per_block",
                "period_duration",
                "power_spread",
            ]
        )
        .collect()
    )
    # ss1_candidates.write_parquet("ss1_candidates.parquet")
    # print(ss1_candidates)
    if ss1_candidates.height == 0:
        diag = annotate_ss1_periods(df_all_possible_periods_w_stats)
        summary = ss1_diagnostics_summary(diag)
        raise SS1NotFoundError(
            "Geen geldige SS1-periode gevonden. "
            f"{summary.get('n_periods', 0)} kandidaat-periode(n): "
            f"{summary.get('n_ok_duration', 0)} met duur ≥ 6 h, "
            f"{summary.get('n_ok_temp_spread', 0)} met temp-spread < 0,25 K, "
            f"{summary.get('n_ok_power_spread', 0)} met P-spread OK, "
            f"{summary.get('n_ok_power_slope', 0)} met |P-helling| < 0,25 %/h, "
            f"{summary.get('n_ok_prev_two', 0)} met geldige start−1/start−2. "
            "Zie de tabel onder Validatie stabiele periode (SS1).",
            diagnostics=diag,
            summary=summary,
        )
    opti = ss1_candidates.row(0, named=True)
    print(
        "SS1: n="
        f"{opti['N_TCC_per_block']} TCC/blok, duur={opti['period_duration']}, "
        f"start={opti['period_start_datetime']}, einde={opti['period_end_datetime']}"
    )

    print(f"ss11 filtering took {time() - start_time:.5f} seconds")
    mean_amb_temp = (
        df_TCC_w_streaks_and_means.lazy()
        .select(weighted_mean_expr("temp_mean_amb", "TCC_duration"))
        .collect()
        .item()
    )
    print(f"{mean_amb_temp = }")
    pss1 = opti["power_avg"]
    amb_target = target_ambient_temp if nominal_ambient is None else nominal_ambient

    teller = sum(
        compartments[compartment]["volume"]
        / (c1 * (18 + compartments[compartment]["target_temp"]) + c2)
        for compartment in compartments.keys()
    )
    noemer = sum(
        (
            compartments[compartment]["volume"]
            * (mean_amb_temp - opti[f"temp_mean_{compartment}"])
        )
        / (c1 * (18 + compartments[compartment]["target_temp"]) + c2)
        for compartment in compartments.keys()
    )

    print(f"{pss1 = }")
    pss = pss1 * (
        1
        + (
            (amb_target - mean_amb_temp)
            / (1 + (amb_target - mean_amb_temp) * delta_cop)
        )
        * teller
        / noemer
    )
    print(f"{pss = }")

    return opti, pss, df_all_possible_periods_w_stats, mean_amb_temp, ss1_candidates


def analyze_defrost(
    df_TCC_w_streaks_and_means: pl.DataFrame,
    df_all_possible_periods_w_stats: pl.DataFrame,
    df_meas_data_w_smooth_w_TCC_id: pl.DataFrame | None = None,
) -> tuple[float, pl.DataFrame, list[dict]]:
    """Bepaal D- en F-perioden rondom ontdooi-cycli en bereken de ontdooi-energie Delta_E_dfj.

    Filters op periodestats lopen lazy; de D/F-combinatie per ontdooi-cyclus blijft een
    Python-lus omdat elke cyclus eigen geldige tijdvensters heeft.
    """
    start_time = time()

    df_defrost_cycles = (
        df_TCC_w_streaks_and_means.lazy()
        .filter("is_defrost_cycle")
        .select("streak_id", "TCC_start_datetime", "TCC_end_datetime")
        .with_columns(
            (pl.col("TCC_start_datetime") + pl.lit(timedelta(hours=2))).alias(
                "nominal_center"
            )
        )
        .with_columns(
            (pl.col("nominal_center") - pl.lit(timedelta(hours=3))).alias("D_border"),
            (pl.col("nominal_center") + pl.lit(timedelta(hours=3))).alias("F_border"),
        )
        .collect()
    )

    # TODO dit klopt nog niet helemaal, zie pag. 45 van norm a.k.a. eq. 26
    # delta_t_df: defrost interval (td-f) — time between consecutive defrost cycle starts, in hours
    delta_t_df = (
        df_defrost_cycles.lazy()
        .with_columns(
            pl.col("TCC_start_datetime")
            .diff()
            .dt.total_hours(fractional=True)
            .alias("delta_t_defrost_h")
        )
        .select(
            ["streak_id", "TCC_start_datetime", "TCC_end_datetime", "delta_t_defrost_h"]
        )
        .drop_nulls()  # first row has no previous defrost, so no interval
        .collect()
    )
    print("Defrost intervals (td-f) in hours:")
    print(delta_t_df)
    print()

    df_all_possible_periods_w_stats_pre_df1_filter = (
        df_all_possible_periods_w_stats.lazy()
        .filter(pl.col("N_TCC_per_block") >= 3)
        .filter(pl.col("period_duration") > timedelta(hours=3))
        .filter(pl.all_horizontal(cs.starts_with("temp_spread") < 0.5))
        .filter((pl.col("power_spread") < 1) | (pl.col("power_spread_pct") < 2))
        .collect()
    )

    Delta_E_dfj_wh = 0.0
    df1_details: list[dict] = []
    # TODO I think this might break if the first TCC is a defrost cycle...
    for idx, defrost_cycle_row in enumerate(df_defrost_cycles.iter_rows(named=True)):
        post_defrost_streak_id = defrost_cycle_row["streak_id"]
        pre_defrost_streak_id = post_defrost_streak_id - 1

        df_D = (
            df_all_possible_periods_w_stats_pre_df1_filter.lazy()
            .filter(pl.col("streak_id") == pre_defrost_streak_id)
            .select(
                [
                    "period_duration",
                    "power_spread",
                    "N_TCC_per_block",
                    "start_TCC_id",
                    "period_start_datetime",
                ]
            )
            .rename(
                {
                    "period_duration": "D_period_duration",
                    "power_spread": "D_power_spread",
                    "start_TCC_id": "D_start_TCC_id",
                    "period_start_datetime": "D_period_start_datetime",
                }
            )
            .collect()
        )

        df_F = (
            df_all_possible_periods_w_stats_pre_df1_filter.lazy()
            .filter(pl.col("streak_id") == post_defrost_streak_id)
            .select(
                [
                    "period_duration",
                    "power_spread",
                    "N_TCC_per_block",
                    "start_TCC_id",
                    "period_start_datetime",
                ]
            )
            .rename(
                {
                    "period_duration": "F_period_duration",
                    "power_spread": "F_power_spread",
                    "start_TCC_id": "F_start_TCC_id",
                    "period_start_datetime": "F_period_start_datetime",
                }
            )
            .collect()
        )

        if df_D.height == 0 or df_F.height == 0:
            print(
                f"skipping defrost streak_id={post_defrost_streak_id}: "
                f"D periods={df_D.height}, F periods={df_F.height}"
            )
            continue

        start_valid_D = df_D.select("D_period_start_datetime").min().item() + timedelta(
            hours=5
        )
        end_valid_D = defrost_cycle_row["D_border"]

        start_valid_F = defrost_cycle_row["F_border"]

        df_D = (
            df_D.lazy()
            .filter(
                pl.col("D_period_start_datetime").is_between(
                    start_valid_D, end_valid_D, closed="both"
                )
            )
            .collect()
        )
        # df_F = df_F.filter(pl.col("period_start_datetime").is_between(start_valid_F, end_valid_F, closed="both"))
        df_F = (
            df_F.lazy()
            .filter(pl.col("F_period_start_datetime") >= start_valid_F)
            .collect()
        )

        optimal_combinations_of_D_and_F_periods = (
            df_D.lazy()
            .join(df_F.lazy(), on="N_TCC_per_block", how="inner")
            .filter(
                (pl.col("D_period_duration") / pl.col("F_period_duration")).is_between(
                    0.8, 1.25, closed="both"
                )
            )
            .with_columns(
                combined_power_spread=pl.col("D_power_spread") + pl.col("F_power_spread")
            )
            .sort("combined_power_spread")
            # .sort("N_TCC_per_block",descending=True)
            .collect()
        )

        if optimal_combinations_of_D_and_F_periods.height == 0:
            print(
                f"skipping defrost streak_id={post_defrost_streak_id}: "
                "no valid D/F combinations"
            )
            continue

        opti = optimal_combinations_of_D_and_F_periods.row(0, named=True)
        # first compute the accumulted energy
        delta_E_accum_wh = (
            df_TCC_w_streaks_and_means.lazy()
            .filter(
                pl.col("TCC_id").is_between(
                    opti["D_start_TCC_id"],
                    opti["F_start_TCC_id"] + 3 * opti["N_TCC_per_block"],
                    closed="both",
                )
            )
            .select(weighted_time_accumulation_expr_hours(power_col, "TCC_duration"))
            .collect()
            .item()
        )  # in Watt h

        # now the difference in avera power consumption
        P_SS_D = (
            df_TCC_w_streaks_and_means.lazy()
            .filter(
                pl.col("TCC_id").is_between(
                    opti["D_start_TCC_id"],
                    opti["D_start_TCC_id"] + 3 * opti["N_TCC_per_block"],
                    closed="both",
                )
            )
            .select(weighted_mean_expr(power_col, "TCC_duration"))
            .collect()
            .item()
        )  # in Watt
        P_SS_F = (
            df_TCC_w_streaks_and_means.lazy()
            .filter(
                pl.col("TCC_id").is_between(
                    opti["F_start_TCC_id"],
                    opti["F_start_TCC_id"] + 3 * opti["N_TCC_per_block"],
                    closed="both",
                )
            )
            .select(weighted_mean_expr(power_col, "TCC_duration"))
            .collect()
            .item()
        )  # in Watt

        t_start_D = opti["D_period_start_datetime"]
        t_end_F = opti["F_period_start_datetime"] + opti["F_period_duration"]
        delta_time_h = (t_end_F - t_start_D).total_seconds() / 60 / 60

        Delta_E_dfj_wh = delta_E_accum_wh - delta_time_h * (P_SS_D + P_SS_F) / 2
        print("=========================")
        print(f"{Delta_E_dfj_wh = }")
        print("=========================")

        d_end_id = opti["D_start_TCC_id"] + 3 * opti["N_TCC_per_block"]
        f_end_id = opti["F_start_TCC_id"] + 3 * opti["N_TCC_per_block"]
        d_slice = df_TCC_w_streaks_and_means.filter(
            pl.col("TCC_id").is_between(opti["D_start_TCC_id"], d_end_id, closed="both")
        )
        f_slice = df_TCC_w_streaks_and_means.filter(
            pl.col("TCC_id").is_between(opti["F_start_TCC_id"], f_end_id, closed="both")
        )
        df_win = df_TCC_w_streaks_and_means.filter(
            pl.col("TCC_id").is_between(
                opti["D_start_TCC_id"],
                f_end_id,
                closed="both",
            )
        )

        def _dur_h(frame: pl.DataFrame) -> float:
            if frame.height == 0:
                return 0.0
            return float(
                frame.select(pl.col("TCC_duration").dt.total_hours(fractional=True).sum()).item()
                or 0.0
            )

        def _on_h(frame: pl.DataFrame) -> float:
            if frame.height == 0:
                return 0.0
            return float(
                frame.select(
                    pl.col("TCC_on_duration").dt.total_hours(fractional=True).sum()
                ).item()
                or 0.0
            )

        def _temps(frame: pl.DataFrame) -> dict[str, float]:
            if frame.height == 0:
                return {}
            out: dict[str, float] = {}
            dur = _dur_h(frame)
            if dur <= 0:
                return out
            for col in mean_temp_cols + ["temp_mean_amb"]:
                if col not in frame.columns:
                    continue
                val = frame.select(weighted_mean_expr(col, "TCC_duration")).item()
                if val is not None:
                    out[col] = float(val)
            return out

        t_d = _temps(d_slice)
        t_f = _temps(f_slice)
        t_win = _temps(df_win)
        dur_d, dur_f = _dur_h(d_slice), _dur_h(f_slice)
        on_d, on_f, on_win = _on_h(d_slice), _on_h(f_slice), _on_h(df_win)
        crt_d = 100.0 * on_d / dur_d if dur_d else 0.0
        crt_f = 100.0 * on_f / dur_f if dur_f else 0.0
        delta_tdr_h = on_win - delta_time_h * (crt_d + crt_f) / 2.0 / 100.0
        dth: dict[str, float] = {}
        for col, t_avg in t_win.items():
            td = t_d.get(col)
            tf = t_f.get(col)
            if td is None or tf is None:
                continue
            dth[col] = delta_time_h * (t_avg - (td + tf) / 2.0)
        df1_details.append(
            {
                "defrost_start": str(defrost_cycle_row["TCC_start_datetime"]),
                "streak_id": post_defrost_streak_id,
                "n_tcc_per_block": int(opti["N_TCC_per_block"]),
                "D_start_TCC_id": int(opti["D_start_TCC_id"]),
                "F_start_TCC_id": int(opti["F_start_TCC_id"]),
                "t_start_D": str(t_start_D),
                "t_end_F": str(t_end_F),
                "delta_time_h": float(delta_time_h),
                "E_D_to_F_wh": float(delta_E_accum_wh),
                "P_SS_D": float(P_SS_D),
                "P_SS_F": float(P_SS_F),
                "Delta_E_dfj_wh": float(Delta_E_dfj_wh),
                "temps_D": t_d,
                "temps_F": t_f,
                "temps_D_to_F": t_win,
                "dth_kh": dth,
                "on_D_h": on_d,
                "on_F_h": on_f,
                "on_D_to_F_h": on_win,
                "CRt_D_pct": crt_d,
                "CRt_F_pct": crt_f,
                "delta_tdr_h": float(delta_tdr_h),
            }
        )

        if DEBUG3:
            assert df_meas_data_w_smooth_w_TCC_id is not None
            t, p = (
                df_meas_data_w_smooth_w_TCC_id["Datetime"][::100],
                df_meas_data_w_smooth_w_TCC_id[power_col][::100],
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axvspan(start_valid_D, end_valid_D, label="D", color="black", alpha=0.1)
            ax.axvspan(start_valid_F, max(t), label="F", color="black", alpha=0.1)
            ax.axvline(
                defrost_cycle_row["nominal_center"],
                label="nominal_center",
                color="red",
                linestyle="dashdot",
            )

            opti = optimal_combinations_of_D_and_F_periods.row(0, named=True)

            ax.axvspan(
                opti["D_period_start_datetime"],
                opti["D_period_start_datetime"] + opti["D_period_duration"],
                label="D",
                color="black",
                alpha=0.1,
            )
            ax.axvspan(
                opti["F_period_start_datetime"],
                opti["F_period_start_datetime"] + opti["F_period_duration"],
                label="D",
                color="black",
                alpha=0.1,
            )

            ax.plot(t, p)

            ax.set_xlim([min(t), max(t)])  # type: ignore
            format_dates(fig, ax)
            ax.set_ylabel("Power (W)")
            ax.set_xlabel("Time")
            # ax.legend(title="Power Data")
            plt.show()

    print(f"df1 filtering took {time() - start_time:.5f} seconds")
    return Delta_E_dfj_wh, delta_t_df, df1_details


if __name__ == "__main__":
    print(f"{compartments_used_for_period_selection=}")

    lf_meas_data = load_and_align_data(power_files, temp_file)

    (
        df_TCC,
        df_TCC_w_streaks_and_means,
        df_meas_data_w_smooth_w_TCC_id,
        _plot_data,
    ) = detect_tcc_cycles(lf_meas_data)

    ss1_opti, pss, df_all_possible_periods_w_stats, mean_amb_temp, _ss1_cands = (
        analyze_steady_state(df_TCC_w_streaks_and_means)
    )

    Delta_E_dfj_wh, delta_t_df, _df1_details = analyze_defrost(
        df_TCC_w_streaks_and_means,
        df_all_possible_periods_w_stats,
        df_meas_data_w_smooth_w_TCC_id,
    )

    daily_energy_consumption_at_this_temp, delta_t_mean = compute_daily_energy_kwh(
        pss, Delta_E_dfj_wh, delta_t_df
    )
    if delta_t_mean is None:
        print("No defrost interval available; daily energy uses pss only")

    # the average time it takes for a cycle in
    cycle_time_seconds = (
        df_TCC.lazy().select(pl.col("TCC_duration").mean()).collect().item().total_seconds()
    )

    # the average time in each cycle that the compressor is on during steady state
    running_time_percentage = (
        df_TCC_w_streaks_and_means.lazy()
        .select(pl.col("TCC_on_duration") / pl.col("TCC_duration"))
        .mean()
        .collect()
        .item()
    ) * 100

    print(
        f"Estimated daily energy consumption at {target_ambient_temp}°C: {daily_energy_consumption_at_this_temp} kWh/day"
    )
    print(f"Average cycle time: {cycle_time_seconds} seconds")
    print(
        f"Average percentage of time compressor is on during steady state: {running_time_percentage:.1f}%"
    )
