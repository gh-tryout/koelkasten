"""EN 62552-3 energie: meetpunten combineren, interpolatie en jaarverbruik.

Beperking van deze module:
- geen ijsbereiding / specified auxiliaries (Eaux = 0)
- Edaily bij target altijd via interpolatie onder/boven doeltemperatuur (Annex E.3)
- CRT-ontdooi (Annex D.3) wordt uit de metingen geschat; andere types geven een waarschuwing
"""

from __future__ import annotations

from dataclasses import dataclass, field

COMPARTMENT_TARGETS_C: dict[str, float] = {
    "Pantry": 17.0,
    "Wine storage": 12.0,
    "Cellar": 12.0,
    "Fresh food": 4.0,
    "Chill": 2.0,
    "Zero-star": 0.0,
    "One-star": -6.0,
    "Two-star": -12.0,
    "Three-star": -18.0,
    "Four-star": -18.0,
}

FROZEN_TYPES = {"Zero-star", "One-star", "Two-star", "Three-star", "Four-star"}

# EN 62552-3 Europese regionale functie (182,5 dagen bij 16 °C + 182,5 dagen bij 32 °C)
DAYS_AT_16_C = 182.5
DAYS_AT_32_C = 182.5


def frozen_unfrozen(compartment_type: str) -> str:
    return "frozen" if compartment_type in FROZEN_TYPES else "unfrozen"


@dataclass
class TestPoint:
    """Eén energie-meetpunt (één omgeving + kouder/warmer dan target)."""

    name: str
    ambient_c: float
    pss_w: float
    temps_c: dict[str, float]
    crt_ss_pct: float
    vs_target: str = "onder"  # "onder" | "boven" (benadering van de knop)
    delta_edf_wh: float = 0.0
    delta_tdf_h: float | None = None
    delta_tdr_h: float = 0.0
    mean_amb_c: float | None = None
    setpoint: str = ""  # backward compatible alias

    @property
    def edaily_wh(self) -> float:
        """Formule (1)/(2): Edaily in Wh/24h."""
        if self.delta_tdf_h is None or self.delta_tdf_h <= 0:
            return 24.0 * self.pss_w
        return 24.0 * self.pss_w + 24.0 * self.delta_edf_wh / self.delta_tdf_h

    @property
    def edaily_kwh(self) -> float:
        return self.edaily_wh / 1000.0

    def all_at_or_below_target(self, targets: dict[str, float]) -> bool:
        return all(self.temps_c[name] <= targets[name] for name in targets)


def defrost_interval_crt(
    trt_h: float,
    crt_ss_pct: float,
    delta_tdr_h: float = 0.0,
    delta_tdh_h: float = 0.0,
    delta_tdxy_h: float | None = None,
) -> float:
    """Annex D.3 formule (26): elapsed defrost-interval uit compressor-run-time.

    Δtdf = (Δtrt − Δtdr − Δtdh) / (CRtSS/100) + Δtdxy
    """
    if crt_ss_pct <= 0:
        raise ValueError("CRtSS moet groter dan 0% zijn.")
    tdxy = delta_tdh_h if delta_tdh_h > 0 else (0.0 if delta_tdxy_h is None else delta_tdxy_h)
    elapsed = (trt_h - delta_tdr_h - delta_tdh_h) / (crt_ss_pct / 100.0) + tdxy
    return round(elapsed, 1)


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (var**0.5) / mean


CRT_CV_LIMIT = 0.10  # Annex D.3: CV van Δtrt < 10 % voor CRT-regelaar
ELAPSED_CV_LIMIT = 0.10  # Annex D.2: CV van elapsed interval < 10 % voor elapsed-time


@dataclass
class DefrostClassification:
    kind: str
    mean_trt_h: float | None
    cv_trt: float | None
    mean_elapsed_h: float | None
    cv_elapsed: float | None
    n: int
    warning: str | None = None


def classify_defrost_controller(
    tcrt_hours: list[float],
    elapsed_hours: list[float],
) -> DefrostClassification:
    """Bepaal ontdooitype uit gemeten compressortijd en elapsed tijd tussen ontdooiingen.

    CRT (D.3): CV van compressor-run-time Δtrt < 10 %.
    Elapsed time (D.2): CV van kloktijdinterval < 10 %.
    Anders: waarschijnlijk variable/demand-defrost.
    """
    cv_trt = coefficient_of_variation(tcrt_hours)
    cv_el = coefficient_of_variation(elapsed_hours)
    n = len(tcrt_hours)
    mean_trt = sum(tcrt_hours) / n if n else None
    mean_el = sum(elapsed_hours) / len(elapsed_hours) if elapsed_hours else None

    if n < 2:
        return DefrostClassification(
            kind="onvoldoende_data",
            mean_trt_h=mean_trt,
            cv_trt=cv_trt,
            mean_elapsed_h=mean_el,
            cv_elapsed=cv_el,
            n=n,
            warning=(
                "Te weinig volledige ontdooi-intervallen om het type te toetsen "
                "(minstens twee intervallen, liefst bij verschillende Tamb/instellingen)."
            ),
        )

    if cv_trt is not None and cv_trt < CRT_CV_LIMIT:
        return DefrostClassification(
            kind="compressor_run_time",
            mean_trt_h=mean_trt,
            cv_trt=cv_trt,
            mean_elapsed_h=mean_el,
            cv_elapsed=cv_el,
            n=n,
        )

    if cv_el is not None and cv_el < ELAPSED_CV_LIMIT:
        return DefrostClassification(
            kind="elapsed_time",
            mean_trt_h=mean_trt,
            cv_trt=cv_trt,
            mean_elapsed_h=mean_el,
            cv_elapsed=cv_el,
            n=n,
            warning=(
                "Waarschuwing: de kloktijd tussen ontdooiingen is bijna constant "
                f"(CV={cv_el:.1%}), terwijl de compressor-run-time meer varieert "
                f"(CV={cv_trt:.1%} indien bekend). Dit lijkt een elapsed-time-regelaar, "
                "geen compressor-run-time-regelaar."
            ),
        )

    return DefrostClassification(
        kind="variable",
        mean_trt_h=mean_trt,
        cv_trt=cv_trt,
        mean_elapsed_h=mean_el,
        cv_elapsed=cv_el,
        n=n,
        warning=(
            "Waarschuwing: compressor-run-time tussen ontdooiingen varieert te sterk "
            f"(CV={cv_trt:.1%} ≥ {CRT_CV_LIMIT:.0%}). Dit lijkt geen CRT-regelaar "
            "maar een variable/demand-defrost. Δtdf volgens D.3 is dan onbetrouwbaar."
        ),
    )


def defrost_event_start_indices(is_defrost: list[bool]) -> list[int]:
    """Index van de eerste TCC van elk ontdooi-event (opeenvolgende True's = één event)."""
    events: list[int] = []
    in_event = False
    for i, flag in enumerate(is_defrost):
        if flag and not in_event:
            events.append(i)
            in_event = True
        elif not flag:
            in_event = False
    return events


def defrost_intervals_from_cycles(
    is_defrost: list[bool],
    tcc_ids: list,
    starts: list,
    on_hours: list[float],
) -> tuple[list[float], list[float]]:
    """Compressor-run-time en kloktijd tussen opeenvolgende ontdooi-events.

    Opeenvolgende defrost-TCC's tellen als één event. Δtcrt is de som van
    TCC_on_duration vanaf de start van event j tot de start van event j+1.
    """
    events = defrost_event_start_indices(is_defrost)
    tcrt_hours: list[float] = []
    elapsed_hours: list[float] = []
    for a, b in zip(events, events[1:]):
        id0, id1 = tcc_ids[a], tcc_ids[b]
        on_sum = sum(
            float(h)
            for tid, h in zip(tcc_ids, on_hours)
            if tid is not None and id0 <= tid < id1 and h is not None
        )
        tcrt_hours.append(on_sum)
        elapsed_hours.append((starts[b] - starts[a]).total_seconds() / 3600.0)
    return tcrt_hours, elapsed_hours


def compute_ss2_cycle_energy(
    is_defrost: list[bool],
    tcc_ids: list,
    starts: list,
    duration_h: list[float],
    power_w: list[float],
    extra_means: dict[str, list] | None = None,
) -> dict | None:
    """SS2: gemiddeld vermogen over complete ontdooi-tot-ontdooi-intervallen.

    Edaily = 24 · P. De ontdooi-energie zit al in het interval, dus geen aparte
    ΔEdf/Δtdf-term. Vereist minstens twee ontdooi-events (één volledig interval).
    """
    events = defrost_event_start_indices(is_defrost)
    if len(events) < 2:
        return None
    extra_means = extra_means or {}
    intervals: list[dict] = []
    tot_e = 0.0
    tot_t = 0.0
    extra_acc = {k: 0.0 for k in extra_means}
    extra_w = {k: 0.0 for k in extra_means}

    for a, b in zip(events, events[1:]):
        id0, id1 = tcc_ids[a], tcc_ids[b]
        energy = 0.0
        dur = 0.0
        extra_e = {k: 0.0 for k in extra_means}
        extra_d = {k: 0.0 for k in extra_means}
        for i, tid in enumerate(tcc_ids):
            if tid is None or not (id0 <= tid < id1):
                continue
            dt = float(duration_h[i] or 0.0)
            pw = float(power_w[i] or 0.0)
            energy += pw * dt
            dur += dt
            for key, vals in extra_means.items():
                val = vals[i] if i < len(vals) else None
                if val is None:
                    continue
                extra_e[key] += float(val) * dt
                extra_d[key] += dt
        if dur <= 0:
            continue
        interval_temps = {
            key: extra_e[key] / extra_d[key] if extra_d[key] else None
            for key in extra_means
        }
        p_mean = energy / dur
        intervals.append(
            {
                "start": str(starts[a]),
                "end": str(starts[b]),
                "duration_h": dur,
                "energy_wh": energy,
                "p_mean_w": p_mean,
                "edaily_kwh": 24.0 * p_mean / 1000.0,
                "mean_amb_c": interval_temps.get("temp_mean_amb"),
                "temps": interval_temps,
            }
        )
        tot_e += energy
        tot_t += dur
        for key in extra_means:
            extra_acc[key] += extra_e[key]
            extra_w[key] += extra_d[key]

    if tot_t <= 0 or not intervals:
        return None
    p_mean = tot_e / tot_t
    overall_temps = {
        key: extra_acc[key] / extra_w[key] if extra_w[key] else None
        for key in extra_means
    }
    return {
        "n_events": len(events),
        "n_intervals": len(intervals),
        "duration_h": tot_t,
        "energy_wh": tot_e,
        "p_mean_w": p_mean,
        "edaily_kwh_measured": 24.0 * p_mean / 1000.0,
        "mean_amb_c": overall_temps.get("temp_mean_amb"),
        "temps": overall_temps,
        "intervals": intervals,
    }


def check_vs_target_label(
    temps: dict[str, float],
    targets: dict[str, float],
    vs_target: str,
) -> str | None:
    """Waarschuwing als de knop-benadering niet bij de gemeten temperaturen past."""
    names = [n for n in targets if n in temps]
    if not names:
        return None
    all_le = all(temps[n] <= targets[n] for n in names)
    all_gt = all(temps[n] > targets[n] for n in names)
    if vs_target == "onder" and all_gt:
        return (
            "Instelling is 'kouder dan target', maar alle gemeten compartimenttemperaturen "
            "liggen boven hun doeltemperatuur."
        )
    if vs_target == "boven" and all_le:
        return (
            "Instelling is 'warmer dan target', maar alle gemeten compartimenttemperaturen "
            "liggen op of onder hun doeltemperatuur. Interpolatie naar de target lukt dan niet."
        )
    return None


@dataclass
class InterpolationResult:
    valid: bool
    edaily_wh: float | None
    method: str
    interpolation_compartment: str | None = None
    slope_wh_per_k: float | None = None
    temps_c: dict[str, float] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    candidate_energies: dict[str, float] = field(default_factory=dict)


def linear_interpolate_case1(
    point1: TestPoint,
    point2: TestPoint,
    targets: dict[str, float],
) -> InterpolationResult:
    """Annex E.3: lineaire interpolatie tussen twee meetpunten bij dezelfde Tamb."""
    messages: list[str] = []
    if abs(point1.ambient_c - point2.ambient_c) > 1.0:
        return InterpolationResult(
            False,
            None,
            "ongeldig",
            messages=["Interpolatie E.3 alleen tussen punten bij dezelfde omgevingstemperatuur."],
        )

    names = list(targets.keys())
    e1, e2 = point1.edaily_wh, point2.edaily_wh
    valid_energies: dict[str, float] = {}
    valid_temps: dict[str, dict[str, float]] = {}

    for i_name in names:
        t1 = point1.temps_c[i_name]
        t2 = point2.temps_c[i_name]
        t_tar = targets[i_name]
        if abs(t1 - t2) > 4.0:
            messages.append(
                f"{i_name}: |T1−T2|={abs(t1 - t2):.2f} K > 4 K — geen interpolatie op dit compartiment."
            )
            if t1 <= t_tar and t2 <= t_tar:
                messages.append(f"{i_name}: beide punten liggen al op/onder target.")
            continue
        if abs(t2 - t1) < 1e-9:
            messages.append(f"{i_name}: geen temperatuurverschil tussen de punten.")
            continue
        fi = (t_tar - t1) / (t2 - t1)
        if fi < 0 or fi > 1:
            messages.append(
                f"{i_name}: target ligt niet tussen T1 ({t1:.2f}) en T2 ({t2:.2f}) (fi={fi:.3f})."
            )
            continue
        temps = {}
        all_ok = True
        for j_name in names:
            tj1 = point1.temps_c[j_name]
            tj2 = point2.temps_c[j_name]
            tj = tj1 + fi * (tj2 - tj1)
            temps[j_name] = tj
            if tj > targets[j_name]:
                all_ok = False
        if not all_ok:
            messages.append(
                f"{i_name} op target: een ander compartiment komt boven target uit."
            )
            continue
        e_tar = e1 + fi * (e2 - e1)
        valid_energies[i_name] = e_tar
        valid_temps[i_name] = temps
        messages.append(f"{i_name} op target: geldig, Edaily={e_tar:.1f} Wh/d.")

    if not valid_energies:
        return InterpolationResult(
            False,
            None,
            "ongeldig",
            messages=messages
            + [
                "Geen geldige interpolatie: nodig is één meting kouder dan target en "
                "één warmer, met de doeltemperatuur ertussen (geen extrapolatie, ΔT ≤ 4 K)."
            ],
        )

    best_name = min(valid_energies, key=lambda n: valid_energies[n])
    e_lin = valid_energies[best_name]
    t1 = point1.temps_c[best_name]
    t2 = point2.temps_c[best_name]
    slope = (e2 - e1) / (t2 - t1) if abs(t2 - t1) > 1e-9 else None
    return InterpolationResult(
        True,
        e_lin,
        "E.3 lineair",
        interpolation_compartment=best_name,
        slope_wh_per_k=slope,
        temps_c=valid_temps[best_name],
        messages=messages,
        candidate_energies=valid_energies,
    )


def resolve_ambient_edaily(
    points: list[TestPoint],
    targets: dict[str, float],
    ambient_c: float,
) -> InterpolationResult:
    """Edaily bij doeltemperatuur via interpolatie onder/boven target (Annex E.3)."""
    group = [p for p in points if abs(p.ambient_c - ambient_c) <= 1.0]
    if not group:
        return InterpolationResult(
            False,
            None,
            "ontbreekt",
            messages=[f"Geen meetpunt bij {ambient_c:.0f} °C."],
        )

    below = [p for p in group if p.vs_target == "onder"]
    above = [p for p in group if p.vs_target == "boven"]
    if not below or not above:
        return InterpolationResult(
            False,
            None,
            "ongeldig",
            messages=[
                f"Bij {ambient_c:.0f} °C zijn metingen kóuder én wármer dan de "
                f"doeltemperatuur nodig (nu: {len(below)} onder, {len(above)} boven)."
            ],
        )

    best: InterpolationResult | None = None
    for p1 in below:
        for p2 in above:
            result = linear_interpolate_case1(p1, p2, targets)
            if result.valid and (
                best is None
                or (result.edaily_wh is not None and result.edaily_wh < (best.edaily_wh or 1e18))
            ):
                best = result
    if best is not None:
        return best
    return InterpolationResult(
        False,
        None,
        "ongeldig",
        messages=[
            f"Onder/boven-punten bij {ambient_c:.0f} °C geven geen geldige interpolatie "
            "(doeltemperatuur niet tussen de gemeten T, of |ΔT| > 4 K)."
        ],
    )


def annual_energy_europe_kwh(edaily_16_kwh: float, edaily_32_kwh: float) -> float:
    """EN 62552-3 regionale functie, zonder Eaux (geen ijsbereiding)."""
    return DAYS_AT_16_C * edaily_16_kwh + DAYS_AT_32_C * edaily_32_kwh


def campaign_status(points: list[TestPoint], targets: dict[str, float]) -> list[dict]:
    """Checklist: interpolatie onder/boven target bij 16 °C en 32 °C."""

    def _count(amb: float, side: str) -> int:
        return sum(
            1
            for p in points
            if abs(p.ambient_c - amb) <= 1 and p.vs_target == side
        )

    n16_lo, n16_hi = _count(16, "onder"), _count(16, "boven")
    n32_lo, n32_hi = _count(32, "onder"), _count(32, "boven")
    r16 = resolve_ambient_edaily(points, targets, 16) if n16_lo and n16_hi else None
    r32 = resolve_ambient_edaily(points, targets, 32) if n32_lo and n32_hi else None
    return [
        {
            "id": "t16_lo",
            "ok": n16_lo > 0,
            "text": "16 °C — instelling kouder dan target",
            "detail": None if n16_lo else "Analyseer een 16 °C-meting met knop kouder dan doel.",
        },
        {
            "id": "t16_hi",
            "ok": n16_hi > 0,
            "text": "16 °C — instelling warmer dan target",
            "detail": None if n16_hi else "Analyseer een 16 °C-meting met knop warmer dan doel.",
        },
        {
            "id": "t32_lo",
            "ok": n32_lo > 0,
            "text": "32 °C — instelling kouder dan target",
            "detail": None if n32_lo else "Analyseer een 32 °C-meting met knop kouder dan doel.",
        },
        {
            "id": "t32_hi",
            "ok": n32_hi > 0,
            "text": "32 °C — instelling warmer dan target",
            "detail": None if n32_hi else "Analyseer een 32 °C-meting met knop warmer dan doel.",
        },
        {
            "id": "e16",
            "ok": bool(r16 and r16.valid),
            "text": "Edaily bij 16 °C via interpolatie (E.3)",
            "detail": None
            if (r16 and r16.valid)
            else (r16.messages[-1] if r16 else "Eerst beide 16 °C-punten analyseren."),
        },
        {
            "id": "e32",
            "ok": bool(r32 and r32.valid),
            "text": "Edaily bij 32 °C via interpolatie (E.3)",
            "detail": None
            if (r32 and r32.valid)
            else (r32.messages[-1] if r32 else "Eerst beide 32 °C-punten analyseren."),
        },
    ]
