# ui_dashboard.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

import streamlit as st
import plotly.graph_objects as go


def _ampel(status: str) -> str:
    s = (status or "").lower().strip()
    if s == "red":
        return "🔴"
    if s == "yellow":
        return "🟡"
    return "🟢"


def _pct(x: float) -> str:
    try:
        return f"{round(float(x) * 100)}%"
    except Exception:
        return "—"


def render(ctx: Dict[str, Any]) -> None:
    """Work dashboard: Capacity (ampel per employee) + Delivery speed/runway."""

    today: date = ctx["today"]
    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]

    capacity_summary = ctx["capacity_summary"]
    build_burndown_series = ctx["build_burndown_series"]

    employees = ctx["employees"]
    open_detail = ctx["open_detail"]

    series = visible_series()

    st.title("Work dashboard")

    if not employees:
        st.info("Keine Employees angelegt. Ohne Employees gibt es keine Auslastungsanzeige.")
        return

    # =========================
    # 1) CAPACITY / WORKLOAD
    # =========================
    st.subheader(
        "Auslastung (Capacity)",
        help="Kapazität: wie viel Units pro Zeitraum realistisch geliefert werden können (z.B. Woche). Ampel = Planned/Capacity.",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    window = c1.selectbox("Zeitraum", ["week", "month"], index=0)
    cap_per_day = c2.number_input("Units/Tag (Default)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)

    # tasks only for capacity
    tasks = [s for s in series if is_task(s)]

    cap = capacity_summary(
        tasks,
        employees=employees,
        today=today,
        window=window,
        default_capacity_per_day=cap_per_day,
    )

    by_emp: Dict[str, Any] = cap.get("by_employee", {}) or {}

    if not by_emp:
        st.info("Keine Capacity-Daten.")
    else:
        rows: List[Tuple[str, str, str, float, float, float, float, int]] = []
        for _eid, r in by_emp.items():
            rows.append(
                (
                    _ampel(r.get("status")),
                    str(_eid),
                    str(r.get("name") or ""),
                    float(r.get("utilization") or 0.0),
                    float(r.get("capacity") or 0.0),
                    float(r.get("planned") or 0.0),
                    float(r.get("remaining") or 0.0),
                    int(r.get("overdue") or 0),
                )
            )

        # sort: red first, then yellow, then green; within by util desc
        color_rank = {"🔴": 0, "🟡": 1, "🟢": 2}
        # (ampel, eid, name, util, ...)
        rows.sort(key=lambda x: (color_rank.get(x[0], 9), -x[3], x[2].lower()))

        st.caption("Klick auf einen Namen öffnet eine Liste der Tasks dieser Person (im Detail klickbar).")

        # allow "click name" -> expand tasks list for that employee
        if "dash_open_emp" not in st.session_state:
            st.session_state.dash_open_emp = ""

        for ampel, eid, name, util, cap_u, planned, remaining, overdue in rows:
            left, mid, right = st.columns([3, 2, 4])
            with left:
                # Name is clickable (opens the task list expander on the right)
                if st.button(f"{ampel} {name}", key=f"dash_empbtn_{eid}"):
                    st.session_state.dash_open_emp = eid
                st.caption(f"Auslastung: {_pct(util)} · Overdue: {overdue}")
            with mid:
                st.metric("Planned", round(planned, 2))
                st.metric("Capacity", round(cap_u, 2))
            with right:
                st.progress(min(1.0, max(0.0, util)))
                st.caption(f"Remaining: {round(remaining, 2)} units")

                # quick list of tasks for this employee
                with st.expander(
                    f"Tasks von {name}",
                    expanded=(str(st.session_state.get("dash_open_emp") or "") == str(eid)),
                ):
                    # find employee id
                    # We only have name here; find first matching id in by_emp
                    # but better: list from tasks filter by owner display.
                    emp_tasks = [t for t in tasks if str(getattr(t, "owner", "")).strip() == name]
                    if not emp_tasks:
                        st.write("—")
                    else:
                        # show up to 20
                        for t in sorted(emp_tasks, key=lambda x: (getattr(x, "end"), getattr(x, "title")))[:20]:
                            b = st.button(
                                f"{t.title} · {t.start.isoformat()}→{t.end.isoformat()}",
                                key=f"dash_open_{t.series_id}",
                            )
                            if b:
                                open_detail(t.series_id)

    st.divider()

    # =========================
    # 2) DELIVERY SPEED & RUNWAY
    # =========================
    st.subheader(
        "Delivery speed & Runway",
        help="Delivery speed = quittierte Units/Tag (letzte N Arbeitstage). Runway = verbleibende Units + Forecast-Enddatum.",
    )

    # Filters for runway
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 1])

    portfolios = sorted({getattr(s, "portfolio", "") for s in tasks if getattr(s, "portfolio", "")})
    projects = sorted({getattr(s, "project", "") for s in tasks if getattr(s, "project", "")})
    themes = sorted({getattr(s, "theme", "") for s in tasks if getattr(s, "theme", "")})
    owners = sorted({getattr(s, "owner", "") for s in tasks if getattr(s, "owner", "")})

    sel_port = f1.multiselect("Portfolio", options=portfolios, default=[])
    sel_proj = f2.multiselect("Project", options=projects, default=[])
    sel_theme = f3.multiselect("Theme", options=themes, default=[])
    sel_owner = f4.multiselect("Owner", options=owners, default=[])
    window_days = f5.number_input("Window (workdays)", min_value=5, max_value=30, value=10)

    def _match(val: str, selected: List[str]) -> bool:
        if not selected:
            return True
        return val in selected

    filtered = []
    for s in tasks:
        if not _match(getattr(s, "portfolio", ""), sel_port):
            continue
        if not _match(getattr(s, "project", ""), sel_proj):
            continue
        if not _match(getattr(s, "theme", ""), sel_theme):
            continue
        if not _match(getattr(s, "owner", ""), sel_owner):
            continue
        filtered.append(s)

    if not filtered:
        st.info("Keine Tasks im Filter.")
        return

    pkg = build_burndown_series(filtered, today=today, window_business_days=int(window_days))

    remaining = pkg.get("remaining_units_day", []) or []
    velocity = float(pkg.get("velocity") or 0.0)
    eta = pkg.get("eta")
    all_days = pkg.get("all_days", []) or []
    done = pkg.get("done_units_day", []) or []

    remaining_now = float(remaining[-1]) if remaining else 0.0
    scope = float(sum(done)) + remaining_now

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scope (units)", round(scope, 2))
    k2.metric("Remaining (units)", round(remaining_now, 2))
    k3.metric("Velocity", f"{round(velocity, 2)} u/day")
    k4.metric("ETA", eta.isoformat() if eta else "—")

    if velocity <= 0.001:
        st.warning("Velocity basiert auf sehr wenigen aktiven DONE-Tagen – Forecast ist instabil.")

    if all_days and remaining:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=all_days,
                y=remaining,
                mode="lines",
                name="Remaining units",
            )
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320, xaxis_title="Date", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)
