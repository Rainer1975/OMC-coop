# ui_dashboard.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st


def _status_badge(status: str) -> str:
    s = (status or "").lower()
    if s == "red":
        return "🔴"
    if s == "yellow":
        return "🟡"
    return "🟢"


def render(ctx: Dict[str, Any]) -> None:
    """Work dashboard: Capacity (Auslastung) + Delivery speed (Velocity) + Runway + Burndown."""

    today: date = ctx["today"]
    visible_series = ctx["visible_series"]
    build_burndown_series = ctx["build_burndown_series"]
    capacity_summary = ctx["capacity_summary"]
    employees = ctx["employees"]
    gloss_tip = ctx.get("gloss_tip", lambda _t: "")

    series = visible_series()
    st.title("Work dashboard")

    # ---------------- Filters (shared) ----------------
    st.subheader("Filter")
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 1])

    portfolios = sorted({getattr(s, "portfolio", "") for s in series if getattr(s, "portfolio", "")})
    projects = sorted({getattr(s, "project", "") for s in series if getattr(s, "project", "")})
    themes = sorted({getattr(s, "theme", "") for s in series if getattr(s, "theme", "")})
    owners = sorted({getattr(s, "owner", "") for s in series if getattr(s, "owner", "")})

    sel_port = f1.selectbox("Portfolio", ["All"] + portfolios, key="dash_f_port")
    sel_proj = f2.selectbox("Project", ["All"] + projects, key="dash_f_proj")
    sel_theme = f3.selectbox("Theme", ["All"] + themes, key="dash_f_theme")
    sel_owner = f4.selectbox("Owner", ["All"] + owners, key="dash_f_owner")
    window = int(f5.number_input("Window", min_value=5, max_value=30, value=10, help="Workdays for Velocity/ETA", key="dash_f_window"))

    filtered: List[Any] = []
    for s in series:
        if sel_port != "All" and getattr(s, "portfolio", "") != sel_port:
            continue
        if sel_proj != "All" and getattr(s, "project", "") != sel_proj:
            continue
        if sel_theme != "All" and getattr(s, "theme", "") != sel_theme:
            continue
        if sel_owner != "All" and getattr(s, "owner", "") != sel_owner:
            continue
        filtered.append(s)

    if not filtered:
        st.info("Keine passenden Einträge für die Filter.")
        return

    # ---------------- Capacity (Auslastung) ----------------
    st.subheader("Auslastung (Capacity)" + " " + gloss_tip("Capacity"), anchor="capacity")

    cap_col1, cap_col2 = st.columns([1, 3])
    cap_window = cap_col1.selectbox("Zeitraum", ["week", "month"], index=0, key="cap_window", help="Woche oder Monat")
    cap_per_day = float(cap_col1.number_input("Units/Tag (Default)", min_value=0.5, max_value=20.0, value=5.0, step=0.5, key="cap_per_day"))

    cap = capacity_summary(
        filtered,
        employees,
        today=today,
        window=cap_window,
        default_capacity_per_day=cap_per_day,
    )

    by_emp = cap.get("by_employee", {}) or {}
    rows = []
    for eid, r in by_emp.items():
        rows.append(
            {
                "Status": _status_badge(r.get("status")) + " " + (r.get("status") or ""),
                "Employee": r.get("name") or eid,
                "Planned": round(float(r.get("planned", 0.0) or 0.0), 2),
                "Done": round(float(r.get("done", 0.0) or 0.0), 2),
                "Remaining": round(float(r.get("remaining", 0.0) or 0.0), 2),
                "Capacity": round(float(r.get("capacity", 0.0) or 0.0), 2),
                "Utilization": round(float(r.get("utilization", 0.0) or 0.0), 2),
                "Overdue": int(r.get("overdue", 0) or 0),
            }
        )

    if rows:
        cap_col2.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("🟢 ok · 🟡 knapp · 🔴 überlast. Utilization = Planned/Capacity." + " " + gloss_tip("Utilization"))
    else:
        st.info("Keine Employees/Kapazitätsdaten verfügbar.")

    st.divider()

    # ---------------- Delivery speed & Runway ----------------
    st.subheader("Delivery speed & Runway" + " " + gloss_tip("Velocity") + gloss_tip("Runway"), anchor="runway")
    st.caption("Selbsterklärend: Velocity = quittierte Units/Tag (letzte N Arbeitstage). Runway = Remaining Units + ETA.")

    pkg = build_burndown_series(filtered, today=today, window_business_days=window)

    remaining = pkg.get("remaining_units_day", []) or []
    velocity = float(pkg.get("velocity", 0.0) or 0.0)
    eta = pkg.get("eta", None)
    all_days = pkg.get("all_days", []) or []
    done = pkg.get("done_units_day", []) or []

    remaining_now = float(remaining[-1]) if remaining else 0.0
    scope = float(sum(done)) + remaining_now

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scope (Units)", round(scope, 2))
    k2.metric("Remaining (Units)", round(remaining_now, 2))
    k3.metric("Velocity", f"{round(velocity, 2)} u/day")
    k4.metric("ETA", eta.isoformat() if eta else "—")

    if float(pkg.get("data_quality_days", 0) or 0) < max(3, window // 4):
        st.warning("Velocity basiert auf sehr wenigen aktiven DONE-Tagen – Forecast ist instabil.")

    if not all_days:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_days, y=remaining, mode="lines", name="Remaining units"))
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        xaxis_title="Date",
        yaxis_title="Units",
    )
    st.plotly_chart(fig, use_container_width=True)
