# ui_dashboard.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import streamlit as st
import plotly.graph_objects as go  # ✅ FIX: required


def render(ctx: Dict[str, Any]) -> None:
    st.title("Delivery speed & Runway")

    today: date = ctx["today"]
    visible_series = ctx["visible_series"]
    build_burndown_series = ctx["build_burndown_series"]

    series = visible_series()

    if not series:
        st.info("No tasks available.")
        return

    # ---------------- Filter row ----------------
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])

    portfolios = sorted({getattr(s, "portfolio", "") for s in series if getattr(s, "portfolio", "")})
    projects = sorted({getattr(s, "project", "") for s in series if getattr(s, "project", "")})
    themes = sorted({getattr(s, "theme", "") for s in series if getattr(s, "theme", "")})
    owners = sorted({getattr(s, "owner", "") for s in series if getattr(s, "owner", "")})

    sel_port = c1.selectbox("Portfolio", ["All"] + portfolios)
    sel_proj = c2.selectbox("Project", ["All"] + projects)
    sel_theme = c3.selectbox("Theme", ["All"] + themes)
    sel_owner = c4.selectbox("Owner", ["All"] + owners)
    window = c5.number_input("Window (workdays)", min_value=5, max_value=30, value=10)

    # ---------------- Filtering ----------------
    filtered = []
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
        st.info("No matching tasks.")
        return

    # ---------------- Velocity / ETA ----------------
    pkg = build_burndown_series(filtered, today=today, window_business_days=window)

    remaining = pkg.get("remaining_units_day", [])
    velocity = pkg.get("velocity", 0.0)
    eta = pkg.get("eta", None)
    all_days = pkg.get("all_days", [])
    done = pkg.get("done_units_day", [])

    if remaining:
        remaining_now = remaining[-1]
    else:
        remaining_now = 0.0

    scope = sum(done) + remaining_now

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scope (units)", round(scope, 2))
    k2.metric("Remaining (units)", round(remaining_now, 2))
    k3.metric("Velocity", f"{round(velocity, 2)} u/day")
    k4.metric("ETA", eta.isoformat() if eta else "—")

    if velocity <= 0:
        st.warning("Velocity basiert auf sehr wenigen aktiven DONE-Tagen – Forecast instabil.")

    # ---------------- Chart ----------------
    if not all_days:
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=all_days,
            y=remaining,
            mode="lines",
            name="Remaining units",
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        xaxis_title="Date",
        yaxis_title="Units",
    )

    st.plotly_chart(fig, use_container_width=True)
