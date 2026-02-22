# ui_dashboard.py
from __future__ import annotations

import pandas as pd
import streamlit as st

def render(ctx: dict) -> None:
    today = ctx["today"]
    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]
    is_overdue = ctx["is_overdue"]
    progress_percent = ctx["progress_percent"]
    compute_done_composition = ctx["compute_done_composition"]
    forecast_eta = ctx["forecast_eta"]

    st.title("Dashboard")
    st.caption("Reporting: Liste + Portfolio-Forecast.")

    tasks = [s for s in visible_series() if is_task(s)]
    if not tasks:
        st.info("No tasks.")
        return

    rows = []
    for s in tasks:
        rows.append(
            {
                "Project": s.project,
                "Theme": s.theme,
                "Owner": s.owner,
                "Owner ID": getattr(s, "owner_id", ""),
                "Task": s.title,
                "Start": s.start.isoformat(),
                "End": s.end.isoformat(),
                "META": "META" if s.is_meta else "",
                "Progress %": progress_percent(s),
                "Overdue": "YES" if is_overdue(s, today) else "",
                "State": getattr(s,"state","PLANNED"), "Depends count": len(getattr(s,"depends_on",[]) or []),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    fc_window = st.number_input("Forecast window (days)", min_value=3, max_value=30, value=7, step=1, key="dash_fc_win")
    all_days, done_port, rem_port, _ = compute_done_composition(tasks)
    anchor, vel, eta = forecast_eta(all_days, done_port, rem_port, today, int(fc_window))
    st.caption(
        f"Portfolio Forecast ETA: {eta.isoformat() if eta else '—'} · "
        f"Velocity(avg/day): {vel:.2f} · Anchor: {anchor.isoformat() if anchor else '—'}"
    )
