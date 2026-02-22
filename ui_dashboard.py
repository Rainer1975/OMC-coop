# ui_dashboard.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def _traffic(util: float) -> str:
    # thresholds: green <=80%, yellow <=100%, red >100%
    try:
        u = float(util)
    except Exception:
        u = 0.0
    if u <= 0.80:
        return "🟢"
    if u <= 1.00:
        return "🟡"
    return "🔴"

def render(ctx: dict) -> None:
    today = ctx["today"]
    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]
    is_overdue = ctx["is_overdue"]
    progress_percent = ctx["progress_percent"]
    compute_done_composition = ctx["compute_done_composition"]
    forecast_eta = ctx["forecast_eta"]
    capacity_summary = ctx.get("capacity_summary")
    week_window = ctx.get("week_window")
    employees = ctx.get("employees", [])

    st.title("Dashboard")
    st.caption("Reporting: Capacity/Workload + Liste + Portfolio-Forecast.")

    # =========================
    # Capacity / Workload
    # =========================
    with st.container(border=True):
        st.subheader("Capacity")

        mode = st.selectbox(
            "Zeitraum",
            options=["This week", "This month"],
            index=0,
            key="cap_mode",
        )

        if mode == "This week" and callable(week_window):
            ws, we = week_window(today)
        else:
            # month window
            try:
                ws = today.replace(day=1)
                # next month first day - 1
                if ws.month == 12:
                    n1 = ws.replace(year=ws.year + 1, month=1, day=1)
                else:
                    n1 = ws.replace(month=ws.month + 1, day=1)
                from datetime import timedelta

                we = n1 - timedelta(days=1)
            except Exception:
                ws, we = today, today

        st.caption(f"Window: {ws.isoformat()} → {we.isoformat()} (business days)")

        tasks_all = [s for s in visible_series() if is_task(s)]
        if callable(capacity_summary):
            cap_rows = capacity_summary(tasks_all, employees, today, ws, we)
        else:
            cap_rows = []

        if not cap_rows:
            st.info("No capacity data (no tasks or missing function).")
        else:
            # compact tiles (max 4 per row)
            cols = st.columns(4)
            for i, r in enumerate(cap_rows):
                c = cols[i % 4]
                emoji = _traffic(r.get("util", 0.0))
                owner = r.get("owner") or "Unassigned"
                planned = float(r.get("planned", 0.0))
                cap = float(r.get("capacity", 1.0))
                remaining = float(r.get("remaining", 0.0))
                overdue = int(r.get("overdue", 0) or 0)
                util_pct = int(round(100.0 * float(r.get("util", 0.0))))

                c.markdown(f"**{emoji} {owner}**")
                c.metric("Utilization", f"{util_pct}%")
                c.caption(f"Planned {planned:.1f} / Cap {cap:.1f} · Remaining {remaining:.1f} · Overdue {overdue}")

            with st.expander("Details", expanded=False):
                # show table + per-person task list
                df = pd.DataFrame(
                    [
                        {
                            "Owner": r.get("owner", ""),
                            "Planned": round(float(r.get("planned", 0.0)), 1),
                            "Done": round(float(r.get("done", 0.0)), 1),
                            "Remaining": round(float(r.get("remaining", 0.0)), 1),
                            "Capacity": round(float(r.get("capacity", 0.0)), 1),
                            "Util%": int(round(100.0 * float(r.get("util", 0.0)))),
                            "Tasks": int(r.get("tasks", 0) or 0),
                            "Overdue": int(r.get("overdue", 0) or 0),
                        }
                        for r in cap_rows
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

                # per owner task list
                by_owner = {}
                for s in tasks_all:
                    oid = getattr(s, "owner_id", "") or ""
                    if not oid:
                        on = getattr(s, "owner", "") or ""
                        oid = f"name:{on.lower()}" if on else "(unassigned)"
                    by_owner.setdefault(oid, []).append(s)

                for r in cap_rows:
                    oid = r.get("owner_id")
                    owner = r.get("owner")
                    lst = by_owner.get(oid, [])
                    if not lst:
                        continue
                    st.markdown(f"##### {owner}")
                    rows = []
                    for s in lst:
                        # overlap check
                        try:
                            if s.end < ws or s.start > we:
                                continue
                        except Exception:
                            continue
                        rows.append(
                            {
                                "Project": getattr(s, "project", ""),
                                "Theme": getattr(s, "theme", ""),
                                "Task": getattr(s, "title", ""),
                                "Start": getattr(s, "start", "").isoformat() if getattr(s, "start", None) else "",
                                "End": getattr(s, "end", "").isoformat() if getattr(s, "end", None) else "",
                                "META": "META" if getattr(s, "is_meta", False) else "",
                                "Overdue": "YES" if is_overdue(s, today) else "",
                                "State": getattr(s, "state", ""),
                            }
                        )
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
                "State": getattr(s,"state","PLANNED"),
                "Depends count": len(getattr(s,"predecessors",[]) or []),
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
