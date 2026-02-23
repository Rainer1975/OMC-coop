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
    compute_units_composition = ctx.get("compute_units_composition")
    forecast_eta_units = ctx.get("forecast_eta_units")
    series_total_units = ctx.get("series_total_units")
    capacity_summary = ctx.get("capacity_summary")
    week_window = ctx.get("week_window")
    employees = ctx.get("employees", [])

    st.title("Dashboard")
    st.caption("Reporting: Capacity/Workload + Delivery speed (Velocity) + Runway/Burndown.")

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

    st.caption('Open task (klick auf Titel)')
    q = st.text_input('Filter', placeholder='search title…', key='dash_open_filter')
    shown = tasks
    if q:
        qq = q.lower()
        shown = [s for s in tasks if qq in ((s.title or '') + ' ' + (s.project or '') + ' ' + (s.owner or '')).lower()]
    for s in shown[:25]:
        if st.button(str(s.title or ''), key=f"dash_open_{s.series_id}", use_container_width=True):
            ctx['open_detail'](s.series_id)
        st.caption(f"{s.project} · {s.theme} · {s.owner} · {s.start.isoformat()}→{s.end.isoformat()}")


    # =========================
    # Delivery speed + Runway (units-based)
    # =========================
    with st.container(border=True):
        st.subheader("Delivery speed & Runway")
        st.caption("Selbsterklärend: *Delivery speed* = quittierte Units/Tag (letzte 10 Arbeitstage). *Runway* = verbleibende Units + Forecast-Enddatum.")

        tasks_all = [s for s in visible_series() if is_task(s)]
        if not tasks_all or not callable(compute_units_composition) or not callable(forecast_eta_units):
            st.info("No velocity/burndown data (no tasks or missing functions).")
        else:
            # lightweight filters
            portfolios = sorted({(getattr(s, "portfolio", "") or "Default").strip() or "Default" for s in tasks_all})
            projects = sorted({(getattr(s, "project", "") or "").strip() for s in tasks_all if (getattr(s, "project", "") or "").strip()})
            themes = sorted({(getattr(s, "theme", "") or "").strip() for s in tasks_all if (getattr(s, "theme", "") or "").strip()})
            owners = sorted({(getattr(s, "owner", "") or "").strip() for s in tasks_all if (getattr(s, "owner", "") or "").strip()})

            f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
            f_port = f1.multiselect("Portfolio", portfolios, default=[], key="dash_v_port")
            f_proj = f2.multiselect("Project", projects, default=[], key="dash_v_proj")
            f_theme = f3.multiselect("Theme", themes, default=[], key="dash_v_theme")
            f_owner = f4.multiselect("Owner", owners, default=[], key="dash_v_owner")
            win = int(f5.number_input("Window (workdays)", min_value=5, max_value=30, value=10, step=1, key="dash_v_win"))

            def _ok(s):
                if f_port and (getattr(s, "portfolio", "") or "Default").strip() not in f_port:
                    return False
                if f_proj and (getattr(s, "project", "") or "").strip() not in f_proj:
                    return False
                if f_theme and (getattr(s, "theme", "") or "").strip() not in f_theme:
                    return False
                if f_owner and (getattr(s, "owner", "") or "").strip() not in f_owner:
                    return False
                return True

            tasks = [s for s in tasks_all if _ok(s)]
            if not tasks:
                st.info("No tasks match filters.")
            else:
                all_days, done_u, rem_u, _ = compute_units_composition(tasks)
                anchor, vel, eta, dq = forecast_eta_units(all_days, done_u, rem_u, today, win)

                # remaining today = value at last day <= today
                rem_today = None
                if all_days:
                    idx = 0
                    for i, d in enumerate(all_days):
                        if d <= today:
                            idx = i
                        else:
                            break
                    rem_today = float(rem_u[idx]) if idx < len(rem_u) else 0.0
                rem_today = float(rem_today or 0.0)

                scope = 0.0
                if callable(series_total_units):
                    scope = float(sum(series_total_units(s) for s in tasks))
                else:
                    scope = float(len(tasks))

                # data quality label
                if dq >= max(3, win // 2):
                    dq_label = "High"
                elif dq >= 2:
                    dq_label = "Med"
                else:
                    dq_label = "Low"

                m1, m2, m3, m4 = st.columns([2, 2, 2, 3])
                m1.metric("Remaining (units)", f"{rem_today:.1f}")
                m2.metric("Delivery speed", f"{vel:.2f} units/day")
                m3.metric("Forecast finish", eta.isoformat() if eta else "—")
                m4.caption(f"Scope: {scope:.1f} units · Data quality: {dq_label} (done-days in window: {dq}/{win})")

                if vel > 0 and rem_today > 0:
                    import math

                    days_needed = int(math.ceil(rem_today / vel))
                    st.caption(f"At current speed: ~{days_needed} workdays remaining.")
                elif rem_today <= 0:
                    st.caption("Nothing remaining (already burned down).")
                else:
                    st.caption("No recent completions → delivery speed is 0. Forecast is not possible.")

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
