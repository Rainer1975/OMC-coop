# ui_burndown.py
from __future__ import annotations

from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def _day_range(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _total_days_inclusive(start: date, end: date) -> int:
    return max(1, (end - start).days + 1)


def _done_upto(s, day: date) -> int:
    dd = getattr(s, "done_days", set()) or set()
    return sum(1 for x in dd if s.start <= x <= day)


def _series_label(s) -> str:
    pf = (getattr(s, "portfolio", "") or "Default").strip() or "Default"
    return f"{pf} · {s.owner} · {s.project} · {s.title} ({s.start.isoformat()}→{s.end.isoformat()})"


def _plot_task_burndown(
    s,
    today: date,
    show_future_ideal: bool = True,
    show_future_actual_as_nan: bool = True,
):
    # X-axis days: full task duration
    days = _day_range(s.start, s.end)
    n = len(days)
    total = _total_days_inclusive(s.start, s.end)

    # Ideal: linear from total -> 0 across full duration
    # (inclusive endpoints)
    ideal = np.linspace(total, 0, n)

    # Actual remaining: total - done_upto(day)
    actual = []
    for d in days:
        rem = max(0, total - _done_upto(s, d))
        # Optional: hide future actual by breaking line after today
        if show_future_actual_as_nan and d > today:
            actual.append(np.nan)
        else:
            actual.append(rem)
    actual = np.array(actual, dtype=float)

    # Today marker if within range
    has_today = (s.start <= today <= s.end)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(days, ideal, marker="o", linestyle="-", label="Ideal (computed)")
    ax.plot(days, actual, marker="o", linestyle="-", label="Actual (done_days)")

    if has_today:
        ax.axvline(x=today, linestyle="--", linewidth=1.2, alpha=0.85)

    ax.set_title(s.title)
    ax.set_ylabel("Remaining done-days")
    ax.set_xlabel("Date")
    ax.set_ylim(bottom=0)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.autofmt_xdate()

    ax.legend(loc="upper right", frameon=True)

    return fig, days, ideal, actual, total


def render(ctx: dict) -> None:
    today = ctx["today"]
    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]
    progress_percent = ctx["progress_percent"]
    compute_units_composition = ctx.get("compute_units_composition")
    forecast_eta_units = ctx.get("forecast_eta_units")
    series_total_units = ctx.get("series_total_units")

    st.title("Burndown")
    st.caption("Selbsterklärend: *Remaining* (Units) sinkt, wenn ihr DONE quittiert. *Delivery speed* = quittierte Units/Tag (letzte 10 Arbeitstage).")

    tasks = [s for s in visible_series() if is_task(s)]
    if not tasks:
        st.info("No tasks.")
        return

    # =========================
    # Overview: Runway + Portfolio burndown (units-based)
    # =========================
    with st.container(border=True):
        st.subheader("Runway")
        st.caption("Units = Task-Weight (falls vorhanden) gleichmäßig über Arbeitstage verteilt; Fallback: 1 Unit pro Task.")

        if not callable(compute_units_composition) or not callable(forecast_eta_units):
            st.info("Units-based burndown not available (missing functions).")
        else:
            # Filters
            portfolios = sorted({(getattr(s, "portfolio", "") or "Default").strip() or "Default" for s in tasks})
            projects = sorted({(getattr(s, "project", "") or "").strip() for s in tasks if (getattr(s, "project", "") or "").strip()})
            themes = sorted({(getattr(s, "theme", "") or "").strip() for s in tasks if (getattr(s, "theme", "") or "").strip()})
            owners = sorted({(getattr(s, "owner", "") or "").strip() for s in tasks if (getattr(s, "owner", "") or "").strip()})

            f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
            f_port = f1.multiselect("Portfolio", portfolios, default=[], key="bd_u_port")
            f_proj = f2.multiselect("Project", projects, default=[], key="bd_u_proj")
            f_theme = f3.multiselect("Theme", themes, default=[], key="bd_u_theme")
            f_owner = f4.multiselect("Owner", owners, default=[], key="bd_u_owner")
            win = int(f5.number_input("Velocity window (workdays)", min_value=5, max_value=30, value=10, step=1, key="bd_u_win"))

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

            ftasks = [s for s in tasks if _ok(s)]
            if not ftasks:
                st.info("No tasks match filters.")
            else:
                all_days, done_u, rem_u, _ = compute_units_composition(ftasks)
                anchor, vel, eta, dq = forecast_eta_units(all_days, done_u, rem_u, today, win)

                # remaining today
                rem_today = 0.0
                if all_days:
                    idx = 0
                    for i, d in enumerate(all_days):
                        if d <= today:
                            idx = i
                        else:
                            break
                    rem_today = float(rem_u[idx]) if idx < len(rem_u) else 0.0

                scope = float(sum(series_total_units(s) for s in ftasks)) if callable(series_total_units) else float(len(ftasks))

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

                # Plot remaining units burndown
                if all_days and rem_u:
                    fig, ax = plt.subplots(figsize=(12, 3.8))
                    ax.plot(all_days, rem_u, marker="o", linestyle="-", label="Remaining (units)")
                    # show 'today' marker
                    if all_days[0] <= today <= all_days[-1]:
                        ax.axvline(x=today, linestyle="--", linewidth=1.2, alpha=0.85)
                    ax.set_title("Burndown (Remaining units)")
                    ax.set_ylabel("Units remaining")
                    ax.set_xlabel("Date")
                    ax.set_ylim(bottom=0)
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
                    fig.autofmt_xdate()
                    ax.legend(loc="upper right", frameon=True)
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Not enough data to plot burndown.")

    # Sort: soonest end first, then owner/project/title
    tasks.sort(key=lambda s: (s.end, (s.owner or ""), (s.project or ""), (s.title or "").lower()))

    # =========================
    # Per-task burndown (done-days) – keep for debugging / drilldown
    # =========================
    with st.expander("Per-task burndown (done-days) – Drilldown", expanded=False):
        # Small controls (minimal)
        c1, c2, c3 = st.columns([2, 2, 6])
        only_active = c1.toggle("Only tasks not completed", value=False, key="bd_only_active")
        show_future_actual = c2.toggle("Show future actual line", value=False, key="bd_show_future_actual")

        if only_active:
            # completion is: progress 100 OR (done_days covers all days)
            # use progress_percent for consistency with your system
            tasks2 = [s for s in tasks if float(progress_percent(s)) < 100.0]
            if not tasks2:
                st.info("No active (not completed) tasks.")
                return
        else:
            tasks2 = list(tasks)

        # Render each task chart
        for s in tasks2:
            label = _series_label(s)
            with st.expander(label, expanded=False):
                fig, days, ideal, actual, total = _plot_task_burndown(
                    s,
                    today=today,
                    show_future_ideal=True,
                    show_future_actual_as_nan=(not show_future_actual),
                )

                # Metrics
                done_today = _done_upto(s, min(today, s.end)) if today >= s.start else 0
                rem_today2 = max(0, total - done_today)

                if today < s.start:
                    ideal_today = total
                elif today > s.end:
                    ideal_today = 0
                else:
                    idx = (today - s.start).days
                    idx = max(0, min(idx, len(ideal) - 1))
                    ideal_today = int(round(float(ideal[idx])))

                m1, m2, m3, m4 = st.columns([2, 2, 2, 2])
                m1.metric("Total done-days", total)
                m2.metric("Done (cum.)", done_today)
                m3.metric("Remaining (actual)", rem_today2)
                m4.metric("Remaining (ideal today)", ideal_today)

                st.pyplot(fig, clear_figure=True)
