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

    st.title("Burndown")
    st.caption("Pro Task: berechneter Ideal-Verlauf vs. tatsächlicher Verlauf (done_days). Gesamtübersicht ist bewusst entfernt.")

    tasks = [s for s in visible_series() if is_task(s)]
    if not tasks:
        st.info("No tasks.")
        return

    # Sort: soonest end first, then owner/project/title
    tasks.sort(key=lambda s: (s.end, (s.owner or ""), (s.project or ""), (s.title or "").lower()))

    # Small controls (minimal)
    c1, c2, c3 = st.columns([2, 2, 6])
    only_active = c1.toggle("Only tasks not completed", value=False, key="bd_only_active")
    show_future_actual = c2.toggle("Show future actual line", value=False, key="bd_show_future_actual")

    if only_active:
        # completion is: progress 100 OR (done_days covers all days)
        # use progress_percent for consistency with your system
        tasks = [s for s in tasks if float(progress_percent(s)) < 100.0]
        if not tasks:
            st.info("No active (not completed) tasks.")
            return

    # Render each task chart
    for s in tasks:
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
            rem_today = max(0, total - done_today)

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
            m3.metric("Remaining (actual)", rem_today)
            m4.metric("Remaining (ideal today)", ideal_today)

            st.pyplot(fig, clear_figure=True)
