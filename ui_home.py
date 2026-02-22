from __future__ import annotations

import datetime as _dt

import streamlit as st

from core import (
    TaskSeries,
    is_appointment,
    is_completed,
    is_task,
    remaining_days,
)


def render(ctx) -> None:
    """Home dashboard.

    Fixes:
    - "Open" muss über ctx.open_detail() gehen (DETAIL ist kein Sidebar-Menüpunkt).
      Sonst springt die Navigation beim Rerun wieder auf HOME zurück.
    - Entfernt unnötige Querlinks (Burndown/Gantt Buttons am Home) – Übersichtlicher.
    """

    st.title("🏠 Home")

    today: _dt.date = getattr(ctx, "today", _dt.date.today())
    series: list[TaskSeries] = list(getattr(ctx, "series", []) or [])

    focus_mode = bool(getattr(ctx, "focus_mode", False))

    def _visible(s: TaskSeries) -> bool:
        if not focus_mode:
            return True
        if getattr(s, "is_meta", False):
            return False
        if is_appointment(s):
            return False
        if is_task(s) and is_completed(s, today):
            return False
        return True

    visible = [s for s in series if _visible(s)]
    tasks = [s for s in visible if is_task(s)]
    appts = [s for s in visible if is_appointment(s)]

    due_today = [t for t in tasks if getattr(t, "start", None) == today]
    overdue = [
        t
        for t in tasks
        if getattr(t, "end", None) is not None
        and getattr(t, "end") < today
        and not is_completed(t, today)
    ]
    active = [
        t
        for t in tasks
        if getattr(t, "start", None) is not None
        and getattr(t, "end", None) is not None
        and getattr(t, "start") <= today <= getattr(t, "end")
        and not is_completed(t, today)
    ]
    meta_open = [t for t in visible if getattr(t, "is_meta", False) and not is_completed(t, today)]
    appt_today = [a for a in appts if getattr(a, "start", None) == today]

    remaining = sum(max(0, remaining_days(t, today)) for t in tasks if not is_completed(t, today))

    st.caption("Wichtigste KPIs – mit Drilldown in Listen und Details.")

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Today", len(due_today))
    c2.metric("Overdue", len(overdue))
    c3.metric("Active", len(active))
    c4.metric("META open", len(meta_open))
    c5.metric("APPT today", len(appt_today))
    c6.metric("Velocity 7d", 0)
    c7.metric("Remaining", remaining)
    c8.metric("ETA", "—")

    st.markdown("---")

    # Quick drill buttons
    b1, b2, b3 = st.columns(3)
    b4, b5, b6 = st.columns(3)
    b1.button("Drill: Due today", key="home_drill_today", on_click=lambda: st.session_state.update({"home_drill": "today"}))
    b2.button("Drill: Overdue", key="home_drill_overdue", on_click=lambda: st.session_state.update({"home_drill": "overdue"}))
    b3.button("Drill: Active", key="home_drill_active", on_click=lambda: st.session_state.update({"home_drill": "active"}))
    b4.button("Drill: META open", key="home_drill_meta", on_click=lambda: st.session_state.update({"home_drill": "meta"}))
    b5.button("Drill: APPT today", key="home_drill_appt", on_click=lambda: st.session_state.update({"home_drill": "appt"}))
    b6.button("Drill: All", key="home_drill_all", on_click=lambda: st.session_state.update({"home_drill": "all"}))

    st.markdown("#### Drilldown")
    drill = st.session_state.get("home_drill", "today")
    drill = st.radio(
        "Drilldown",
        options=["today", "overdue", "active", "meta", "appt", "critical", "all"],
        format_func=lambda x: {
            "today": "Due today",
            "overdue": "Overdue",
            "active": "Active",
            "meta": "META open",
            "appt": "Appointments today",
            "critical": "Critical Path",
            "all": "All",
        }.get(x, x),
        index=["today", "overdue", "active", "meta", "appt", "critical", "all"].index(drill),
        horizontal=True,
        key="home_drill_radio",
    )
    st.session_state.home_drill = drill

    def _list_for_drill(key: str) -> list[TaskSeries]:
        if key == "today":
            return due_today
        if key == "overdue":
            return overdue
        if key == "active":
            return active
        if key == "meta":
            return meta_open
        if key == "appt":
            return appt_today
        if key == "critical":
            # Placeholder: falls du später echten Critical-Path machst.
            return tasks
        return tasks

    rows = _list_for_drill(drill)
    title = {
        "today": "Due today",
        "overdue": "Overdue",
        "active": "Active",
        "meta": "META open",
        "appt": "Appointments today",
        "critical": "Critical Path",
        "all": "All",
    }.get(drill, "")

    st.subheader(title)

    if not rows:
        st.info("Keine Einträge.")
        return

    # table
    st.dataframe(
        [
            {
                "Project": getattr(s, "project", ""),
                "Owner": getattr(s, "owner", ""),
                "Kind": getattr(s, "kind", ""),
                "Title": getattr(s, "title", ""),
                "Start": str(getattr(s, "start", "")),
                "End": str(getattr(s, "end", "")),
                "META": "META" if getattr(s, "is_meta", False) else "",
            }
            for s in rows
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.caption("Eintrag öffnen")
    for s in rows[:25]:
        sid = getattr(s, "series_id", "")
        if not sid:
            continue
        cols = st.columns([6, 1, 3])
        cols[0].markdown(
            f"**{getattr(s, 'title', '')}** — {getattr(s, 'project', '')} · {getattr(s, 'owner', '')}"
        )
        # on_click -> ctx.open_detail: zuverlässig trotz Sidebar-Navigation
        cols[1].button("Open", key=f"home_open_{sid}", on_click=ctx.open_detail, args=(sid,))
        cols[2].caption(f"{getattr(s, 'start', '')} → {getattr(s, 'end', '')}")


