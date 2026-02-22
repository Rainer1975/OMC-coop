# ui_today.py
from __future__ import annotations

import streamlit as st

def render(ctx: dict) -> None:
    today = ctx["today"]
    safe_container = ctx["safe_container"]
    segmented = ctx["segmented"]
    appointment_label = ctx["appointment_label"]
    open_detail = ctx["open_detail"]
    request_done_single = ctx["request_done_single"]

    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]
    is_appointment = ctx["is_appointment"]
    is_active = ctx["is_active"]
    is_completed = ctx["is_completed"]
    is_overdue = ctx["is_overdue"]

    st.title("Heute")
    if st.session_state.get("focus_mode"):
        st.caption("Focus mode: nur aktive Tasks (keine META, keine Termine).")
    else:
        st.caption("Operativ: schnell quittieren, Detail öffnen.")

    # Minimal, oma-taugliche Filter: standardmäßig nur das Nötigste sichtbar.
    view = "Tasks" if st.session_state.get("focus_mode") else "All"
    show_meta = False if st.session_state.get("focus_mode") else True
    q = ""

    with st.expander("Filter (optional)", expanded=False):
        c1, c2 = st.columns([2, 6])
        if not st.session_state.get("focus_mode"):
            view = c1.selectbox("Ansicht", ["All", "Tasks", "Overdue", "Appointments"], index=0, key="today_view_simple")
            show_meta = c2.toggle("META anzeigen", value=True, key="today_show_meta_simple")
        q = st.text_input("Suche", placeholder="Titel…", key="today_search_simple")

    def match(s) -> bool:
        if q and q.lower() not in (s.title or "").lower():
            return False
        if (not show_meta) and s.is_meta:
            return False
        if st.session_state.get("focus_mode"):
            if is_appointment(s):
                return False
            if is_task(s) and is_completed(s, today):
                return False
        return True

    items = [s for s in visible_series() if match(s)]

    todays_appts = [s for s in items if is_appointment(s) and (s.start <= today <= s.end)]
    due_today = [s for s in items if is_task(s) and is_active(s, today) and (today not in s.done_days) and (not is_completed(s, today))]
    overdue = [s for s in items if is_task(s) and is_overdue(s, today)]

    if view in ("All", "Appointments") and (not st.session_state.get("focus_mode")):
        st.subheader("Appointments today")
        if not todays_appts:
            st.caption("No appointments today.")
        else:
            todays_appts.sort(key=lambda s: ((getattr(s, "time_start", "") or "99:99"), (s.title or "").lower()))
            for s in todays_appts:
                with safe_container(border=s.is_meta):
                    l, m, r = st.columns([1, 8, 1], vertical_alignment="center")
                    l.markdown("📅")
                    m.markdown(f"**{appointment_label(s)}**  \n{s.project} · {s.theme} · {s.owner}{(' · META' if s.is_meta else '')}")
                    if r.button("›", key=f"today_open_appt_{s.series_id}"):
                        open_detail(s.series_id)

    if view in ("All", "Tasks"):
        st.subheader("Due today (tasks)")
        if not due_today:
            st.caption("Nothing due today.")
        for s in sorted(due_today, key=lambda x: (x.end, (x.owner or ""), (x.title or "").lower())):
            with safe_container(border=s.is_meta):
                l, m, r = st.columns([1, 8, 1], vertical_alignment="center")
                if l.button("◯", key=f"today_done_{s.series_id}"):
                    request_done_single(s.series_id, today, "Today: quick check-off")
                m.markdown(f"**{s.title}**  \n{s.project} · {s.theme} · {s.owner}{(' · META' if s.is_meta else '')}")
                if r.button("›", key=f"today_open_{s.series_id}"):
                    open_detail(s.series_id)

    if view in ("All", "Overdue") and (not st.session_state.get("focus_mode")):
        st.subheader("Overdue (tasks)")
        if not overdue:
            st.caption("No overdue items.")
        for s in sorted(overdue, key=lambda x: (x.end, (x.owner or ""), (x.title or "").lower())):
            with safe_container(border=True):
                l, m, r = st.columns([1, 8, 1], vertical_alignment="center")
                l.markdown("⚠️")
                m.markdown(f"**{s.title}**  \n{s.project} · {s.theme} · {s.owner}{(' · META' if s.is_meta else '')}")
                if r.button("›", key=f"over_open_{s.series_id}"):
                    open_detail(s.series_id)
