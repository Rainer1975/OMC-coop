# ui_employees.py
from __future__ import annotations

import re
from datetime import date

import streamlit as st


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _get_notes(s) -> str:
    # TaskSeries has no .notes field; we store notes in meta["notes"]
    try:
        meta = getattr(s, "meta", {}) or {}
        if isinstance(meta, dict):
            return str(meta.get("notes") or "")
    except Exception:
        pass
    return ""


def _series_exists(
    ctx: dict,
    *,
    kind: str,
    owner_id: str,
    owner: str,
    title: str,
    portfolio: str,
    project: str,
    theme: str,
    start: date,
    end: date,
    time_start: str = "",
    time_end: str = "",
    location: str = "",
    notes: str = "",
) -> bool:
    """Idempotency guard: prevent duplicates on repeated submit."""
    is_task = ctx["is_task"]
    is_appointment = ctx["is_appointment"]

    k = (kind or "").strip().lower()

    for s in st.session_state.series:
        try:
            # owner match
            if _norm(getattr(s, "owner_id", "")) != _norm(owner_id):
                continue

            # kind match
            if k == "task":
                if not is_task(s):
                    continue
            elif k == "appointment":
                if not is_appointment(s):
                    continue
            else:
                continue

            # core identity
            if _norm(getattr(s, "title", "")) != _norm(title):
                continue
            if _norm(getattr(s, "project", "")) != _norm(project):
                continue
            if _norm(getattr(s, "theme", "")) != _norm(theme):
                continue
            if _norm(getattr(s, "portfolio", "")) != _norm(portfolio):
                continue

            if getattr(s, "start", None) != start:
                continue
            if getattr(s, "end", None) != end:
                continue

            # appointments: include time/location/notes in identity to avoid false positives
            if k == "appointment":
                if _norm(getattr(s, "time_start", "")) != _norm(time_start):
                    continue
                if _norm(getattr(s, "time_end", "")) != _norm(time_end):
                    continue
                if _norm(getattr(s, "location", "")) != _norm(location):
                    continue
                if _norm(_get_notes(s)) != _norm(notes):
                    continue

            return True
        except Exception:
            # if anything is weird, do not block creation
            continue

    return False


def render(ctx: dict) -> None:
    today = ctx["today"]
    pick_from_list = ctx["pick_from_list"]
    persist = ctx["persist"]
    sync_lists_from_data = ctx["sync_lists_from_data"]
    request_done_grid = ctx["request_done_grid"]

    new_series = ctx["new_series"]
    is_task = ctx["is_task"]
    is_appointment = ctx["is_appointment"]
    is_completed = ctx["is_completed"]
    bulk_set_done = ctx["bulk_set_done"]
    find_series = ctx["find_series"]
    open_detail = ctx["open_detail"]

    employees = ctx["employees"]

    st.title("Employees")
    st.caption("Self-service: Tasks/Termine anlegen + Grid quittieren (Future disabled).")

    if not employees:
        st.info("Keine Employees angelegt. Bitte im Admin/Data Bereich importieren oder Employees anlegen.")
        return

    # -------- Pick employee
    emp_labels = [e.get("display_name", "") for e in employees if e.get("id") and e.get("display_name")]
    emp_ids = [e.get("id", "") for e in employees if e.get("id") and e.get("display_name")]

    if not emp_labels:
        st.warning("Employees vorhanden, aber ohne gültige IDs/Display Names.")
        return

    pick_label = st.selectbox("Employee", options=emp_labels, index=0, key="emp_pick")
    pick_idx = emp_labels.index(pick_label)
    pick_id = emp_ids[pick_idx]
    pick_name = pick_label

    # -------- Tabs
    tab_task, tab_appt = st.tabs(["➕ Create task", "📅 Create appointment"])

    # -------------------------
    # TASK CREATE (idempotent)
    # -------------------------
    with tab_task:
        form_key = f"emp_create_task_{pick_id}"
        with st.form(form_key):
            t_title = st.text_input("Title", key=f"emp_task_title_{pick_id}")

            t_portfolio = pick_from_list(
                "Portfolio",
                key=f"emp_task_portfolio_{pick_id}",
                values=st.session_state.lists.get("portfolios", []),
                current="Default",
                require=True,
                list_key="portfolios",
            )
            t_project = pick_from_list(
                "Project",
                key=f"emp_task_project_{pick_id}",
                values=st.session_state.lists.get("projects", []),
                current="",
                require=True,
                list_key="projects",
            )
            t_theme = pick_from_list(
                "Theme",
                key=f"emp_task_theme_{pick_id}",
                values=st.session_state.lists.get("themes", []),
                current="General",
                require=True,
                list_key="themes",
            )

            c1, c2 = st.columns(2)
            t_start = c1.date_input("Start", today, key=f"emp_task_start_{pick_id}")
            t_end = c2.date_input("End", today, key=f"emp_task_end_{pick_id}")

            t_meta = st.checkbox("META", value=False, key=f"emp_task_meta_{pick_id}", disabled=st.session_state.get("focus_mode", False))

            ok = st.form_submit_button("Create task")

        if ok:
            if not (t_title or "").strip():
                st.warning("Title required.")
            elif not (t_project or "").strip():
                st.warning("Project required.")
            elif not (t_theme or "").strip():
                st.warning("Theme required.")
            elif t_start > t_end:
                st.warning("Start must be <= End.")
            else:
                is_meta_val = False if st.session_state.get("focus_mode") else bool(t_meta)

                if _series_exists(
                    ctx,
                    kind="task",
                    owner_id=pick_id,
                    owner=pick_name,
                    title=t_title,
                    portfolio=t_portfolio,
                    project=t_project,
                    theme=t_theme,
                    start=t_start,
                    end=t_end,
                ):
                    st.warning("Nicht angelegt: identischer Task existiert bereits (Doppelklick/Mehrfach-Submit abgefangen).")
                else:
                    s = new_series(
                        title=t_title,
                        portfolio=t_portfolio,
                        project=t_project,
                        theme=t_theme,
                        owner=pick_name,
                        owner_id=pick_id,
                        start=t_start,
                        end=t_end,
                        is_meta=is_meta_val,
                        kind="task",
                        state="PLANNED",
                    )
                    st.session_state.series.append(s)
                    persist()
                    sync_lists_from_data()
                    st.success("Task created.")
                st.rerun()

    # -------------------------
    # APPOINTMENT CREATE (idempotent)
    # -------------------------
    with tab_appt:
        if st.session_state.get("focus_mode"):
            st.caption("Disabled in Focus mode.")
        else:
            form_key = f"emp_create_appt_{pick_id}"
            with st.form(form_key):
                a_title = st.text_input("Appointment title", key=f"emp_appt_title_{pick_id}")

                a_portfolio = pick_from_list(
                    "Portfolio",
                    key=f"emp_appt_portfolio_{pick_id}",
                    values=st.session_state.lists.get("portfolios", []),
                    current="Default",
                    require=True,
                    list_key="portfolios",
                )
                a_project = pick_from_list(
                    "Project",
                    key=f"emp_appt_project_{pick_id}",
                    values=st.session_state.lists.get("projects", []),
                    current="",
                    require=True,
                    list_key="projects",
                )
                a_theme = pick_from_list(
                    "Theme",
                    key=f"emp_appt_theme_{pick_id}",
                    values=st.session_state.lists.get("themes", []),
                    current="Termin",
                    require=False,
                    list_key="themes",
                )

                c1, c2 = st.columns(2)
                a_start = c1.date_input("Date (start)", today, key=f"emp_appt_start_{pick_id}")
                a_end = c2.date_input("Date (end)", today, key=f"emp_appt_end_{pick_id}")

                a_meta = st.checkbox("META", value=False, key=f"emp_appt_meta_{pick_id}")

                c3, c4 = st.columns(2)
                a_ts = c3.text_input("Time start (HH:MM)", key=f"emp_appt_ts_{pick_id}")
                a_te = c4.text_input("Time end (HH:MM)", key=f"emp_appt_te_{pick_id}")

                a_loc = st.text_input("Location", key=f"emp_appt_loc_{pick_id}")
                a_notes = st.text_area("Notes", height=80, key=f"emp_appt_notes_{pick_id}")

                ok2 = st.form_submit_button("Create appointment")

            if ok2:
                if not (a_title or "").strip():
                    st.warning("Appointment title required.")
                elif not (a_project or "").strip():
                    st.warning("Project required.")
                elif a_start > a_end:
                    st.warning("Start must be <= End.")
                else:
                    theme_val = (a_theme or "Termin")

                    if _series_exists(
                        ctx,
                        kind="appointment",
                        owner_id=pick_id,
                        owner=pick_name,
                        title=a_title,
                        portfolio=a_portfolio,
                        project=a_project,
                        theme=theme_val,
                        start=a_start,
                        end=a_end,
                        time_start=a_ts,
                        time_end=a_te,
                        location=a_loc,
                        notes=a_notes,
                    ):
                        st.warning("Nicht angelegt: identischer Termin existiert bereits (Doppelklick/Mehrfach-Submit abgefangen).")
                    else:
                        # IMPORTANT: core.new_series does NOT accept time_start/time_end/location/notes.
                        # Create first, then set appointment fields and store notes in meta.
                        s = new_series(
                            title=a_title,
                            portfolio=a_portfolio,
                            project=a_project,
                            theme=theme_val,
                            owner=pick_name,
                            owner_id=pick_id,
                            start=a_start,
                            end=a_end,
                            is_meta=bool(a_meta),
                            kind="appointment",
                            state="PLANNED",
                        )
                        # set appointment fields
                        try:
                            s.time_start = str(a_ts or "")
                            s.time_end = str(a_te or "")
                            s.location = str(a_loc or "")
                        except Exception:
                            pass
                        # store notes in meta
                        try:
                            if not isinstance(getattr(s, "meta", None), dict):
                                s.meta = {}
                            s.meta["notes"] = str(a_notes or "")
                        except Exception:
                            pass

                        st.session_state.series.append(s)
                        persist()
                        sync_lists_from_data()
                        st.success("Appointment created.")
                    st.rerun()

    # -------------------------
    # GRID / DONE
    # -------------------------
    st.divider()
    st.subheader("Quittierung (Grid)")

    tasks = [s for s in st.session_state.series if is_task(s) and getattr(s, "owner_id", "") == pick_id]
    if st.session_state.get("focus_mode"):
        tasks = [s for s in tasks if (not getattr(s, "is_meta", False)) and (not is_completed(s, today))]

    if not tasks:
        st.info("Keine Tasks für diesen Employee.")
        return

    # show tasks list + open button + grid done selection
    st.caption("Markiere DONE pro Tag (Future disabled).")

    # quick list with open
    with st.expander("📋 Tasks (open detail)", expanded=True):
        for s in sorted(tasks, key=lambda x: (getattr(x, "start", today), getattr(x, "end", today), _norm(getattr(x, "title", "")))):
            c1, c2, c3, c4 = st.columns([5, 2, 2, 3])
            c1.write(f"**{s.title}**")
            c2.write(f"{s.start.isoformat()} → {s.end.isoformat()}")
            c3.write(getattr(s, "state", ""))
            if c4.button("Open", key=f"emp_open_{s.series_id}"):
                open_detail(s.series_id)

    # Grid editor: choose dates to set done
    st.markdown("**DONE setzen (mehrere Einträge, quittierungspflichtig):**")

    # build selectable day range for each task
    items = []
    for s in tasks:
        d = s.start
        while d <= s.end:
            if d <= today:
                items.append((s.series_id, d))
            d = d.fromordinal(d.toordinal() + 1)

    if not items:
        st.info("Keine quittierbaren Tage (alles liegt in der Zukunft).")
        return

    # compact selection: multi-select by human label
    labels = []
    lookup = {}
    for sid, d in items:
        s = find_series(sid)
        lab = f"{d.isoformat()} · {s.title}"
        labels.append(lab)
        lookup[lab] = {"series_id": sid, "day_iso": d.isoformat()}

    picked = st.multiselect("Auswahl", options=labels, default=[], key=f"emp_done_pick_{pick_id}")

    if st.button("DONE setzen (Quittierung anfordern)", key=f"emp_done_request_{pick_id}"):
        req_items = [lookup[x] for x in picked if x in lookup]
        if not req_items:
            st.warning("Nichts ausgewählt.")
        else:
            request_done_grid(req_items, reason=f"Employees/Grid: {pick_name}")
            st.rerun()

    st.caption("Hinweis: DONE entfernen erfolgt in der Detailansicht bzw. über die jeweiligen Seitenfunktionen.")
