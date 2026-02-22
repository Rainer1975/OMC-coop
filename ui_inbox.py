from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st


def render(ctx: dict) -> None:
    today: date = ctx["today"]
    safe_container = ctx["safe_container"]
    persist = ctx["persist"]
    sync_lists_from_data = ctx["sync_lists_from_data"]

    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]
    is_completed = ctx["is_completed"]

    new_series = ctx["new_series"]
    open_detail = ctx["open_detail"]

    employees = ctx["employees"]
    pick_owner = ctx["pick_owner"]
    pick_from_list = ctx["pick_from_list"]

    TASK_STATES = ctx.get("TASK_STATES", ["PLANNED", "ACTIVE", "BLOCKED", "DONE", "CANCELLED"])

    st.title("Inbox")
    st.caption("Un-triaged tasks. Ziel: schnell einsammeln → später sauber triagieren (Portfolio/Projekt/Owner/Datum/State).")

    st.subheader("Capture")
    with st.form("inbox_capture"):
        title = st.text_input("Title", placeholder="e.g., Follow up with Roman re: PowerBI lineage")
        notes = st.text_area("Notes (optional)", height=90)
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", today)
        end = c2.date_input("End", today)
        ok = st.form_submit_button("Add to Inbox")

    if ok:
        if not (title or "").strip():
            st.warning("Title required.")
        elif start > end:
            st.warning("Start must be <= End.")
        else:
            s = new_series(
                title=title,
                project="",  # triage later
                theme="",
                owner="",
                start=start,
                end=end,
                is_meta=False,
                kind="task",
                notes=notes or "",
                portfolio="Default",
                state="PLANNED",
                inbox=True,
            )
            st.session_state.series.append(s)
            persist()
            sync_lists_from_data()
            st.success("Captured.")
            st.rerun()

    st.divider()
    st.subheader("Triage")

    items = [s for s in visible_series() if is_task(s) and bool(getattr(s, "inbox", False))]
    if not items:
        st.info("Inbox empty.")
        return

    # Quick filter
    q = st.text_input("Search", placeholder="search title / notes…", key="inbox_search")
    if q:
        items = [s for s in items if q.lower() in ((s.title or "") + " " + (getattr(s, "notes", "") or "")).lower()]

    # Table overview
    rows = []
    for s in items:
        rows.append(
            {
                "ID": s.series_id,
                "Title": s.title,
                "Start": s.start,
                "End": s.end,
                "State": getattr(s, "state", "PLANNED"),
                "Portfolio": getattr(s, "portfolio", "Default"),
                "Project": s.project,
                "Theme": s.theme,
                "Owner": s.owner,
                "Done": "YES" if is_completed(s, today) else "",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Edit selected")

    pick = st.selectbox(
        "Inbox item",
        options=[f"{r['Title']} ({r['Start']}→{r['End']})" for r in rows],
        index=0,
        key="inbox_pick",
    )
    sid = rows[[f"{r['Title']} ({r['Start']}→{r['End']})" for r in rows].index(pick)]["ID"]
    s = next(x for x in st.session_state.series if x.series_id == sid)

    with safe_container(border=True):
        st.markdown(f"**{s.title}**")
        st.caption(s.series_id)

        t_title = st.text_input("Title", value=s.title, key="inbox_edit_title")
        t_notes = st.text_area("Notes", value=getattr(s, "notes", "") or "", height=120, key="inbox_edit_notes")

        c1, c2 = st.columns(2)
        t_start = c1.date_input("Start", value=s.start, key="inbox_edit_start")
        t_end = c2.date_input("End", value=s.end, key="inbox_edit_end")

        c3, c4 = st.columns(2)
        t_state = c3.selectbox(
            "State",
            options=TASK_STATES,
            index=TASK_STATES.index(getattr(s, "state", "PLANNED")) if getattr(s, "state", "PLANNED") in TASK_STATES else 0,
            key="inbox_edit_state",
        )
        t_portfolio = pick_from_list(
            "Portfolio",
            key="inbox_edit_portfolio",
            values=st.session_state.lists.get("portfolios", ["Default"]),
            current=getattr(s, "portfolio", "Default") or "Default",
            list_key="portfolios",
        )

        t_project = pick_from_list(
            "Project",
            key="inbox_edit_project",
            values=st.session_state.lists.get("projects", []),
            current=s.project or "",
            require=False,
            list_key="projects",
        )
        t_theme = pick_from_list(
            "Theme",
            key="inbox_edit_theme",
            values=st.session_state.lists.get("themes", []),
            current=s.theme or "",
            require=False,
            list_key="themes",
        )

        oid, od = pick_owner(
            "Owner",
            key="inbox_edit_owner",
            current_owner_id=getattr(s, "owner_id", ""),
            current_owner_display=s.owner or "",
            employees=employees,
        )

        c5, c6, c7 = st.columns([2, 2, 6])
        if c5.button("Save", key="inbox_save"):
            if (t_start > t_end):
                st.warning("Start must be <= End.")
            else:
                s.title = (t_title or "").strip()
                s.notes = t_notes or ""
                s.start = t_start
                s.end = t_end
                s.state = (t_state or "PLANNED").strip().upper()
                s.portfolio = (t_portfolio or "Default").strip() or "Default"
                s.project = (t_project or "").strip()
                s.theme = (t_theme or "").strip()
                s.owner_id = oid or ""
                s.owner = od or ""
                persist()
                sync_lists_from_data()
                st.success("Saved.")
                st.rerun()

        if c6.button("Convert (remove from Inbox)", key="inbox_convert"):
            # minimal validation on convert
            if not (s.title or "").strip():
                st.warning("Title required.")
            else:
                s.inbox = False
                persist()
                sync_lists_from_data()
                st.success("Converted.")
                st.rerun()

        if c7.button("Open detail ›", key="inbox_open"):
            open_detail(s.series_id)

    st.caption("Hinweis: Inbox erlaubt absichtlich unvollständige Daten. 'Convert' macht daraus einen normalen Task.")
