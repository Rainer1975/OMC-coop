# ui_meta.py
from __future__ import annotations

import pandas as pd
import streamlit as st

def render(ctx: dict) -> None:
    today = ctx["today"]
    visible_series = ctx["visible_series"]
    is_task = ctx["is_task"]
    is_completed = ctx["is_completed"]
    progress_percent = ctx["progress_percent"]
    open_detail = ctx["open_detail"]

    st.title("Meta")
    st.caption("META-Tasks: sichtbar nur wenn Focus mode aus.")

    if st.session_state.get("focus_mode"):
        st.info("Focus mode aktiv: META ist ausgeblendet.")
        return

    metas = [s for s in visible_series() if is_task(s) and s.is_meta]
    if not metas:
        st.info("No META tasks.")
        return

    rows = []
    for s in metas:
        rows.append(
            {
                "Project": s.project,
                "Theme": s.theme,
                "Owner": s.owner,
                "Task": s.title,
                "Start": s.start.isoformat(),
                "End": s.end.isoformat(),
                "State": getattr(s,"state","PLANNED"), "Progress %": progress_percent(s),
                "Done": "YES" if is_completed(s, today) else "",
                "ID": s.series_id,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

    pick = st.selectbox("Open META task", options=[f"{r['Project']} · {r['Task']}" for r in rows], index=0)
    if st.button("Open selected", key="meta_open"):
        sid = rows[[f"{r['Project']} · {r['Task']}" for r in rows].index(pick)]["ID"]
        open_detail(sid)
