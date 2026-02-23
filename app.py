# app.py
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (
    TaskPart,
    TaskSeries,
    bulk_set_done,
    can_depend_series,
    capacity_summary,
    compute_units_composition,
    build_burndown_series,
    calc_velocity,
    forecast_finish_date,
    gantt_items,
    is_active,
    is_appointment,
    is_completed,
    is_overdue,
    is_task,
    load_state,
    mark_done,
    new_part,
    new_series,
    progress_percent,
    save_state,
    total_days,
    unmark_done,
    would_create_cycle,
)

from ui_admin import render as render_admin
from ui_burndown import render as render_burndown
from ui_dashboard import render as render_dashboard
from ui_data import render as render_data
from ui_detail import render as render_detail
from ui_employees import render as render_employees
from ui_gantt import render as render_gantt
from ui_home import render as render_home
from ui_kanban import render as render_kanban

APP_TITLE = "OMG Coop"

STATE_PATH = Path("data.json")
EMP_PATH = Path("employees.json")
LISTS_PATH = Path("lists.json")

ADD_NEW = "➕ Add new..."

DEFAULT_LISTS = {
    "portfolios": ["Default"],
    "projects": [],
    "themes": ["General", "Termin"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _list_union(a: List[str], b: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in (a or []) + (b or []):
        x = _norm(x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def load_employees() -> List[Dict[str, Any]]:
    if not EMP_PATH.exists():
        return []
    try:
        data = json.loads(EMP_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_employees(employees: List[Dict[str, Any]]) -> None:
    EMP_PATH.write_text(json.dumps(employees, ensure_ascii=False, indent=2), encoding="utf-8")


def load_lists() -> Dict[str, List[str]]:
    if not LISTS_PATH.exists():
        return dict(DEFAULT_LISTS)
    try:
        data = json.loads(LISTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out = dict(DEFAULT_LISTS)
            for k in DEFAULT_LISTS:
                if k in data and isinstance(data[k], list):
                    out[k] = [_norm(x) for x in data[k] if _norm(x)]
            return out
    except Exception:
        pass
    return dict(DEFAULT_LISTS)


def save_lists(portfolios: List[str], projects: List[str], themes: List[str]) -> None:
    data = {
        "portfolios": _list_union(["Default"], portfolios),
        "projects": _list_union([], projects),
        "themes": _list_union(["General", "Termin"], themes),
    }
    LISTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def persist() -> None:
    save_state(STATE_PATH, st.session_state.series)


def sync_lists_from_data() -> None:
    portfolios = set()
    projects = set()
    themes = set()
    for s in st.session_state.series:
        portfolios.add(_norm(getattr(s, "portfolio", "") or "Default") or "Default")
        if _norm(getattr(s, "project", "")):
            projects.add(_norm(getattr(s, "project", "")))
        if _norm(getattr(s, "theme", "")):
            themes.add(_norm(getattr(s, "theme", "")))

    st.session_state.lists["portfolios"] = _list_union(["Default"], sorted(portfolios))
    st.session_state.lists["projects"] = _list_union([], sorted(projects))
    st.session_state.lists["themes"] = _list_union(["General", "Termin"], sorted(themes))
    save_lists(
        st.session_state.lists.get("portfolios", []),
        st.session_state.lists.get("projects", []),
        st.session_state.lists.get("themes", []),
    )


def open_detail(series_id: str) -> None:
    st.session_state.open_series = series_id
    st.session_state.page_prev = st.session_state.page
    st.session_state.nav_select_before_detail = st.session_state.get("nav_select")
    st.session_state.page = "DETAIL"
    st.session_state.nav_force_sync = True
    st.rerun()


def close_detail() -> None:
    st.session_state.open_series = None
    back = st.session_state.page_prev if st.session_state.page_prev != "DETAIL" else "HOME"
    st.session_state.page = back
    st.session_state.nav_force_sync = True
    st.rerun()


def find_series(series_id: str) -> TaskSeries:
    return next(x for x in st.session_state.series if x.series_id == series_id)


def delete_series(series_id: str) -> None:
    st.session_state.series = [x for x in st.session_state.series if x.series_id != series_id]
    persist()


def appointment_label(s: TaskSeries) -> str:
    ts = (getattr(s, "time_start", "") or "").strip()
    te = (getattr(s, "time_end", "") or "").strip()
    loc = (getattr(s, "location", "") or "").strip()
    parts = []
    if ts and te:
        parts.append(f"{ts}–{te}")
    elif ts:
        parts.append(ts)
    if loc:
        parts.append(loc)
    suffix = (" · " + " · ".join(parts)) if parts else ""
    return f"{s.title}{suffix}"


def visible_series() -> List[TaskSeries]:
    out: List[TaskSeries] = []
    for s in st.session_state.series:
        if st.session_state.focus_mode:
            if getattr(s, "is_meta", False):
                continue
            if is_appointment(s):
                continue
        out.append(s)
    return out


def pick_from_list(
    label: str,
    key: str,
    values: List[str],
    current: str = "",
    require: bool = True,
    list_key: str = "",
) -> str:
    values = [_norm(v) for v in (values or []) if _norm(v)]
    cur = _norm(current)

    if cur and cur not in values:
        values = [cur] + values

    options = values + [ADD_NEW] if values else [ADD_NEW]
    idx = options.index(cur) if cur in options else 0

    pick = st.selectbox(label, options=options, index=idx, key=key)

    # FORM-SAFE: input always visible
    new_key = f"{key}__new"
    new_val = st.text_input(
        f"{label} (new – nur wenn 'Add new...' gewählt)",
        value=st.session_state.get(new_key, ""),
        key=new_key,
        placeholder=f"Neuen Wert für {label} eingeben …",
    ).strip()

    if pick == ADD_NEW:
        if not new_val:
            return "" if require else ""
        new_val = _norm(new_val)
        if list_key:
            if new_val not in st.session_state.lists.get(list_key, []):
                st.session_state.lists[list_key] = _list_union(st.session_state.lists.get(list_key, []), [new_val])
                save_lists(
                    st.session_state.lists.get("portfolios", []),
                    st.session_state.lists.get("projects", []),
                    st.session_state.lists.get("themes", []),
                )
        return new_val

    return pick


def request_done_single(series_id: str, day: date, reason: str = "") -> None:
    st.session_state.done_requests = st.session_state.get("done_requests", [])
    st.session_state.done_requests.append({"series_id": series_id, "day_iso": day.isoformat(), "reason": reason or ""})


def request_done_grid(items: List[Dict[str, str]], reason: str = "") -> None:
    st.session_state.done_requests = st.session_state.get("done_requests", [])
    for it in items:
        st.session_state.done_requests.append({"series_id": it["series_id"], "day_iso": it["day_iso"], "reason": reason})


def handle_done_requests() -> None:
    reqs = st.session_state.get("done_requests", [])
    if not reqs:
        return
    with st.sidebar.expander("✅ Quittierung nötig", expanded=True):
        st.caption("Es gibt DONE-Anfragen, die bestätigt werden müssen.")
        keep = []
        for i, r in enumerate(reqs):
            sid = r.get("series_id", "")
            day_iso = r.get("day_iso", "")
            reason = r.get("reason", "")
            try:
                s = find_series(sid)
            except Exception:
                continue
            row = st.columns([5, 2, 2])
            row[0].write(f"**{s.title}** · {day_iso}")
            if reason:
                row[0].caption(reason)
            if row[1].button("Quittieren", key=f"done_ok_{i}_{sid}_{day_iso}"):
                mark_done(s, date.fromisoformat(day_iso))
                persist()
            elif row[2].button("Ablehnen", key=f"done_no_{i}_{sid}_{day_iso}"):
                pass
            else:
                keep.append(r)
        st.session_state.done_requests = keep


# ------------------ Streamlit setup ------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.page = "HOME"
    st.session_state.page_prev = "HOME"
    st.session_state.open_series = None
    st.session_state.focus_mode = False
    st.session_state.nav_force_sync = True
    st.session_state.nav_select_before_detail = None

    st.session_state.series = load_state(STATE_PATH)
    st.session_state.employees = load_employees()
    st.session_state.lists = load_lists()

    sync_lists_from_data()

today = date.today()

# ------------------ Sidebar (header + quick stats) ------------------

st.sidebar.title(APP_TITLE)

st.session_state.focus_mode = st.sidebar.toggle(
    "Focus mode (only active tasks,\nno META, no appointments)",
    value=bool(st.session_state.get("focus_mode", False)),
    key="focus_mode_toggle",
)

v = visible_series()
k_today = sum(1 for s in v if is_task(s) and s.start <= today <= s.end and not is_completed(s, today))
k_overdue = sum(1 for s in v if is_task(s) and is_overdue(s, today) and not is_completed(s, today))
k_active = sum(1 for s in v if is_task(s) and is_active(s) and not is_completed(s, today))
k_meta = sum(1 for s in v if getattr(s, "is_meta", False))

c1, c2, c3, c4 = st.sidebar.columns(4)
c1.metric("Today", k_today)
c2.metric("Overdue", k_overdue)
c3.metric("Active", k_active)
c4.metric("Meta", k_meta)

handle_done_requests()

with st.sidebar.expander("Hilfe", expanded=False):
    st.button("❓ Hilfe", key="help_btn")
    st.button("🧙 Anfänger", key="beginner_btn")
    st.caption("Hilfe ist kontextsensitiv: öffnet automatisch\nden passenden Abschnitt zur aktuellen\nSeite.")

# ------------------ Navigation ------------------

NAV_SECTIONS = [
    ("Start", [("Home", "HOME")]),
    ("Work", [("Kanban", "KANBAN"), ("Gantt", "GANTT"), ("Burndown", "BURNDOWN"), ("Dashboard", "DASHBOARD")]),
    ("People", [("Employees", "EMPLOYEES")]),
    ("Admin", [("Data", "DATA"), ("Admin", "ADMIN")]),
]

display_options: List[str] = []
display_to_code: Dict[str, str] = {}
code_to_display: Dict[str, str] = {}

for sec, items in NAV_SECTIONS:
    for label, code in items:
        disp = f"{sec} › {label}"
        display_to_code[disp] = code
        code_to_display[code] = disp
        display_options.append(disp)

if st.session_state.page == "DETAIL":
    cur_disp = st.session_state.get("nav_select_before_detail") or code_to_display.get(
        st.session_state.get("page_prev") or "HOME",
        display_options[0],
    )
else:
    cur_code = st.session_state.page if st.session_state.page in code_to_display else "HOME"
    cur_disp = code_to_display.get(cur_code, display_options[0])

# IMPORTANT: session_state drives the selectbox default (no index/value)
if "nav_select" not in st.session_state:
    st.session_state["nav_select"] = cur_disp

if st.session_state.nav_force_sync:
    st.session_state["nav_select"] = cur_disp
    st.session_state.nav_force_sync = False

picked_disp = st.sidebar.selectbox(
    label="Navigation",
    options=display_options,
    label_visibility="collapsed",
    key="nav_select",
)

picked_page = display_to_code.get(picked_disp, "HOME")

if st.session_state.page == "DETAIL":
    before = st.session_state.get("nav_select_before_detail")
    if before is not None and picked_disp == before:
        picked_page = "DETAIL"

if picked_page != st.session_state.page:
    st.session_state.page_prev = st.session_state.page
    st.session_state.page = picked_page
    st.session_state.nav_force_sync = False
    if picked_page != "DETAIL":
        st.session_state.open_series = None
    st.rerun()

with st.sidebar.expander("Quick add (task)", expanded=False):
    qa_title = st.text_input("Title", key="qa_title")

    qa_owner = st.selectbox(
        "Owner",
        options=[e.get("display_name", "") for e in st.session_state.employees if e.get("display_name")],
        index=0 if st.session_state.employees else 0,
        key="qa_owner",
    )
    qa_owner_id = ""
    for e in st.session_state.employees:
        if e.get("display_name") == qa_owner:
            qa_owner_id = e.get("id", "")
            break

    qa_project = pick_from_list("Project", key="qa_project", values=st.session_state.lists.get("projects", []), current="", require=True, list_key="projects")
    qa_theme = pick_from_list("Theme", key="qa_theme", values=st.session_state.lists.get("themes", []), current="General", require=True, list_key="themes")
    qa_portfolio = pick_from_list("Portfolio", key="qa_portfolio", values=st.session_state.lists.get("portfolios", []), current="Default", require=True, list_key="portfolios")

    qa_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=5, step=1, key="qa_days")
    qa_weight = st.number_input("Weight/Units", min_value=0.0, max_value=1000.0, value=1.0, step=1.0, key="qa_weight")

    if st.button("Add", key="qa_add"):
        if qa_title.strip():
            s = new_series(
                title=qa_title.strip(),
                portfolio=qa_portfolio,
                project=qa_project,
                theme=qa_theme,
                owner=qa_owner,
                owner_id=qa_owner_id,
                start=today,
                end=today + timedelta(days=int(qa_days) - 1),
                is_meta=False,
                kind="task",
                state="ACTIVE",
            )
            try:
                s.weight = float(qa_weight)
            except Exception:
                pass
            st.session_state.series.append(s)
            persist()
            sync_lists_from_data()
            st.success("Added.")
            st.session_state.qa_title = ""
            st.session_state.nav_force_sync = True
            st.rerun()
        else:
            st.warning("Title required.")

# ------------------ Page render context ------------------

ctx = {
    "today": today,
    "persist": persist,
    "sync_lists_from_data": sync_lists_from_data,
    "visible_series": visible_series,
    "find_series": find_series,
    "open_detail": open_detail,
    "close_detail": close_detail,
    "delete_series": delete_series,
    "new_series": new_series,
    "new_part": new_part,
    "is_task": is_task,
    "is_appointment": is_appointment,
    "is_completed": is_completed,
    "is_overdue": is_overdue,
    "is_active": is_active,
    "progress_percent": progress_percent,
    "gantt_items": gantt_items,
    "would_create_cycle": would_create_cycle,
    "can_depend_series": can_depend_series,
    "total_days": total_days,
    "bulk_set_done": bulk_set_done,
    "mark_done": mark_done,
    "unmark_done": unmark_done,
    "appointment_label": appointment_label,
    "pick_from_list": pick_from_list,
    "request_done_single": request_done_single,
    "request_done_grid": request_done_grid,
    "employees": st.session_state.employees,
    "lists": st.session_state.lists,
    "capacity_summary": capacity_summary,
    "compute_units_composition": compute_units_composition,
    "build_burndown_series": build_burndown_series,
    "calc_velocity": calc_velocity,
    "forecast_finish_date": forecast_finish_date,

    # ✅ REQUIRED by ui_data.py (fix for KeyError)
    "DATA_FILE": str(STATE_PATH),
    "EMP_FILE": str(EMP_PATH),
    "LISTS_FILE": str(LISTS_PATH),
}

# ------------------ Main render ------------------

if st.session_state.page == "DETAIL":
    if not st.session_state.open_series:
        st.warning("No item selected.")
        st.session_state.page = "HOME"
        st.session_state.nav_force_sync = True
        st.rerun()
    else:
        render_detail(ctx)
elif st.session_state.page == "HOME":
    render_home(ctx)
elif st.session_state.page == "KANBAN":
    render_kanban(ctx)
elif st.session_state.page == "GANTT":
    render_gantt(ctx)
elif st.session_state.page == "BURNDOWN":
    render_burndown(ctx)
elif st.session_state.page == "DASHBOARD":
    render_dashboard(ctx)
elif st.session_state.page == "EMPLOYEES":
    render_employees(ctx)
elif st.session_state.page == "DATA":
    render_data(ctx)
elif st.session_state.page == "ADMIN":
    render_admin(ctx)
else:
    render_home(ctx)
