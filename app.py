# app.py
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    capacity_summary as core_capacity_summary,
    compute_units_composition,
    build_burndown_series,
    calc_velocity,
    forecast_finish_date,
    forecast_eta_units,
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
    """Load employees from employees.json.

    Compatibility:
    - canonical: list[dict]
    - container: {"schema_version": 1, "employees": [...]}
    - legacy/dirty: list[str] (names) or list[str(dict-repr)]

    Output: list[{id, display_name, aliases}]
    """

    def _slugify(s: str) -> str:
        s = (s or "").strip().lower()
        out = []
        for ch in s:
            if ch.isalnum():
                out.append(ch)
            else:
                out.append("_")
        v = "".join(out).strip("_")
        while "__" in v:
            v = v.replace("__", "_")
        return v or "unknown"

    def _maybe_parse_dict_string(s: str) -> Dict[str, Any] | None:
        s = (s or "").strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        try:
            import ast

            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        return None

    if not EMP_PATH.exists():
        return []

    try:
        raw = json.loads(EMP_PATH.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    if isinstance(raw, dict) and isinstance(raw.get("employees"), list):
        raw_list = raw.get("employees", [])
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for item in raw_list:
        try:
            if isinstance(item, str):
                d = _maybe_parse_dict_string(item)
                if isinstance(d, dict):
                    item = d
                else:
                    name = _norm(item)
                    if not name:
                        continue
                    eid = _slugify(name)
                    if eid in seen:
                        continue
                    seen.add(eid)
                    cleaned.append({"id": eid, "display_name": name, "aliases": [name]})
                    continue

            if not isinstance(item, dict):
                continue

            dn_raw = str(item.get("display_name", "") or "").strip()
            if dn_raw.startswith("{") and "display_name" in dn_raw:
                d2 = _maybe_parse_dict_string(dn_raw)
                if isinstance(d2, dict):
                    item = d2
                    dn_raw = str(item.get("display_name", "") or "").strip()

            dn = _norm(dn_raw)
            if not dn:
                continue

            eid = str(item.get("id", "") or "").strip() or _slugify(dn)
            if eid.lower() == "none":
                eid = _slugify(dn)
            if eid in seen:
                continue
            seen.add(eid)

            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            aliases = [_norm(str(a)) for a in aliases if _norm(str(a))]
            if dn and dn not in aliases:
                aliases = [dn] + aliases

            cleaned.append({"id": eid, "display_name": dn, "aliases": aliases})
        except Exception:
            continue

    cleaned.sort(key=lambda x: (str(x.get("display_name") or "").lower()))
    return cleaned


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


def _extract_list_values(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            if isinstance(item, dict):
                v = item.get("value")
                if v is None:
                    continue
                v = _norm(v)
                if v:
                    out.append(v)
            else:
                v = _norm(item)
                if v:
                    out.append(v)
        return out
    v = _norm(x)
    return [v] if v else []


def save_lists(*args, **kwargs) -> None:
    """Persist portfolios/projects/themes.

    Supports both:
      1) save_lists(portfolios, projects, themes)
      2) save_lists({"portfolios": ..., "projects": ..., "themes": ...})
    """
    portfolios: List[str] = []
    projects: List[str] = []
    themes: List[str] = []

    if len(args) == 1 and isinstance(args[0], dict):
        d = args[0]
        portfolios = _extract_list_values(d.get("portfolios"))
        projects = _extract_list_values(d.get("projects"))
        themes = _extract_list_values(d.get("themes"))
    elif len(args) >= 3:
        portfolios = _extract_list_values(args[0])
        projects = _extract_list_values(args[1])
        themes = _extract_list_values(args[2])
    else:
        portfolios = _extract_list_values(kwargs.get("portfolios"))
        projects = _extract_list_values(kwargs.get("projects"))
        themes = _extract_list_values(kwargs.get("themes"))

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
    for s in st.session_state.series:
        if getattr(s, "id", "") == series_id:
            return s
    raise KeyError(series_id)


def visible_series() -> List[TaskSeries]:
    out: List[TaskSeries] = []
    for s in st.session_state.series:
        if st.session_state.get("focus_mode", False):
            if getattr(s, "is_meta", False):
                continue
            if is_appointment(s):
                continue
            if is_task(s) and not is_active(s):
                continue
        out.append(s)
    return out


def appointment_label(s: TaskSeries) -> str:
    owner = getattr(s, "owner", "") or "—"
    proj = getattr(s, "project", "") or "—"
    return f"{owner} · {proj}"


def pick_owner(label: str, key: str, current_owner: str = "") -> Tuple[str, str]:
    employees = st.session_state.employees
    names = [e.get("display_name", "") for e in employees if e.get("display_name")]
    names = [n for n in names if n]
    if not names:
        st.selectbox(label, ["—"], index=0, key=key)
        return "—", ""
    idx = names.index(current_owner) if current_owner in names else 0
    picked = st.selectbox(label, names, index=idx, key=key)
    owner_id = ""
    for e in employees:
        if e.get("display_name") == picked:
            owner_id = str(e.get("id") or "")
            break
    return picked, owner_id


def resolve_owner_id(owner_name: str) -> str:
    owner_name = _norm(owner_name)
    if not owner_name:
        return ""
    for e in st.session_state.employees:
        if _norm(e.get("display_name", "")) == owner_name:
            return str(e.get("id") or "")
        aliases = e.get("aliases", []) or []
        for a in aliases:
            if _norm(str(a)) == owner_name:
                return str(e.get("id") or "")
    return ""


def capacity_summary(
    tasks: List[TaskSeries],
    window: Tuple[date, date] | None = None,
    default_capacity_per_day: float = 5.0,
    employees: Any = None,  # <-- UI passes this; we don't need it here.
    ws: Optional[date] = None,
    we: Optional[date] = None,
    **_ignored: Any,  # <-- ignore future/extra keywords safely
) -> Dict[str, Any]:
    """Compatibility wrapper around core.capacity_summary.

    UI variants:
      - capacity_summary(tasks, window=(ws,we), employees=..., default_capacity_per_day=..)
      - older call sites might pass ws/we separately

    This wrapper keeps both working and returns the dict structure expected by ui_dashboard.py.
    """
    if window is None and ws is not None and we is not None:
        window = (ws, we)

    if window is None:
        ws0 = date.today()
        we0 = ws0 + timedelta(days=6)
        window = (ws0, we0)

    try:
        return core_capacity_summary(
            tasks=tasks,
            window=window,
            default_capacity_per_day=float(default_capacity_per_day),
        )
    except TypeError:
        ws2, we2 = window
        return {
            "window": (ws2, we2),
            "by_owner": {},
            "totals": {
                "capacity": 0.0,
                "planned": 0.0,
                "remaining": 0.0,
                "pct": 0.0,
                "overdue": 0,
            },
        }


def delete_series(series_id: str) -> None:
    st.session_state.series = [s for s in st.session_state.series if getattr(s, "id", "") != series_id]
    persist()


def save_series(series: TaskSeries) -> None:
    sid = getattr(series, "id", "")
    for i, s in enumerate(st.session_state.series):
        if getattr(s, "id", "") == sid:
            st.session_state.series[i] = series
            break
    persist()


def safe_container(n: int = 1):
    return st.columns(n)


def segmented(options: List[str], key: str, default: str) -> str:
    if default not in options:
        default = options[0]
    idx = options.index(default)
    return st.radio("", options, horizontal=True, index=idx, key=key)


def compute_done_composition(series: TaskSeries, today: date) -> Dict[str, float]:
    parts = getattr(series, "parts", []) or []
    if not parts:
        return {"done": 0.0, "open": 0.0}
    done = 0.0
    open_u = 0.0
    for p in parts:
        try:
            w = float(getattr(p, "weight", 0.0) or 0.0)
        except Exception:
            w = 0.0
        if getattr(p, "done", False):
            done += w
        else:
            open_u += w
    return {"done": done, "open": open_u}


def forecast_eta(series: TaskSeries, today: date) -> Dict[str, Any]:
    try:
        return forecast_eta_units(series, today=today)
    except Exception:
        return {"eta_days": None, "eta_date": None, "velocity": None}


def week_window(start: date) -> Tuple[date, date]:
    ws = start - timedelta(days=start.weekday())
    we = ws + timedelta(days=6)
    return ws, we


def pick_from_list(
    label: str,
    key: str,
    values: List[str],
    current: str,
    require: bool = False,
    list_key: Optional[str] = None,
) -> str:
    values = values or []
    opts = list(values)
    if not require:
        opts = [""] + opts
    opts = opts + [ADD_NEW]

    idx = opts.index(current) if current in opts else 0
    sel = st.selectbox(label, opts, index=idx, key=key)

    if sel == ADD_NEW:
        new_val = st.text_input(f"{label} (new – nur wenn 'Add new...' gewählt)", key=f"{key}_new")
        if new_val and _norm(new_val):
            v = _norm(new_val)
            if list_key:
                st.session_state.lists[list_key] = _list_union(st.session_state.lists.get(list_key, []), [v])
                try:
                    save_lists(st.session_state.lists)
                except Exception:
                    save_lists(
                        st.session_state.lists.get("portfolios", []),
                        st.session_state.lists.get("projects", []),
                        st.session_state.lists.get("themes", []),
                    )
                st.session_state["_last_list_add"] = {"key": list_key, "value": v}
            st.session_state[key] = v
            st.session_state[f"{key}_new"] = ""
            st.rerun()
        return ""
    return sel


def request_done_single(series_id: str, day: date) -> None:
    reqs = st.session_state.get("done_requests", [])
    reqs.append({"series_id": series_id, "day": day.isoformat()})
    st.session_state.done_requests = reqs


def request_done_grid(series_id: str, start_day: date, end_day: date) -> None:
    d = start_day
    while d <= end_day:
        request_done_single(series_id, d)
        d += timedelta(days=1)


def handle_done_requests() -> None:
    if "done_requests" not in st.session_state:
        st.session_state.done_requests = []
    reqs = st.session_state.done_requests
    if not reqs:
        return
    with st.sidebar.expander("✅ Quittierung nötig", expanded=True):
        st.caption("Es gibt DONE-Anfragen, die bestätigt werden müssen.")
        keep = []
        for i, r in enumerate(reqs):
            sid = r.get("series_id")
            day_iso = r.get("day")
            try:
                s = find_series(sid)
            except Exception:
                continue
            row = st.columns([6, 2, 2])
            row[0].write(f"**{s.title}** · {day_iso}")
            if row[1].button(
                "✅ Quittieren",
                key=f"done_ok_{i}_{sid}_{day_iso}",
                type="primary",
                use_container_width=True,
            ):
                mark_done(s, date.fromisoformat(day_iso))
                persist()
            elif row[2].button(
                "✋ Ablehnen",
                key=f"done_no_{i}_{sid}_{day_iso}",
                type="secondary",
                use_container_width=True,
            ):
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

# ------------------ Sidebar stats ------------------

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

_notice = st.session_state.pop("_last_list_add", None)
if isinstance(_notice, dict) and _notice.get("value"):
    try:
        st.toast(f"Neuer Eintrag gespeichert: {_notice['value']}")
    except Exception:
        st.success(f"Neuer Eintrag gespeichert: {_notice['value']}")

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

    owner_names = [e.get("display_name", "") for e in st.session_state.employees if e.get("display_name")]
    qa_owner = st.selectbox(
        "Owner",
        options=owner_names if owner_names else ["—"],
        index=0,
        key="qa_owner",
    )

    qa_owner_id = ""
    if qa_owner != "—":
        for e in st.session_state.employees:
            if e.get("display_name") == qa_owner:
                qa_owner_id = str(e.get("id") or "")
                break

    qa_project = pick_from_list(
        "Project",
        key="qa_project",
        values=st.session_state.lists.get("projects", []),
        current="",
        require=True,
        list_key="projects",
    )
    qa_theme = pick_from_list(
        "Theme",
        key="qa_theme",
        values=st.session_state.lists.get("themes", []),
        current="General",
        require=True,
        list_key="themes",
    )
    qa_portfolio = pick_from_list(
        "Portfolio",
        key="qa_portfolio",
        values=st.session_state.lists.get("portfolios", []),
        current="Default",
        require=True,
        list_key="portfolios",
    )

    qa_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=5, step=1, key="qa_days")
    qa_weight = st.number_input("Weight/Units", min_value=0.0, max_value=1000.0, value=1.0, step=1.0, key="qa_weight")

    if st.button("Add", key="qa_add"):
        if qa_title.strip():
            s = new_series(
                title=qa_title.strip(),
                portfolio=qa_portfolio,
                project=qa_project,
                theme=qa_theme,
                owner="" if qa_owner == "—" else qa_owner,
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

# ------------------ Context for UI modules ------------------

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
    "pick_owner": pick_owner,
    "request_done_single": request_done_single,
    "request_done_grid": request_done_grid,
    "employees": st.session_state.employees,
    "lists": st.session_state.lists,
    "capacity_summary": capacity_summary,
    "compute_units_composition": compute_units_composition,
    "build_burndown_series": build_burndown_series,
    "calc_velocity": calc_velocity,
    "forecast_finish_date": forecast_finish_date,
    "series": st.session_state.series,
    "save_series": save_series,
    "save_employees": save_employees,
    "save_lists": save_lists,
    "resolve_owner_id": resolve_owner_id,
    "safe_container": safe_container,
    "segmented": segmented,
    "compute_done_composition": compute_done_composition,
    "forecast_eta": forecast_eta,
    "week_window": week_window,
    "forecast_eta_units": forecast_eta_units,
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
