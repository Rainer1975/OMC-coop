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

__version__ = "2026.03.03.5"

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
        x = _norm(str(x))
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

    Historical compatibility:
    - canonical: list[dict]
    - some exports/imports: {"schema_version": 1, "employees": [...]}
    - legacy/dirty: list[str] (names) or list[str(dict-repr)]

    Output is normalized into: list[{id, display_name, aliases}]
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
        """Best-effort parse of a string that looks like a dict (python or json)."""
        s = (s or "").strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        # Try JSON first
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        # Try python literal
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

    # unwrap schema container if present
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
            # item can be dict or string
            if isinstance(item, str):
                # could be a dict-repr string
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

            # Some broken states had display_name containing the whole dict as a string.
            # Try to recover it.
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
                    out[k] = [_norm(str(x)) for x in data[k] if _norm(str(x))]
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
                v = _norm(str(v))
                if v:
                    out.append(v)
            else:
                v = _norm(str(item))
                if v:
                    out.append(v)
        return out
    v = _norm(str(x))
    return [v] if v else []


def save_lists(*args, **kwargs) -> None:
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
    save_state(STATE_PATH, st.session_state.series, app_version=__version__)


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
    st.session_state.open_series = str(series_id)
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
    sid = str(series_id)
    for x in st.session_state.series:
        if str(getattr(x, "series_id", getattr(x, "id", ""))) == sid:
            return x
    raise KeyError(sid)


def delete_series(series_id: str) -> None:
    sid = str(series_id)
    st.session_state.series = [
        x
        for x in st.session_state.series
        if str(getattr(x, "series_id", getattr(x, "id", ""))) != sid
    ]
    persist()


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
    values = [_norm(str(v)) for v in (values or []) if _norm(str(v))]
    cur = _norm(str(current))

    if cur and cur not in values:
        values = [cur] + values

    options = values + [ADD_NEW] if values else [ADD_NEW]
    idx = options.index(cur) if cur in options else 0

    pick = st.selectbox(label, options=options, index=idx, key=key)

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
            prev = list(st.session_state.lists.get(list_key, []) or [])
            if new_val not in prev:
                st.session_state.lists[list_key] = _list_union(prev, [new_val])
                save_lists(
                    st.session_state.lists.get("portfolios", []),
                    st.session_state.lists.get("projects", []),
                    st.session_state.lists.get("themes", []),
                )
                try:
                    st.session_state[key] = new_val
                except Exception:
                    pass
                try:
                    st.session_state[new_key] = ""
                except Exception:
                    pass
                st.session_state["_last_list_add"] = {"list_key": list_key, "value": new_val}
        return new_val

    return pick


def pick_owner(
    label: str,
    key: str,
    current_owner_id: str,
    current_owner_display: str,
    employees: List[Dict[str, Any]],
) -> Tuple[str, str]:
    cur_id = str(current_owner_id or "").strip()
    cur_name = str(current_owner_display or "").strip()

    opts: List[Tuple[str, str]] = []
    for e in employees or []:
        eid = str(e.get("id") or "").strip()
        dn = str(e.get("display_name") or "").strip()
        if not eid or not dn:
            continue
        opts.append((dn, eid))

    seen = set()
    uniq: List[Tuple[str, str]] = []
    for dn, eid in opts:
        k = (dn, eid)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((dn, eid))
    opts = uniq

    if cur_id and cur_name and (cur_name, cur_id) not in opts:
        opts = [(cur_name, cur_id)] + opts

    display_options = ["—"] + [dn for dn, _eid in opts]
    idx = display_options.index(cur_name) if cur_name and cur_name in display_options else 0

    picked_name = st.selectbox(label, options=display_options, index=idx, key=key)

    if picked_name == "—":
        return "", ""

    for dn, eid in opts:
        if dn == picked_name:
            return eid, dn

    return cur_id, picked_name


def resolve_owner_id(owner_display: str, employees: List[Dict[str, Any]]) -> str:
    owner_display = str(owner_display or "").strip()
    if not owner_display:
        return ""
    for e in employees or []:
        if str(e.get("display_name") or "").strip() == owner_display:
            return str(e.get("id") or "").strip()
    return ""


def appointment_label(s: TaskSeries) -> str:
    title = getattr(s, "title", "")
    loc = getattr(s, "location", "")
    t0 = getattr(s, "time_start", "")
    t1 = getattr(s, "time_end", "")
    bits = [b for b in [title, f"{t0}-{t1}" if (t0 or t1) else "", loc] if b]
    return " · ".join(bits)


def segmented(label: str, options: List[str], key: str, index: int = 0) -> str:
    # simple fallback segmented control
    return st.radio(label, options=options, key=key, index=index, horizontal=True)


def safe_container() -> Any:
    return st.container()


def week_window(today: date) -> Tuple[date, date]:
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start, end


def capacity_summary(
    tasks_all: List[TaskSeries],
    employees: List[Dict[str, Any]],
    today: date,
    window: str = "week",
    default_capacity_per_day: float = 5.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    return core_capacity_summary(
        tasks_all,
        employees=employees,
        today=today,
        window=window,
        default_capacity_per_day=default_capacity_per_day,
        **kwargs,
    )


def compute_done_composition(tasks: List[TaskSeries]):
    return compute_units_composition(tasks)


def forecast_eta(all_days, done_per_day, remaining_per_day, today: date, window_business_days: int = 10):
    return forecast_eta_units(all_days, done_per_day, remaining_per_day, today, window_business_days)


# ✅ REQUIRED by ui_kanban.py
def request_done_single(series_id: str, day: date, reason: str = "") -> None:
    st.session_state.done_requests = st.session_state.get("done_requests", [])
    st.session_state.done_requests.append({"series_id": series_id, "day_iso": day.isoformat(), "reason": reason or ""})


# ✅ REQUIRED by ui_kanban.py
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
            row = st.columns([6, 2, 2])
            row[0].write(f"**{s.title}** · {day_iso}")
            if reason:
                row[0].caption(reason)
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

# ------------------ Sidebar (header + quick stats) ------------------

st.sidebar.title(APP_TITLE)

with st.sidebar.expander("Versionen", expanded=False):
    st.markdown(
        "\n".join(
            [
                f"- **App**: {__version__}",
                f"- **Core**: {getattr(__import__('core'), '__version__', 'n/a')}",
                f"- **ui_gantt**: {getattr(__import__('ui_gantt'), '__version__', 'n/a')}",
            ]
        )
    )

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

# ------------------ Page render context ------------------

# NOTE: keep ctx complete; ui_* modules assume keys exist.
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

    # ✅ FIX for ui_kanban.py
    "request_done_single": request_done_single,
    "request_done_grid": request_done_grid,

    "employees": st.session_state.employees,
    "lists": st.session_state.lists,
    "capacity_summary": capacity_summary,
    "compute_units_composition": compute_units_composition,
    "build_burndown_series": build_burndown_series,
    "calc_velocity": calc_velocity,
    "forecast_finish_date": forecast_finish_date,
    "forecast_eta_units": forecast_eta_units,

    # REQUIRED by ui_data.py (keep)
    "series": st.session_state.series,
    "save_employees": save_employees,
    "save_lists": save_lists,
    "resolve_owner_id": resolve_owner_id,
    "DATA_FILE": str(STATE_PATH),
    "EMP_FILE": str(EMP_PATH),
    "LISTS_FILE": str(LISTS_PATH),
}

# ------------------ Main render ------------------

page = st.session_state.page
if page == "HOME":
    render_home(ctx)
elif page == "KANBAN":
    render_kanban(ctx)
elif page == "GANTT":
    render_gantt(ctx)
elif page == "BURNDOWN":
    render_burndown(ctx)
elif page == "DASHBOARD":
    render_dashboard(ctx)
elif page == "EMPLOYEES":
    render_employees(ctx)
elif page == "DATA":
    render_data(ctx)
elif page == "ADMIN":
    render_admin(ctx)
elif page == "DETAIL":
    render_detail(ctx)
else:
    render_home(ctx)
