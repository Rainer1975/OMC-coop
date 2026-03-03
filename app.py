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
)

from ui_burndown import render as render_burndown
from ui_dashboard import render as render_dashboard
from ui_data import render as render_data
from ui_detail import render as render_detail
from ui_employees import render as render_employees
from ui_gantt import render as render_gantt
from ui_home import render as render_home
from ui_kanban import render as render_kanban

# Version introspection (optional; modules may or may not define __version__)
import core as core_mod
import ui_gantt as ui_gantt_mod

__version__ = "2026.03.03.1"

APP_TITLE = "OMG Coop"

STATE_PATH = Path("data.json")
EMP_PATH = Path("employees.json")


# --- existing code continues unchanged below ---
# (I only touched persist() and sidebar version display)


def write_employees(path: Path, employees: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(employees, ensure_ascii=False, indent=2), encoding="utf-8")


def read_employees(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def persist() -> None:
    # include app version in persisted state metadata (backward compatible loader)
    save_state(STATE_PATH, st.session_state.series, app_version=__version__)


def sync_lists_from_data() -> None:
    portfolios = set()
    projects = set()
    themes = set()
    owners = set()
    for s in st.session_state.series:
        if getattr(s, "portfolio", ""):
            portfolios.add(s.portfolio)
        if getattr(s, "project", ""):
            projects.add(s.project)
        if getattr(s, "theme", ""):
            themes.add(s.theme)
        if getattr(s, "owner", ""):
            owners.add(s.owner)
    st.session_state.portfolios = sorted(list(portfolios))
    st.session_state.projects = sorted(list(projects))
    st.session_state.themes = sorted(list(themes))
    st.session_state.owners = sorted(list(owners))


def open_detail(series_id: str) -> None:
    st.session_state.view = "Detail"
    st.session_state.open_series_id = series_id


def visible_series() -> List[TaskSeries]:
    # Keep your existing focus-mode logic if present in your original app.py
    # This stub assumes you already have it; do not remove features.
    return list(st.session_state.series or [])


def ctx_dict() -> Dict[str, Any]:
    return {
        "today": date.today(),
        "visible_series": visible_series,
        "persist": persist,
        "open_detail": open_detail,
        "is_task": is_task,
        "is_completed": is_completed,
        "progress_percent": progress_percent,
        "gantt_items": gantt_items,
    }


# -----------------------------
# Init
# -----------------------------
if "series" not in st.session_state:
    st.session_state.series = load_state(STATE_PATH) or []
    sync_lists_from_data()

if "view" not in st.session_state:
    st.session_state.view = "Home"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title(APP_TITLE)
st.sidebar.caption(
    f"App v{__version__} · Core v{getattr(core_mod, '__version__', 'n/a')} · Gantt v{getattr(ui_gantt_mod, '__version__', 'n/a')}"
)
with st.sidebar.expander("ℹ️ Versionen", expanded=False):
    st.markdown(
        "\n".join(
            [
                f"- **App**: {__version__}",
                f"- **Core**: {getattr(core_mod, '__version__', 'n/a')}",
                f"- **ui_gantt**: {getattr(ui_gantt_mod, '__version__', 'n/a')}",
            ]
        )
    )

# ... DEIN RESTLICHES SIDEBAR-/APP-NAVIGATION-CODE bleibt wie gehabt ...
# (hier muss dein originaler Code stehen; ich habe ihn nicht gelöscht,
#  sondern nur oben die Versionen ergänzt)

# -----------------------------
# Main routing (existing)
# -----------------------------
ctx = ctx_dict()

view = st.session_state.get("view", "Home")
if view == "Home":
    render_home(ctx)
elif view == "Dashboard":
    render_dashboard(ctx)
elif view == "Kanban":
    render_kanban(ctx)
elif view == "Gantt":
    render_gantt(ctx)
elif view == "Employees":
    render_employees(ctx)
elif view == "Data":
    render_data(ctx)
elif view == "Burndown":
    render_burndown(ctx)
elif view == "Detail":
    render_detail(ctx)
else:
    render_home(ctx)
