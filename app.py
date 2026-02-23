# app.py
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

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
    capacity_summary,
    week_window,
    compute_units_composition,
    forecast_eta_units,
    series_total_units,
)

try:
    from core import critical_path_series  # optional
except Exception:
    critical_path_series = None  # type: ignore


# =========================
# Config / CSS
# =========================
st.set_page_config(page_title="OMG Coop", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.kpi { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
.kpi-label { font-size: 12px; opacity: .65; }
hr { opacity: .25; }
code { font-size: 0.9em; }
</style>
""",
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent

# Data directory for internet deployments:
# - default: repo/app folder (works locally)
# - optional: set env var OMG_COOP_DATA_DIR to a writable path (e.g. a mounted volume)
DATA_DIR = Path(os.getenv("OMG_COOP_DATA_DIR", str(APP_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "data.json"
EMP_FILE = DATA_DIR / "employees.json"
LISTS_FILE = DATA_DIR / "lists.json"

today = date.today()

ADD_NEW = "➕ Add new…"
TASK_STATES = ["PLANNED", "ACTIVE", "BLOCKED", "DONE", "CANCELLED"]


# =========================
# Atomic JSON writes
# =========================
def _acquire_lock(lock_path: Path, timeout_s: float = 5.0, poll_s: float = 0.05) -> int:
    import time as _time

    start = _time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            if (_time.monotonic() - start) >= timeout_s:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            _time.sleep(poll_s)


def _release_lock(fd: int, lock_path: Path) -> None:
    try:
        try:
            os.close(fd)
        finally:
            if lock_path.exists():
                lock_path.unlink()
    except Exception:
        pass


def _atomic_write_json(path: Path, payload: dict) -> None:
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = _acquire_lock(lock_path)
    try:
        fd2, tmp_path = tempfile.mkstemp(prefix="omg_", suffix=".tmp", dir=str(folder))
        try:
            with os.fdopen(fd2, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(path))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    finally:
        _release_lock(fd, lock_path)


# =========================
# Small helpers
# =========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


# =========================
# Employees
# =========================
def load_employees() -> list[dict]:
    if not EMP_FILE.exists():
        return []
    try:
        payload = json.loads(EMP_FILE.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    if isinstance(payload, list):
        names = [str(x).strip() for x in payload if str(x).strip()]
        out = [{"id": _slugify(n), "display_name": n, "aliases": [n]} for n in names]
        out.sort(key=lambda x: _norm(x["display_name"]))
        return out

    if isinstance(payload, dict):
        emps = payload.get("employees", [])
        if isinstance(emps, list) and emps and isinstance(emps[0], str):
            names = [str(x).strip() for x in emps if str(x).strip()]
            out = [{"id": _slugify(n), "display_name": n, "aliases": [n]} for n in names]
            out.sort(key=lambda x: _norm(x["display_name"]))
            return out

        if isinstance(emps, list):
            out: list[dict] = []
            for e in emps:
                if not isinstance(e, dict):
                    continue
                dn = str(e.get("display_name", "")).strip()
                if not dn:
                    continue
                eid = str(e.get("id", "")).strip() or _slugify(dn)
                aliases = e.get("aliases", []) if isinstance(e.get("aliases", []), list) else []
                aliases = [str(a).strip() for a in aliases if str(a).strip()]
                if dn and dn not in aliases:
                    aliases = [dn] + aliases
                out.append({"id": eid, "display_name": dn, "aliases": aliases})
            out.sort(key=lambda x: _norm(x["display_name"]))
            return out

    return []


def save_employees(employees: list[dict]) -> None:
    cleaned: list[dict] = []
    seen = set()
    for e in employees or []:
        if not isinstance(e, dict):
            continue
        dn = str(e.get("display_name", "")).strip()
        if not dn:
            continue
        eid = str(e.get("id", "")).strip() or _slugify(dn)
        if eid in seen:
            continue
        seen.add(eid)
        aliases = e.get("aliases", []) if isinstance(e.get("aliases", []), list) else []
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        if dn and dn not in aliases:
            aliases = [dn] + aliases
        cleaned.append({"id": eid, "display_name": dn, "aliases": aliases})
    cleaned.sort(key=lambda x: _norm(x["display_name"]))
    _atomic_write_json(EMP_FILE, {"schema_version": 1, "employees": cleaned})


def resolve_owner_id(owner_display: str, employees: list[dict]) -> str:
    od = _norm(owner_display)
    for e in employees or []:
        if _norm(e.get("display_name", "")) == od:
            return str(e.get("id", "")).strip()
        for a in e.get("aliases", []) or []:
            if _norm(a) == od:
                return str(e.get("id", "")).strip()
    return _slugify(owner_display) if (owner_display or "").strip() else ""


def display_name_for_owner_id(owner_id: str, employees: list[dict]) -> str:
    oid = (owner_id or "").strip()
    if not oid:
        return ""
    for e in employees or []:
        if str(e.get("id", "")).strip() == oid:
            return str(e.get("display_name", "")).strip()
    return ""


# =========================
# Lists
# =========================
def load_lists() -> dict:
    if not LISTS_FILE.exists():
        return {"portfolios": [], "projects": [], "themes": []}
    try:
        payload = json.loads(LISTS_FILE.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return {"portfolios": [], "projects": [], "themes": []}

        def norm_list(xs: Any) -> list[str]:
            if not isinstance(xs, list):
                return []
            out: list[str] = []
            for x in xs:
                s = re.sub(r"\s+", " ", str(x).strip())
                if s and s not in out:
                    out.append(s)
            return out

        return {
            "portfolios": norm_list(payload.get("portfolios", [])),
            "projects": norm_list(payload.get("projects", [])),
            "themes": norm_list(payload.get("themes", [])),
        }
    except Exception:
        return {"portfolios": [], "projects": [], "themes": []}


def save_lists(portfolios: list[str], projects: list[str], themes: list[str]) -> None:
    def uniq(xs: list[str]) -> list[str]:
        out: list[str] = []
        for x in xs:
            s = re.sub(r"\s+", " ", (x or "").strip())
            if s and s not in out:
                out.append(s)
        return out

    _atomic_write_json(
        LISTS_FILE,
        {
            "portfolios": uniq(portfolios),
            "projects": uniq(projects),
            "themes": uniq(themes),
        },
    )


def _list_union(existing: list[str], incoming: list[str]) -> list[str]:
    out: list[str] = []
    for x in (existing or []) + (incoming or []):
        s = re.sub(r"\s+", " ", (x or "").strip())
        if s and s not in out:
            out.append(s)
    return out


def sync_lists_from_data() -> None:
    pf: list[str] = []
    pr: list[str] = []
    th: list[str] = []

    for s in st.session_state.series:
        pfv = re.sub(r"\s+", " ", (getattr(s, "portfolio", "") or "").strip())
        prv = re.sub(r"\s+", " ", (getattr(s, "project", "") or "").strip())
        thv = re.sub(r"\s+", " ", (getattr(s, "theme", "") or "").strip())
        if pfv:
            pf.append(pfv)
        if prv:
            pr.append(prv)
        if thv:
            th.append(thv)

    cur = st.session_state.lists
    new_pf = _list_union(cur.get("portfolios", []), pf)
    new_pr = _list_union(cur.get("projects", []), pr)
    new_th = _list_union(cur.get("themes", []), th)

    if new_pf != cur.get("portfolios", []) or new_pr != cur.get("projects", []) or new_th != cur.get("themes", []):
        st.session_state.lists["portfolios"] = new_pf
        st.session_state.lists["projects"] = new_pr
        st.session_state.lists["themes"] = new_th
        save_lists(new_pf, new_pr, new_th)


# =========================
# Persistence
# =========================
def persist() -> None:
    save_state(str(DATA_FILE), st.session_state.series)


def load_or_empty() -> list[TaskSeries]:
    loaded = load_state(str(DATA_FILE))
    return loaded if loaded else []


# =========================
# Session init
# =========================
if "series" not in st.session_state:
    st.session_state.series = load_or_empty()
    persist()

if "employees" not in st.session_state:
    st.session_state.employees = load_employees()

if "lists" not in st.session_state:
    st.session_state.lists = load_lists()

if "focus_mode" not in st.session_state:
    st.session_state.focus_mode = False

if "page" not in st.session_state:
    st.session_state.page = "HOME"
if "page_prev" not in st.session_state:
    st.session_state.page_prev = "HOME"
if "open_series" not in st.session_state:
    st.session_state.open_series = None

if "pending_done" not in st.session_state:
    st.session_state.pending_done = None

# Help state
if "help_mode" not in st.session_state:
    st.session_state.help_mode = "full"  # "full" | "beginner"
if "help_target" not in st.session_state:
    st.session_state.help_target = None  # section key

# NAV: wichtig – verhindert "springt zurück auf Home"
if "nav_force_sync" not in st.session_state:
    st.session_state.nav_force_sync = True  # initial einmal synchronisieren

# Backward-compat defaults
changed = False
for s in st.session_state.series:
    if not getattr(s, "owner_id", "").strip():
        s.owner_id = resolve_owner_id(getattr(s, "owner", ""), st.session_state.employees)
        changed = True
    if (not (getattr(s, "owner", "") or "").strip()) and getattr(s, "owner_id", "").strip():
        dn = display_name_for_owner_id(s.owner_id, st.session_state.employees)
        if dn:
            s.owner = dn
            changed = True

    if not hasattr(s, "portfolio") or not (getattr(s, "portfolio", "") or "").strip():
        try:
            s.portfolio = "Default"
            changed = True
        except Exception:
            pass

    if not hasattr(s, "state") or not (getattr(s, "state", "") or "").strip():
        try:
            if is_task(s) and is_completed(s, today):
                s.state = "DONE"
            else:
                s.state = "ACTIVE" if is_task(s) and is_active(s, today) else "PLANNED"
            changed = True
        except Exception:
            pass

if "Default" not in st.session_state.lists.get("portfolios", []):
    st.session_state.lists["portfolios"] = ["Default"] + st.session_state.lists.get("portfolios", [])
    save_lists(
        st.session_state.lists.get("portfolios", []),
        st.session_state.lists.get("projects", []),
        st.session_state.lists.get("themes", []),
    )

sync_lists_from_data()
if changed:
    persist()


# =========================
# UI helpers
# =========================
def safe_container(border: bool = False):
    try:
        return st.container(border=border)
    except TypeError:
        return st.container()


def segmented(label: str, options: list[str], default: str):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options, default=default)
    return st.radio(label, options, index=options.index(default), horizontal=True)


def open_detail(series_id: str) -> None:
    # DETAIL ist absichtlich KEIN Sidebar-Menüpunkt.
    # Sonst kracht Navigation, wenn ein "Detail"-Item entfernt wird oder nicht existiert.
    # Deshalb merken wir uns das aktive Sidebar-Item beim Eintritt und verhindern,
    # dass die Sidebar-Auswahl den Detail-View beim Rerun wieder überschreibt.
    st.session_state.open_series = series_id
    st.session_state.page_prev = st.session_state.page
    st.session_state.nav_select_before_detail = st.session_state.get("nav_select")
    st.session_state.page = "DETAIL"
    st.session_state.nav_force_sync = True  # Sidebar soll stabil bleiben
    st.rerun()


def close_detail() -> None:
    st.session_state.open_series = None
    back = st.session_state.page_prev if st.session_state.page_prev != "DETAIL" else "HOME"
    st.session_state.page = back
    st.session_state.nav_force_sync = True  # <<< wichtig
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


def visible_series() -> list[TaskSeries]:
    out: list[TaskSeries] = []
    for s in st.session_state.series:
        if st.session_state.focus_mode:
            if getattr(s, "is_meta", False):
                continue
            if is_appointment(s):
                continue
            if is_task(s) and is_completed(s, today):
                continue
        out.append(s)
    return out


# =========================
# Pickers
# =========================
def pick_from_list(
    label: str,
    key: str,
    values: list[str],
    current: str = "",
    require: bool = True,
    list_key: str = "",
) -> str:
    values = [re.sub(r"\s+", " ", (v or "").strip()) for v in (values or []) if (v or "").strip()]
    cur = re.sub(r"\s+", " ", (current or "").strip())
    if cur and cur not in values:
        values = [cur] + values

    options = values + [ADD_NEW] if values else [ADD_NEW]
    idx = options.index(cur) if cur in options else 0

    pick = st.selectbox(label, options=options, index=idx, key=key)
    if pick == ADD_NEW:
        new_val = st.text_input(f"{label} (new)", value="", key=f"{key}__new").strip()
        if not new_val:
            return "" if require else ""
        new_val = re.sub(r"\s+", " ", new_val)
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


def pick_owner(
    label: str,
    key: str,
    current_owner_id: str,
    current_owner_display: str,
    employees: list[dict],
) -> tuple[str, str]:
    emps = employees or []
    options = [(e["display_name"], e["id"]) for e in emps if (e.get("id") and e.get("display_name"))]
    options.sort(key=lambda x: _norm(x[0]))
    labels = [x[0] for x in options]
    ids = [x[1] for x in options]

    cur_display = (current_owner_display or "").strip()
    cur_id = (current_owner_id or "").strip()

    if cur_id and not cur_display:
        cur_display = display_name_for_owner_id(cur_id, employees)

    if cur_display and (cur_display not in labels):
        labels = [cur_display] + labels
        ids = [resolve_owner_id(cur_display, employees)] + ids

    labels = labels + [ADD_NEW] if labels else [ADD_NEW]
    idx = labels.index(cur_display) if cur_display in labels else 0

    pick = st.selectbox(label, options=labels, index=idx, key=key)
    if pick == ADD_NEW:
        new_val = st.text_input(f"{label} (new)", value="", key=f"{key}__new").strip()
        if not new_val:
            return "", ""
        return resolve_owner_id(new_val, employees), new_val

    if pick in [x[0] for x in options]:
        oid = ids[[x[0] for x in options].index(pick)]
    else:
        oid = resolve_owner_id(pick, employees)
    return oid, pick


# =========================
# DONE confirmation (unchanged)
# =========================
def request_done_single(series_id: str, d: date, reason: str) -> None:
    if d > today:
        st.warning("Future is disabled: cannot set DONE in the future.")
        return
    st.session_state.pending_done = {"type": "single", "series_id": series_id, "day_iso": d.isoformat(), "reason": reason}
    st.rerun()


def request_done_grid(items: list[dict], reason: str) -> None:
    items = [it for it in (items or []) if date.fromisoformat(it["day_iso"]) <= today]
    if not items:
        st.warning("Nothing to confirm (future disabled).")
        return
    st.session_state.pending_done = {"type": "grid", "items": items, "reason": reason}
    st.rerun()


def clear_pending() -> None:
    st.session_state.pending_done = None


def render_pending_banner() -> None:
    pdn = st.session_state.pending_done
    if not pdn:
        return

    with safe_container(border=True):
        if pdn["type"] == "single":
            s = find_series(pdn["series_id"])
            d = date.fromisoformat(pdn["day_iso"])
            if d in getattr(s, "done_days", set()):
                clear_pending()
                st.rerun()

            st.warning(f"Quittierung nötig: **{s.title}** ({s.owner}) am **{d.isoformat()}** als **DONE** setzen?")
            st.caption(pdn.get("reason", ""))

            c1, c2, c3 = st.columns([2, 2, 6])
            if c1.button("✅ Quittieren (DONE setzen)", key=f"confirm_single_{s.series_id}_{d}"):
                if "actor" in mark_done.__code__.co_varnames:
                    mark_done(s, d, actor="ui")
                else:
                    mark_done(s, d)
                try:
                    if is_task(s) and is_completed(s, today):
                        s.state = "DONE"
                except Exception:
                    pass
                persist()
                clear_pending()
                st.rerun()
            if c2.button("↩️ Abbrechen", key=f"cancel_single_{s.series_id}_{d}"):
                clear_pending()
                st.rerun()
            c3.caption("DONE entfernen ist sofort (für Korrekturen).")

        elif pdn["type"] == "grid":
            items = pdn.get("items") or []
            st.warning(f"Quittierung nötig: **{len(items)}** Grid-Einträge als DONE setzen?")
            st.caption(pdn.get("reason", ""))

            c1, c2, c3 = st.columns([2, 2, 6])
            if c1.button("✅ Quittieren (Grid DONE)", key="confirm_grid"):
                by_series: dict[str, list[date]] = {}
                for it in items:
                    sid = it["series_id"]
                    dd = date.fromisoformat(it["day_iso"])
                    if dd > today:
                        continue
                    by_series.setdefault(sid, []).append(dd)

                changed_total = 0
                for sid, ds in by_series.items():
                    s = find_series(sid)
                    if "actor" in bulk_set_done.__code__.co_varnames:
                        changed_total += bulk_set_done(s, ds, set_done=True, actor=s.owner)
                    else:
                        changed_total += bulk_set_done(s, ds, set_done=True)
                    try:
                        if is_task(s) and is_completed(s, today):
                            s.state = "DONE"
                        elif is_task(s) and (getattr(s, "state", "") == "PLANNED") and is_active(s, today):
                            s.state = "ACTIVE"
                    except Exception:
                        pass

                persist()
                clear_pending()
                st.success(f"Grid quittiert. Änderungen: {changed_total}.")
                st.rerun()

            if c2.button("↩️ Abbrechen", key="cancel_grid"):
                clear_pending()
                st.rerun()
            c3.caption("DONE entfernen ist sofort (für Korrekturen).")


# =========================
# Burndown helpers (existing ctx expects these)
# =========================
def avg_last_window(values: list[int], end_idx: int, window: int) -> float:
    if not values:
        return 0.0
    end_idx = max(0, min(end_idx, len(values) - 1))
    window = max(1, int(window))
    start = max(0, end_idx - window + 1)
    chunk = values[start : end_idx + 1]
    return float(sum(chunk)) / float(len(chunk)) if chunk else 0.0


def rolling_sum(values: list[int], window: int) -> list[int]:
    window = max(1, int(window))
    out: list[int] = []
    s = 0
    q = deque()
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.popleft()
        out.append(int(s))
    return out


def compute_done_composition(task_series: list[TaskSeries]):
    tasks = [s for s in task_series if is_task(s)]
    if not tasks:
        return [], [], [], []

    min_day = min(s.start for s in tasks)
    max_day = max(s.end for s in tasks)

    all_days: list[date] = []
    d = min_day
    while d <= max_day:
        all_days.append(d)
        d += timedelta(days=1)

    done_port: list[int] = []
    remaining_port: list[int] = []
    comp_by_day: list[list[dict]] = []

    for day in all_days:
        done_today = 0
        rem = 0
        comp: list[dict] = []
        for s in tasks:
            tdays = total_days(s)
            if day < s.start:
                rem += tdays
                continue
            if day > s.end:
                rem += 0
                continue

            done_upto = sum(1 for x in getattr(s, "done_days", set()) if s.start <= x <= day)
            rem += max(0, tdays - done_upto)

            if day in getattr(s, "done_days", set()):
                done_today += 1
                comp.append({"series_id": s.series_id, "owner": s.owner, "project": s.project, "task": s.title})

        done_port.append(done_today)
        remaining_port.append(rem)
        comp_by_day.append(comp)

    return all_days, done_port, remaining_port, comp_by_day


def forecast_eta(all_days: list[date], done_port: list[int], remaining_port: list[int], today_: date, window: int):
    if not all_days:
        return None, None, None

    last_le_today = 0
    last_done = None
    for i, dd in enumerate(all_days):
        if dd <= today_:
            last_le_today = i
            if done_port[i] > 0:
                last_done = i
        else:
            break

    anchor_idx = last_done if last_done is not None else last_le_today
    anchor_day = all_days[anchor_idx]
    rem_anchor = remaining_port[anchor_idx]
    vel = avg_last_window(done_port, anchor_idx, int(window))

    if vel <= 0 and window < 14:
        vel = avg_last_window(done_port, anchor_idx, 14)

    if rem_anchor <= 0:
        return anchor_day, 0.0, anchor_day
    if vel <= 0:
        return anchor_day, 0.0, None

    days_needed = int(math.ceil(rem_anchor / vel))
    eta = anchor_day + timedelta(days=days_needed)
    return anchor_day, vel, eta


def plot_burndown_matplotlib(days: list[date], remaining: list[int], title: str, xlim=None):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(days, remaining, marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_ylabel("Remaining done-days")
    ax.set_xlabel("Date")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    fig.autofmt_xdate()
    return fig


def plot_velocity_plotly(all_days: list[date], vel7: list[int], comp_by_day: list[list[dict]]):
    hover = []
    for i, d in enumerate(all_days):
        items = comp_by_day[i]
        if not items:
            hover.append(f"{d.isoformat()}<br><b>0</b> done-days")
            continue
        lines = []
        for it in items[:12]:
            lines.append(f"- {it['owner']} · {it['project']} · {it['task']}")
        if len(items) > 12:
            lines.append(f"... +{len(items)-12} more")
        hover.append(
            f"{d.isoformat()}<br><b>{vel7[i]}</b> (7d rolling)<br>"
            f"Done on day:<br>" + "<br>".join(lines)
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=all_days,
            y=vel7,
            mode="lines+markers",
            name="Velocity (7d rolling)",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Velocity (7-day rolling done-days)",
        xaxis_title="Date",
        yaxis_title="Done-days (rolling 7d sum)",
        hovermode="closest",
        margin=dict(l=40, r=20, t=55, b=40),
        height=360,
    )
    return fig


# =========================
# Help integration (context mapping)
# =========================
PAGE_TO_HELP = {
    "HOME": "home",
    "INBOX": "inbox",
    "TODAY": "today",
    "DETAIL": "detail",
    "KANBAN": "kanban",
    "GANTT": "gantt",
    "BURNDOWN": "burndown",
    "DASHBOARD": "dashboard",
    "META": "meta",
    "ADMIN": "admin",
    "DATA": "data",
    # help pages map to themselves
    "HELP": "wozu",
    "BEGINNER": "wozu",
}


def open_help(mode: str) -> None:
    # mode: "full" | "beginner"
    cur_page = st.session_state.page
    st.session_state.help_target = PAGE_TO_HELP.get(cur_page, "wozu")
    st.session_state.help_mode = "beginner" if mode == "beginner" else "full"
    st.session_state.page_prev = st.session_state.page
    st.session_state.page = "HELP"
    st.session_state.nav_force_sync = True  # <<< wichtig
    st.rerun()


# =========================
# Sidebar (wie vorher) + Help Icons
# =========================
st.sidebar.markdown("### OMG Coop")

st.sidebar.toggle(
    "Focus mode (only active tasks, no META, no appointments)",
    value=st.session_state.focus_mode,
    key="focus_mode",
)


def series_counts(series_list: list[TaskSeries]):
    tasks = [s for s in series_list if is_task(s)]
    due_today = [
        s
        for s in tasks
        if is_active(s, today)
        and (today not in getattr(s, "done_days", set()))
        and (not is_completed(s, today))
        and (getattr(s, "state", "") not in ("CANCELLED",))
    ]
    overdue = [s for s in tasks if is_overdue(s, today)]
    active = [s for s in tasks if not is_completed(s, today)]
    meta_open = [s for s in tasks if bool(getattr(s, "is_meta", False)) and not is_completed(s, today)]
    return len(due_today), len(overdue), len(active), len(meta_open)


visible = visible_series()
due_n, overdue_n, active_n, meta_n = series_counts(visible)

k1, k2, k3, k4 = st.sidebar.columns(4)
k1.markdown(f"<div class='kpi'>{due_n}</div><div class='kpi-label'>Today</div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'>{overdue_n}</div><div class='kpi-label'>Overdue</div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'>{active_n}</div><div class='kpi-label'>Active</div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi'>{meta_n}</div><div class='kpi-label'>Meta</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Help icons: always available + context-sensitive
hc1, hc2 = st.sidebar.columns(2)
if hc1.button("❓ Hilfe", key="btn_help_full", use_container_width=True):
    open_help("full")
if hc2.button("🧑‍🎓 Anfänger", key="btn_help_beginner", use_container_width=True):
    open_help("beginner")

st.sidebar.caption("Hilfe ist kontextsensitiv: öffnet automatisch den passenden Abschnitt zur aktuellen Seite.")
st.sidebar.markdown("---")


# =========================
# Navigation (letzter Eintrag = Gesamte Anleitung)
# FIX: Dropdown darf nicht auf Home zurückspringen.
# =========================
NAV_SECTIONS = [
    ("Start", [("Home", "HOME")]),
    # DETAIL ist kein eigener Menüpunkt (sonst springt Navigation bei Open/Back gerne falsch).
    ("Dateneingabe", [("Inbox", "INBOX"), ("Today", "TODAY"), ("Employees", "EMPLOYEES")]),
    ("Reporting", [("Kanban", "KANBAN"), ("Dashboard", "DASHBOARD"), ("Burndown", "BURNDOWN"), ("Gantt", "GANTT"), ("Meta", "META")]),
    ("Wartung", [("Admin", "ADMIN"), ("Data", "DATA")]),
    ("Hilfe", [("Anfänger (15 Min)", "BEGINNER"), ("Gesamte Anleitung", "HELP")]),
]

display_to_code: dict[str, str] = {}
code_to_display: dict[str, str] = {}
display_options: list[str] = []

for sec, items in NAV_SECTIONS:
    for label, code in items:
        disp = f"{sec} › {label}"
        display_to_code[disp] = code
        code_to_display[code] = disp
        display_options.append(disp)

# Zielanzeige aus aktueller Page ableiten
# DETAIL ist kein Sidebar-Menüpunkt. Wenn wir im Detail sind, soll die Sidebar optisch
# beim vorherigen Menüpunkt bleiben (sonst springt sie auf HOME und überschreibt DETAIL).
if st.session_state.page == "DETAIL":
    cur_disp = st.session_state.get("nav_select_before_detail") or code_to_display.get(
        st.session_state.get("page_prev") or "HOME",
        display_options[0],
    )
else:
    cur_code = st.session_state.page if st.session_state.page in code_to_display else "HOME"
    cur_disp = code_to_display.get(cur_code, display_options[0])

# Nur bei programmatischen Page-Wechseln: nav_select auf aktuelle Page setzen
# Achtung: im DETAIL-Modus setzen wir nav_select NICHT auf HOME.
if st.session_state.nav_force_sync:
    st.session_state["nav_select"] = cur_disp
    st.session_state.nav_force_sync = False

# selectbox rendern
current_sel = st.session_state.get("nav_select", cur_disp)
current_idx = display_options.index(current_sel) if current_sel in display_options else 0

picked_disp = st.sidebar.selectbox(
    label="Navigation",
    options=display_options,
    index=current_idx,
    label_visibility="collapsed",
    key="nav_select",
)

# User-Auswahl -> page setzen
picked_page = display_to_code.get(picked_disp, "HOME")

# Sonderfall DETAIL:
# Wenn wir im Detail sind, bleibt die Sidebar optisch auf dem vorherigen Menüpunkt stehen.
# Das darf den Detail-View nicht sofort wieder weg-navigieren.
if st.session_state.page == "DETAIL":
    before = st.session_state.get("nav_select_before_detail")
    # Nur wenn der User im Dropdown wirklich etwas anderes auswählt, verlassen wir DETAIL.
    if before is not None and picked_disp == before:
        picked_page = "DETAIL"

if picked_page != st.session_state.page:
    st.session_state.page_prev = st.session_state.page
    st.session_state.page = picked_page
    # Beim Verlassen von DETAIL: Marker löschen
    if picked_page != "DETAIL":
        st.session_state.pop("nav_select_before_detail", None)
    # absichtlich KEIN nav_force_sync hier – sonst überschreiben wir den User-Klick


# Quick add (task) – wie vorher
with st.sidebar.expander("Quick add (task)", expanded=False):
    q_title = st.text_input("Title", key="qa_title")

    q_portfolio = pick_from_list(
        "Portfolio",
        key="qa_portfolio_pick",
        values=st.session_state.lists.get("portfolios", []),
        current="Default",
        require=True,
        list_key="portfolios",
    )
    q_project = pick_from_list(
        "Project",
        key="qa_project_pick",
        values=st.session_state.lists.get("projects", []),
        current="",
        require=True,
        list_key="projects",
    )
    q_theme = pick_from_list(
        "Theme",
        key="qa_theme_pick",
        values=st.session_state.lists.get("themes", []),
        current="General",
        require=True,
        list_key="themes",
    )

    q_owner_id, q_owner_display = pick_owner(
        "Owner",
        key="qa_owner_pick",
        current_owner_id="",
        current_owner_display="",
        employees=st.session_state.employees,
    )

    q_state = st.selectbox("State", options=TASK_STATES, index=1, key="qa_state")  # ACTIVE default
    q_meta = st.checkbox("META", value=False, key="qa_meta", disabled=st.session_state.focus_mode)

    c1, c2 = st.columns(2)
    q_start = c1.date_input("Start", today, key="qa_start")
    q_end = c2.date_input("End", today, key="qa_end")

    if st.button("Create", key="qa_create"):
        if not (q_title or "").strip():
            st.warning("Title is required.")
        elif not (q_portfolio or "").strip():
            st.warning("Portfolio is required.")
        elif not (q_project or "").strip():
            st.warning("Project is required.")
        elif not (q_theme or "").strip():
            st.warning("Theme is required.")
        elif not (q_owner_display or "").strip():
            st.warning("Owner is required.")
        elif q_start > q_end:
            st.warning("Start must be <= End.")
        else:
            is_meta_val = False if st.session_state.focus_mode else bool(q_meta)
            s = new_series(
                title=q_title,
                project=q_project,
                theme=q_theme,
                owner=q_owner_display,
                owner_id=q_owner_id,
                start=q_start,
                end=q_end,
                is_meta=is_meta_val,
                kind="task",
                portfolio=q_portfolio,
                state=q_state,
            )
            st.session_state.series.append(s)
            persist()
            sync_lists_from_data()
            st.success("Created.")
            st.rerun()

# Banner (Quittierung)
render_pending_banner()


# =========================
# Context for pages
# =========================
ctx = {
    "today": today,
    "ADD_NEW": ADD_NEW,
    "TASK_STATES": TASK_STATES,
    "DATA_FILE": DATA_FILE,
    "EMP_FILE": EMP_FILE,
    "LISTS_FILE": LISTS_FILE,
    "safe_container": safe_container,
    "segmented": segmented,
    "appointment_label": appointment_label,
    "visible_series": visible_series,
    "find_series": find_series,
    "open_detail": open_detail,
    "close_detail": close_detail,
    "delete_series": delete_series,
    "pick_from_list": pick_from_list,
    "pick_owner": pick_owner,
    "persist": persist,
    "sync_lists_from_data": sync_lists_from_data,
    "request_done_single": request_done_single,
    "request_done_grid": request_done_grid,
    "employees": st.session_state.employees,
    "save_employees": save_employees,
    "resolve_owner_id": resolve_owner_id,
    "display_name_for_owner_id": display_name_for_owner_id,
    "lists": st.session_state.lists,
    "save_lists": save_lists,
    # help
    "help_mode": st.session_state.help_mode,
    "help_target": st.session_state.help_target,
    # core
    "TaskPart": TaskPart,
    "TaskSeries": TaskSeries,
    "new_series": new_series,
    "new_part": new_part,
    "mark_done": mark_done,
    "unmark_done": unmark_done,
    "bulk_set_done": bulk_set_done,
    "progress_percent": progress_percent,
    "total_days": total_days,
    "is_task": is_task,
    "is_appointment": is_appointment,
    "is_active": is_active,
    "is_completed": is_completed,
    "is_overdue": is_overdue,
    "can_depend_series": can_depend_series,
    "would_create_cycle": would_create_cycle,
    "gantt_items": gantt_items,
    "critical_path_series": critical_path_series,
    # reporting
    "compute_done_composition": compute_done_composition,
    "forecast_eta": forecast_eta,
    # units-based reporting (self-explaining)
    "compute_units_composition": compute_units_composition,
    "forecast_eta_units": forecast_eta_units,
    "series_total_units": series_total_units,
    "plot_burndown_matplotlib": plot_burndown_matplotlib,
    "plot_velocity_plotly": plot_velocity_plotly,
    "rolling_sum": rolling_sum,
    # capacity
    "capacity_summary": capacity_summary,
    "week_window": week_window,
}

page = st.session_state.page


# =========================
# Routing
# =========================
if page == "HOME":
    from ui_home import render

    render(ctx)

elif page == "INBOX":
    from ui_inbox import render

    render(ctx)

elif page == "KANBAN":
    from ui_kanban import render

    render(ctx)

elif page == "TODAY":
    from ui_today import render

    render(ctx)

elif page == "EMPLOYEES":
    from ui_employees import render

    render(ctx)

elif page == "DASHBOARD":
    from ui_dashboard import render

    render(ctx)

elif page == "BURNDOWN":
    from ui_burndown import render

    render(ctx)

elif page == "GANTT":
    from ui_gantt import render

    render(ctx)

elif page == "META":
    from ui_meta import render

    render(ctx)

elif page == "ADMIN":
    from ui_admin import render

    render(ctx)

elif page == "DATA":
    from ui_data import render

    render(ctx)

elif page == "DETAIL":
    from ui_detail import render

    render(ctx)

elif page == "BEGINNER":
    # same help page, just set mode and render
    st.session_state.help_mode = "beginner"
    st.session_state.help_target = PAGE_TO_HELP.get(st.session_state.page_prev, "wozu")
    st.session_state.nav_force_sync = True  # <<< wichtig
    from ui_help import render

    render(ctx)

elif page == "HELP":
    from ui_help import render

    render(ctx)

else:
    st.session_state.page = "HOME"
    st.session_state.nav_force_sync = True  # <<< wichtig
    st.rerun()
