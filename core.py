# core.py
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# =========================
# Versioning
# =========================
__version__ = "2026.03.03.4"
STATE_SCHEMA_VERSION = 2


# =========================
# Basics
# =========================
def _iso(d: date) -> str:
    return d.isoformat()


def _parse_date(x: Any) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return date.fromisoformat(s[:10])
        except Exception:
            return None
    return None


# =========================
# Atomic file write (Streamlit-safe)
# =========================
def _acquire_lock(lock_path: Path, timeout_s: float = 5.0, poll_s: float = 0.05) -> int:
    start = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            if (time.monotonic() - start) >= timeout_s:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            time.sleep(poll_s)


def _release_lock(fd: int, lock_path: Path) -> None:
    try:
        try:
            os.close(fd)
        finally:
            if lock_path.exists():
                lock_path.unlink()
    except Exception:
        pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = _acquire_lock(lock_path)
    try:
        fd2, tmp_path = tempfile.mkstemp(prefix="omg_", suffix=".tmp", dir=str(path.parent))
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
# Data model
# =========================
@dataclass
class TaskPart:
    part_id: str
    label: str
    start: date
    end: date
    weight: float = 100.0
    predecessors: List[str] = field(default_factory=list)  # part_ids within same series
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_id": self.part_id,
            "label": self.label,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "weight": float(self.weight),
            "predecessors": list(self.predecessors or []),
            "meta": dict(self.meta or {}),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskPart":
        st = _parse_date(d.get("start")) or date.today()
        en = _parse_date(d.get("end")) or st
        preds = d.get("predecessors") or []
        if not isinstance(preds, list):
            preds = []
        preds = [str(x).strip() for x in preds if str(x).strip()]
        return TaskPart(
            part_id=str(d.get("part_id") or ""),
            label=str(d.get("label") or ""),
            start=st,
            end=en,
            weight=float(d.get("weight") or 100.0),
            predecessors=preds,
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class TaskSeries:
    series_id: str
    title: str
    start: date
    end: date

    kind: str = "task"  # "task" | "appointment"
    is_meta: bool = False

    portfolio: str = "Default"
    project: str = ""
    theme: str = "General"

    owner: str = ""
    owner_id: str = ""

    state: str = "ACTIVE"  # PLANNED | ACTIVE | BLOCKED | DONE | CANCELLED

    # appointment optional fields
    time_start: str = ""
    time_end: str = ""
    location: str = ""

    parts: List[TaskPart] = field(default_factory=list)

    # series dependencies: list of predecessor SERIES IDS
    predecessors: List[str] = field(default_factory=list)

    # done tracking
    done_days: Set[date] = field(default_factory=set)

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "title": self.title,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "kind": self.kind,
            "is_meta": bool(self.is_meta),
            "portfolio": self.portfolio,
            "project": self.project,
            "theme": self.theme,
            "owner": self.owner,
            "owner_id": self.owner_id,
            "state": self.state,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "location": self.location,
            "parts": [p.to_dict() for p in (self.parts or [])],
            "predecessors": list(self.predecessors or []),
            "done_days": sorted([_iso(d) for d in (self.done_days or set())]),
            "meta": dict(self.meta or {}),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskSeries":
        st = _parse_date(d.get("start")) or date.today()
        en = _parse_date(d.get("end")) or st

        parts = d.get("parts") or []
        if not isinstance(parts, list):
            parts = []
        parts_obj: List[TaskPart] = []
        for x in parts:
            try:
                if isinstance(x, dict):
                    parts_obj.append(TaskPart.from_dict(x))
            except Exception:
                continue

        preds = d.get("predecessors") or []
        if not isinstance(preds, list):
            preds = []
        preds = [str(x).strip() for x in preds if str(x).strip()]

        dd = d.get("done_days") or []
        if not isinstance(dd, list):
            dd = []
        done_set: Set[date] = set()
        for x in dd:
            pd = _parse_date(x)
            if pd:
                done_set.add(pd)

        meta = d.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        return TaskSeries(
            series_id=str(d.get("series_id") or ""),
            title=str(d.get("title") or ""),
            start=st,
            end=en,
            kind=str(d.get("kind") or "task"),
            is_meta=bool(d.get("is_meta") or False),
            portfolio=str(d.get("portfolio") or "Default"),
            project=str(d.get("project") or ""),
            theme=str(d.get("theme") or "General"),
            owner=str(d.get("owner") or ""),
            owner_id=str(d.get("owner_id") or ""),
            state=str(d.get("state") or "ACTIVE"),
            time_start=str(d.get("time_start") or ""),
            time_end=str(d.get("time_end") or ""),
            location=str(d.get("location") or ""),
            parts=parts_obj,
            predecessors=preds,
            done_days=done_set,
            meta=meta,
        )


# =========================
# IDs
# =========================
def _make_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{ts}"


# =========================
# Constructors
# =========================
def new_part(
    label: str,
    start: date,
    end: date,
    weight: float = 100.0,
    predecessors: Optional[List[str]] = None,
) -> TaskPart:
    return TaskPart(
        part_id=_make_id("p"),
        label=str(label or ""),
        start=start,
        end=end,
        weight=float(weight or 0.0),
        predecessors=list(predecessors or []),
        meta={},
    )


def new_series(
    *,
    title: str,
    portfolio: str,
    project: str,
    theme: str,
    owner: str,
    owner_id: str,
    start: date,
    end: date,
    is_meta: bool,
    kind: str = "task",
    state: str = "ACTIVE",
) -> TaskSeries:
    return TaskSeries(
        series_id=_make_id("s"),
        title=str(title or ""),
        start=start,
        end=end,
        kind=str(kind or "task"),
        is_meta=bool(is_meta),
        portfolio=str(portfolio or "Default"),
        project=str(project or ""),
        theme=str(theme or "General"),
        owner=str(owner or ""),
        owner_id=str(owner_id or ""),
        state=str(state or "ACTIVE"),
        parts=[],
        predecessors=[],
        done_days=set(),
        meta={},
    )


# =========================
# Predicates / status
# =========================
def is_task(s: TaskSeries) -> bool:
    return str(getattr(s, "kind", "task") or "task").lower() == "task"


def is_appointment(s: TaskSeries) -> bool:
    return str(getattr(s, "kind", "") or "").lower() == "appointment"


def is_completed(s: TaskSeries, today: date) -> bool:
    try:
        if today in (getattr(s, "done_days", set()) or set()):
            return True
    except Exception:
        pass
    try:
        stt = str(getattr(s, "state", "") or "").upper()
        if stt == "DONE" and getattr(s, "end", today) <= today:
            return True
    except Exception:
        pass
    return False


def is_active(s: TaskSeries) -> bool:
    stt = str(getattr(s, "state", "") or "").upper()
    return stt == "ACTIVE"


def is_overdue(s: TaskSeries, today: date) -> bool:
    try:
        return is_task(s) and getattr(s, "end", today) < today and not is_completed(s, today)
    except Exception:
        return False


# =========================
# Done operations
# =========================
def mark_done(s: TaskSeries, d: date) -> None:
    if d is None:
        return
    if not isinstance(getattr(s, "done_days", None), set):
        try:
            s.done_days = set()
        except Exception:
            return
    s.done_days.add(d)


def unmark_done(s: TaskSeries, d: date) -> None:
    try:
        if isinstance(getattr(s, "done_days", None), set) and d in s.done_days:
            s.done_days.remove(d)
    except Exception:
        pass


def bulk_set_done(series_list: List[TaskSeries], items: List[Dict[str, str]]) -> None:
    idx = {s.series_id: s for s in (series_list or [])}
    for it in items or []:
        sid = str(it.get("series_id") or "")
        dd = _parse_date(it.get("day_iso"))
        if not sid or not dd:
            continue
        s = idx.get(sid)
        if not s:
            continue
        mark_done(s, dd)


# =========================
# Date helpers
# =========================
def total_days(s: TaskSeries) -> int:
    try:
        st = getattr(s, "start", None)
        en = getattr(s, "end", None)
        if not isinstance(st, date) or not isinstance(en, date):
            return 0
        return max(1, (en - st).days + 1)
    except Exception:
        return 0


def progress_percent(s: TaskSeries, today: date) -> float:
    try:
        td = total_days(s)
        if td <= 0:
            return 0.0
        dd = getattr(s, "done_days", set()) or set()
        if not isinstance(dd, set):
            dd = set()
        done_in_range = [d for d in dd if getattr(s, "start") <= d <= getattr(s, "end") and d <= today]
        return max(0.0, min(1.0, float(len(done_in_range)) / float(td)))
    except Exception:
        return 0.0


def remaining_days(s: TaskSeries, today: Optional[date] = None) -> int:
    if today is None:
        today = date.today()

    try:
        start = getattr(s, "start", today)
        end = getattr(s, "end", today)
        if not isinstance(start, date) or not isinstance(end, date):
            return 0
    except Exception:
        return 0

    if today > end:
        return 0

    lo = today if today > start else start
    hi = end
    total = (hi - lo).days + 1
    if total <= 0:
        return 0

    done = 0
    try:
        dd = getattr(s, "done_days", set()) or set()
        if not isinstance(dd, set):
            dd = set()
        for d in dd:
            if isinstance(d, date) and lo <= d <= hi:
                done += 1
    except Exception:
        done = 0

    return max(0, int(total - done))


# =========================
# Dependencies (series-level)
# =========================
def would_create_cycle(
    series_list: List[TaskSeries],
    src_series_id: Optional[str] = None,
    dst_series_id: Optional[str] = None,
    *,
    src_id: Optional[str] = None,
    tgt_id: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    src = (src_series_id or src_id or "").strip()
    dst = (dst_series_id or tgt_id or "").strip()
    if not src or not dst:
        return False
    if src == dst:
        return True

    succ: Dict[str, List[str]] = {}
    for s in series_list or []:
        sid = s.series_id
        preds = list(getattr(s, "predecessors", []) or [])
        for p in preds:
            succ.setdefault(p, []).append(sid)

    succ.setdefault(src, []).append(dst)

    seen = set()
    q = [dst]
    while q:
        x = q.pop(0)
        if x == src:
            return True
        if x in seen:
            continue
        seen.add(x)
        for y in succ.get(x, []):
            q.append(y)
    return False


def can_depend_series(series_list: List[TaskSeries], src_series_id: str, dst_series_id: str) -> bool:
    if not src_series_id or not dst_series_id:
        return False
    if src_series_id == dst_series_id:
        return False
    return not would_create_cycle(series_list, src_series_id, dst_series_id)


# =========================
# Gantt
# =========================
def gantt_items(series_list: List[TaskSeries], today: date) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for s in series_list or []:
        try:
            items.append(
                {
                    "series_id": s.series_id,
                    "title": s.title,
                    "start": s.start,
                    "end": s.end,
                    "owner": s.owner,
                    "owner_id": s.owner_id,
                    "portfolio": s.portfolio,
                    "project": s.project,
                    "theme": s.theme,
                    "kind": s.kind,
                    "is_meta": bool(s.is_meta),
                    "state": s.state,
                    "progress": progress_percent(s, today),
                    "predecessors": list(getattr(s, "predecessors", []) or []),
                }
            )
        except Exception:
            continue
    return items


# =========================
# Persistence (v2 + backward compatible)
# =========================
def _normalize_series_dependencies(series_list: List["TaskSeries"]) -> None:
    for s in series_list or []:
        attr_preds: List[str] = []
        try:
            v = getattr(s, "predecessors", None)
            if isinstance(v, list):
                attr_preds = [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            attr_preds = []

        meta_preds: List[str] = []
        try:
            m = getattr(s, "meta", {}) or {}
            if isinstance(m, dict):
                v2 = m.get("predecessors", [])
                if isinstance(v2, list):
                    meta_preds = [str(x).strip() for x in v2 if str(x).strip()]
        except Exception:
            meta_preds = []

        merged: List[str] = []
        seen = set()
        for p in attr_preds + meta_preds:
            if p and p not in seen:
                seen.add(p)
                merged.append(p)

        try:
            setattr(s, "predecessors", list(merged))
        except Exception:
            pass

        try:
            m = getattr(s, "meta", {}) or {}
            if not isinstance(m, dict):
                m = {}
            m = dict(m)
            m["predecessors"] = list(merged)
            setattr(s, "meta", m)
        except Exception:
            pass


def save_state(
    path: Union[str, Path],
    series_list: List[TaskSeries],
    *,
    app_version: Optional[str] = None,
    **kwargs: Any,
) -> None:
    p = Path(path)
    _normalize_series_dependencies(series_list)
    payload = {
        "schema_version": int(STATE_SCHEMA_VERSION),
        "app_version": str(app_version or __version__),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "series": [s.to_dict() for s in (series_list or [])],
    }
    _atomic_write_json(p, payload)


def load_state(path: Union[str, Path]) -> List[TaskSeries]:
    # ✅ FIX: do NOT reference undefined series_list here
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))

        if isinstance(data, dict):
            series_data = data.get("series", [])
        else:
            series_data = data

        if not isinstance(series_data, list):
            return []

        out: List[TaskSeries] = []
        for x in series_data:
            if isinstance(x, dict):
                try:
                    out.append(TaskSeries.from_dict(x))
                except Exception:
                    continue
        _normalize_series_dependencies(out)
        return out
    except Exception:
        return []


# =========================
# Capacity (simple)
# =========================
def _business_days_between(start: date, end: date) -> int:
    if start > end:
        return 0
    d = start
    n = 0
    while d <= end:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def _week_window(today: date) -> tuple[date, date]:
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start, end


def _month_window(today: date) -> tuple[date, date]:
    start = today.replace(day=1)
    if start.month == 12:
        nstart = start.replace(year=start.year + 1, month=1, day=1)
    else:
        nstart = start.replace(month=start.month + 1, day=1)
    end = nstart - timedelta(days=1)
    return start, end


def capacity_summary(
    series_list: List[TaskSeries],
    employees: List[Dict[str, Any]],
    today: date,
    window: str = "week",
    default_capacity_per_day: float = 5.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    for k in ("cap_per_day", "capacity_per_day", "default_cap_per_day", "default_capacity"):
        if k in kwargs and kwargs[k] is not None:
            try:
                default_capacity_per_day = float(kwargs[k])
            except Exception:
                pass

    if window == "month":
        w_start, w_end = _month_window(today)
    else:
        w_start, w_end = _week_window(today)

    emp_by_id: Dict[str, Dict[str, Any]] = {}
    for e in employees or []:
        try:
            eid = str(e.get("id") or "").strip()
            if eid:
                emp_by_id[eid] = e
        except Exception:
            continue

    rows: List[Dict[str, Any]] = []
    for eid, e in emp_by_id.items():
        cap = default_capacity_per_day * float(_business_days_between(w_start, w_end))
        load = 0.0
        for s in series_list or []:
            if str(getattr(s, "owner_id", "") or "").strip() != eid:
                continue
            if not is_task(s):
                continue
            stt = getattr(s, "start", w_start)
            en = getattr(s, "end", w_end)
            if en < w_start or stt > w_end:
                continue
            load += float(getattr(s, "meta", {}).get("units", getattr(s, "meta", {}).get("weight", 1.0)) or 1.0)

        util = (load / cap) if cap > 0 else 0.0
        rows.append(
            {
                "employee_id": eid,
                "employee": str(e.get("display_name") or e.get("name") or ""),
                "capacity": cap,
                "load": load,
                "utilization": util,
                "window_start": w_start,
                "window_end": w_end,
            }
        )

    return {"rows": rows, "window_start": w_start, "window_end": w_end}


# =========================
# Additional helpers used by UI modules
# =========================
def compute_units_composition(series_list: List[TaskSeries], today: date) -> Dict[str, Any]:
    by_project: Dict[str, float] = {}
    for s in series_list or []:
        if not is_task(s):
            continue
        proj = str(getattr(s, "project", "") or "").strip() or "—"
        by_project[proj] = by_project.get(proj, 0.0) + float(getattr(s, "meta", {}).get("units", 1.0) or 1.0)
    return {"by_project": by_project}


def build_burndown_series(series_list: List[TaskSeries], today: date, horizon_days: int = 30) -> Dict[str, Any]:
    start = today
    end = today + timedelta(days=int(horizon_days))
    days = []
    remaining = []
    total = 0.0
    for s in series_list or []:
        if not is_task(s):
            continue
        total += float(getattr(s, "meta", {}).get("units", 1.0) or 1.0)

    rem = total
    for i in range((end - start).days + 1):
        d = start + timedelta(days=i)
        done_units = 0.0
        for s in series_list or []:
            if not is_task(s):
                continue
            if is_completed(s, d):
                done_units += float(getattr(s, "meta", {}).get("units", 1.0) or 1.0)
        rem = max(0.0, total - done_units)
        days.append(d)
        remaining.append(rem)

    return {"days": days, "remaining": remaining, "total": total}


def calc_velocity(series_list: List[TaskSeries], today: date, window_days: int = 7) -> float:
    start = today - timedelta(days=int(window_days))
    done = 0.0
    for s in series_list or []:
        if not is_task(s):
            continue
        dd = getattr(s, "done_days", set()) or set()
        if not isinstance(dd, set):
            continue
        for d in dd:
            if start <= d <= today:
                done += float(getattr(s, "meta", {}).get("units", 1.0) or 1.0) / max(1, total_days(s))
    return done


def forecast_eta_units(series_list: List[TaskSeries], today: date, velocity_per_day: float) -> Dict[str, Any]:
    total = 0.0
    done = 0.0
    for s in series_list or []:
        if not is_task(s):
            continue
        units = float(getattr(s, "meta", {}).get("units", 1.0) or 1.0)
        total += units
        if is_completed(s, today):
            done += units
    remaining = max(0.0, total - done)
    eta_days = (remaining / velocity_per_day) if velocity_per_day > 0 else None
    return {"total": total, "done": done, "remaining": remaining, "eta_days": eta_days}


def forecast_finish_date(series_list: List[TaskSeries], today: date, velocity_per_day: float) -> Optional[date]:
    eta = forecast_eta_units(series_list, today, velocity_per_day)
    if eta.get("eta_days") is None:
        return None
    try:
        return today + timedelta(days=int(round(float(eta["eta_days"]))))
    except Exception:
        return None
