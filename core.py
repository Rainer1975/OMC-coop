# core.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union


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

    kind: str = "task"          # "task" | "appointment"
    is_meta: bool = False

    portfolio: str = "Default"
    project: str = ""
    theme: str = "General"

    owner: str = ""
    owner_id: str = ""

    state: str = "ACTIVE"       # PLANNED | ACTIVE | BLOCKED | DONE | CANCELLED

    time_start: str = ""        # appointment optional
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
        # Hard guard: must be dict
        if not isinstance(d, dict):
            raise TypeError("TaskSeries.from_dict expects dict")

        st = _parse_date(d.get("start")) or date.today()
        en = _parse_date(d.get("end")) or st

        parts_raw = d.get("parts") or []
        parts: List[TaskPart] = []
        if isinstance(parts_raw, list):
            for pr in parts_raw:
                if isinstance(pr, dict):
                    try:
                        parts.append(TaskPart.from_dict(pr))
                    except Exception:
                        continue

        dd_raw = d.get("done_days") or []
        done_days: Set[date] = set()
        if isinstance(dd_raw, list):
            for x in dd_raw:
                dx = _parse_date(x)
                if dx:
                    done_days.add(dx)

        preds = d.get("predecessors") or d.get("depends_on") or []
        if not isinstance(preds, list):
            preds = []
        preds = [str(x).strip() for x in preds if str(x).strip()]

        return TaskSeries(
            series_id=str(d.get("series_id") or ""),
            title=str(d.get("title") or ""),
            start=st,
            end=en,
            kind=str(d.get("kind") or "task"),
            is_meta=bool(d.get("is_meta", False)),
            portfolio=str(d.get("portfolio") or "Default"),
            project=str(d.get("project") or ""),
            theme=str(d.get("theme") or "General"),
            owner=str(d.get("owner") or ""),
            owner_id=str(d.get("owner_id") or ""),
            state=str(d.get("state") or "ACTIVE"),
            time_start=str(d.get("time_start") or ""),
            time_end=str(d.get("time_end") or ""),
            location=str(d.get("location") or ""),
            parts=parts,
            predecessors=preds,
            done_days=done_days,
            meta=dict(d.get("meta") or {}),
        )


# =========================
# Constructors
# =========================
def new_part(
    part_id: str,
    label: str,
    start: date,
    end: date,
    weight: float = 100.0,
    predecessors: Optional[List[str]] = None,
) -> TaskPart:
    return TaskPart(
        part_id=str(part_id),
        label=str(label),
        start=start,
        end=end,
        weight=float(weight),
        predecessors=list(predecessors or []),
    )


def new_series(
    title: str,
    project: str,
    theme: str,
    owner: str,
    owner_id: str,
    start: date,
    end: date,
    is_meta: bool = False,
    kind: str = "task",
    portfolio: str = "Default",
    state: str = "ACTIVE",
) -> TaskSeries:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    pid = os.getpid()
    sid = f"s_{ts}_{pid}_{abs(hash((title, project, owner, start.isoformat(), end.isoformat())))%10**8}"
    return TaskSeries(
        series_id=sid,
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
    )


# =========================
# Classification helpers
# =========================
def is_task(s: TaskSeries) -> bool:
    return (getattr(s, "kind", "") or "task").lower() != "appointment"


def is_appointment(s: TaskSeries) -> bool:
    return (getattr(s, "kind", "") or "").lower() == "appointment"


def total_days(s: TaskSeries) -> int:
    if not getattr(s, "start", None) or not getattr(s, "end", None):
        return 0
    return max(0, (s.end - s.start).days + 1)



def remaining_days(s: TaskSeries, today: date) -> int:
    """Remaining planned days from *today* to end (inclusive), excluding done days."""
    if not isinstance(today, date):
        return 0
    if not getattr(s, "start", None) or not getattr(s, "end", None):
        return 0
    if today > s.end:
        return 0

    start = s.start if today <= s.start else today
    if start > s.end:
        return 0

    done = getattr(s, "done_days", set()) or set()
    rem = 0
    d = start
    while d <= s.end:
        if d not in done:
            rem += 1
        d += timedelta(days=1)
    return rem

def is_completed(s: TaskSeries, today: date) -> bool:
    if getattr(s, "state", "") == "DONE":
        return True
    if not is_task(s):
        return False
    t = total_days(s)
    if t <= 0:
        return s.start in (s.done_days or set())
    needed = {s.start + timedelta(days=i) for i in range(t)}
    return needed.issubset(set(s.done_days or set()))


def is_active(s: TaskSeries, today: date) -> bool:
    if is_appointment(s):
        return s.start <= today <= s.end
    if getattr(s, "state", "") == "CANCELLED":
        return False
    return s.start <= today <= s.end and not is_completed(s, today)


def is_overdue(s: TaskSeries, today: date) -> bool:
    if not is_task(s):
        return False
    if getattr(s, "state", "") == "CANCELLED":
        return False
    if is_completed(s, today):
        return False
    return s.end < today


def progress_percent(s: TaskSeries) -> float:
    if not is_task(s):
        return 0.0
    t = total_days(s)
    if t <= 0:
        return 100.0 if s.start in (s.done_days or set()) else 0.0
    done = sum(1 for d in (s.done_days or set()) if s.start <= d <= s.end)
    return max(0.0, min(100.0, 100.0 * (done / float(t))))


# =========================
# Done operations
# =========================
def mark_done(s: TaskSeries, d: date, actor: Optional[str] = None) -> None:
    if not is_task(s):
        return
    if d < s.start or d > s.end:
        return
    if s.done_days is None:
        s.done_days = set()
    s.done_days.add(d)


def unmark_done(s: TaskSeries, d: date, actor: Optional[str] = None) -> None:
    if not is_task(s):
        return
    if not s.done_days:
        return
    s.done_days.discard(d)
    if getattr(s, "state", "") == "DONE":
        s.state = "ACTIVE"


def bulk_set_done(s: TaskSeries, days: List[date], set_done: bool = True, actor: Optional[str] = None) -> int:
    if not is_task(s):
        return 0
    if s.done_days is None:
        s.done_days = set()
    changed = 0
    for d in days or []:
        if d < s.start or d > s.end:
            continue
        if set_done:
            if d not in s.done_days:
                s.done_days.add(d)
                changed += 1
        else:
            if d in s.done_days:
                s.done_days.remove(d)
                changed += 1
    if set_done and is_completed(s, date.today()):
        s.state = "DONE"
    if (not set_done) and getattr(s, "state", "") == "DONE":
        s.state = "ACTIVE"
    return changed


# =========================
# Dependencies (robust)
# =========================
SeriesOrId = Union[str, TaskSeries, None]


def _sid(x: SeriesOrId) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(getattr(x, "series_id", "") or "").strip()


def can_depend_series(a: SeriesOrId, b: SeriesOrId) -> bool:
    sa = _sid(a)
    sb = _sid(b)
    if not sa or not sb:
        return False
    return sa != sb


def _pred_map(series_list: List[TaskSeries]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for s in series_list or []:
        sid = (s.series_id or "").strip()
        if not sid:
            continue
        preds = getattr(s, "predecessors", []) or []
        if not isinstance(preds, list):
            preds = []
        m[sid] = [str(x).strip() for x in preds if str(x).strip()]
    return m


def would_create_cycle(pred_id: str, succ_id: str, series_list: List[TaskSeries]) -> bool:
    pred_id = (pred_id or "").strip()
    succ_id = (succ_id or "").strip()
    if not pred_id or not succ_id or pred_id == succ_id:
        return True

    preds = _pred_map(series_list)

    succ_preds = set(preds.get(succ_id, []))
    succ_preds.add(pred_id)
    preds[succ_id] = list(succ_preds)

    stack = [succ_id]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for p in preds.get(cur, []):
            if p == succ_id:
                return True
            stack.append(p)
    return False


# =========================
# Gantt items
# =========================
def gantt_items(series_list: List[TaskSeries], today: Optional[date] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in series_list or []:
        sid = (s.series_id or "").strip()
        if not sid:
            continue
        out.append(
            {
                "id": sid,
                "title": s.title,
                "start": s.start,
                "end": s.end,
                "portfolio": getattr(s, "portfolio", ""),
                "project": getattr(s, "project", ""),
                "theme": getattr(s, "theme", ""),
                "owner": getattr(s, "owner", ""),
                "owner_id": getattr(s, "owner_id", ""),
                "is_meta": bool(getattr(s, "is_meta", False)),
                "kind": getattr(s, "kind", "task"),
                "state": getattr(s, "state", ""),
                "predecessors": list(getattr(s, "predecessors", []) or []),
                "progress": progress_percent(s),
                "done_days": set(getattr(s, "done_days", set()) or set()),
            }
        )
    return out


# =========================
# Critical path
# =========================
def critical_path_series(series_list: List[TaskSeries]) -> List[str]:
    preds = _pred_map(series_list)
    dur: Dict[str, int] = {}
    for s in series_list or []:
        sid = (s.series_id or "").strip()
        if not sid:
            continue
        dur[sid] = max(1, total_days(s))

    if not dur:
        return []

    best_len: Dict[str, int] = {}
    best_prev: Dict[str, Optional[str]] = {}
    visiting: Set[str] = set()

    def dp(node: str) -> int:
        if node in best_len:
            return best_len[node]
        if node in visiting:
            best_len[node] = dur.get(node, 1)
            best_prev[node] = None
            return best_len[node]
        visiting.add(node)

        pbest = 0
        pbest_id: Optional[str] = None
        for p in preds.get(node, []):
            if not p:
                continue
            v = dp(p)
            if v > pbest:
                pbest = v
                pbest_id = p

        visiting.remove(node)
        best_len[node] = dur.get(node, 1) + pbest
        best_prev[node] = pbest_id
        return best_len[node]

    end_node = max(dur.keys(), key=lambda n: dp(n))
    path: List[str] = []
    cur: Optional[str] = end_node
    while cur:
        path.append(cur)
        cur = best_prev.get(cur)
    path.reverse()
    return path


# =========================
# Persistence
# =========================
def _atomic_write(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="omg_", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def save_state(path: str, series_list: List[TaskSeries]) -> None:
    payload: List[Dict[str, Any]] = []
    for s in series_list or []:
        if isinstance(s, TaskSeries):
            payload.append(s.to_dict())
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_state(path: str) -> List[TaskSeries]:
    p = Path(path)
    if not p.exists():
        return []

    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        data = raw.get("series") or raw.get("data") or raw.get("items") or []
        if not isinstance(data, list):
            data = []
    else:
        data = []

    out: List[TaskSeries] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            s = TaskSeries.from_dict(d)
            if not (s.series_id or "").strip():
                continue
            out.append(s)
        except Exception:
            continue
    return out


# =========================
# Capacity / Workload
# =========================
def business_days_between(start: date, end: date) -> int:
    """Count Mon–Fri days in [start,end]."""
    if not isinstance(start, date) or not isinstance(end, date):
        return 0
    if end < start:
        return 0
    n = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def week_window(today: date) -> tuple[date, date]:
    """Return Monday..Sunday window for given day."""
    if not isinstance(today, date):
        today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start, end


def _overlap_days(start_a: date, end_a: date, start_b: date, end_b: date) -> tuple[date, date, bool]:
    """Return (start,end,has_overlap) for overlap of [a] and [b]."""
    st = max(start_a, start_b)
    en = min(end_a, end_b)
    return st, en, (st <= en)


def capacity_summary(
    series_list: List[TaskSeries],
    employees: List[Dict[str, Any]],
    today: date,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Compute Planned/Done/Remaining per owner for a date window.

    Planned/Done/Remaining are counted in *business-day units*.
    - Planned: business days in overlap of task and window
    - Done: done_days inside window (business days)
    - Remaining: max(Planned - Done, 0)

    Capacity is business days in window (Mon–Fri). Utilization = Planned/Capacity.
    """

    if window_start is None or window_end is None:
        ws, we = week_window(today)
        window_start = window_start or ws
        window_end = window_end or we

    if window_end < window_start:
        window_start, window_end = window_end, window_start

    # Map owner_id -> display
    emp_map: Dict[str, str] = {}
    for e in employees or []:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id", "") or "").strip()
        dn = str(e.get("display_name", "") or "").strip()
        if eid and dn:
            emp_map[eid] = dn

    cap = float(business_days_between(window_start, window_end) or 0)
    cap = cap if cap > 0 else 1.0

    agg: Dict[str, Dict[str, Any]] = {}

    def _bucket(owner_id: str, owner_name: str) -> Dict[str, Any]:
        if owner_id not in agg:
            agg[owner_id] = {
                "owner_id": owner_id,
                "owner": owner_name,
                "planned": 0.0,
                "done": 0.0,
                "remaining": 0.0,
                "capacity": cap,
                "util": 0.0,
                "tasks": 0,
                "overdue": 0,
            }
        return agg[owner_id]

    for s in series_list or []:
        # Capacity is about tasks only; appointments/meta/cancelled don't contribute.
        if not is_task(s):
            continue
        if getattr(s, "state", "") in ("CANCELLED",):
            continue

        st = getattr(s, "start", None)
        en = getattr(s, "end", None)
        if not isinstance(st, date) or not isinstance(en, date):
            continue

        ov_st, ov_en, ok = _overlap_days(st, en, window_start, window_end)
        if not ok:
            continue

        owner_id = str(getattr(s, "owner_id", "") or "").strip()
        owner_name = str(getattr(s, "owner", "") or "").strip()
        if not owner_name and owner_id:
            owner_name = emp_map.get(owner_id, "")
        if not owner_id and owner_name:
            # keep stable bucket for names if IDs missing
            owner_id = f"name:{owner_name.lower()}"

        if not owner_id:
            owner_id = "(unassigned)"
            owner_name = "Unassigned"

        b = _bucket(owner_id, owner_name or emp_map.get(owner_id, owner_id))
        b["tasks"] += 1
        if is_overdue(s, today):
            b["overdue"] += 1

        planned_bd = float(business_days_between(ov_st, ov_en))

        done_bd = 0.0
        for d in (getattr(s, "done_days", set()) or set()):
            if not isinstance(d, date):
                continue
            if window_start <= d <= window_end and d.weekday() < 5:
                done_bd += 1.0

        # clamp
        done_bd = min(done_bd, planned_bd)
        rem_bd = max(planned_bd - done_bd, 0.0)

        b["planned"] += planned_bd
        b["done"] += done_bd
        b["remaining"] += rem_bd

    out: List[Dict[str, Any]] = list(agg.values())
    for r in out:
        r["util"] = (float(r.get("planned", 0.0)) / float(r.get("capacity", cap) or cap)) if cap else 0.0

    # stable ordering: high util first, then name
    out.sort(key=lambda x: (-float(x.get("util", 0.0)), str(x.get("owner", ""))))
    return out


# =========================
# Velocity / Burndown (units)
# =========================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def business_day_range(start: date, end: date) -> List[date]:
    """Return Mon–Fri days in [start,end]."""
    if not isinstance(start, date) or not isinstance(end, date):
        return []
    if end < start:
        return []
    out: List[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def add_business_days(start: date, n: int) -> date:
    """Add n business days (Mon–Fri) to start. n can be 0."""
    if not isinstance(start, date):
        start = date.today()
    n = int(n)
    if n <= 0:
        return start
    d = start
    left = n
    while left > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            left -= 1
    return d


def series_total_units(s: TaskSeries) -> float:
    """Total units for a series.

    Rule (per your decision):
    - If explicit weight exists, use it.
    - Else fallback to 1.
    """
    # 1) explicit weight in meta
    meta = getattr(s, "meta", {}) or {}
    w = _safe_float(meta.get("weight", None), default=0.0)
    if w > 0:
        return w

    # 2) parts weight (if parts are used)
    parts = getattr(s, "parts", []) or []
    if parts:
        sw = 0.0
        for p in parts:
            sw += _safe_float(getattr(p, "weight", 0.0), default=0.0)
        if sw > 0:
            return sw

    # 3) fallback
    return 1.0


def series_units_per_business_day(s: TaskSeries) -> float:
    """Distribute total units across business days in the series duration."""
    st = getattr(s, "start", None)
    en = getattr(s, "end", None)
    if not isinstance(st, date) or not isinstance(en, date):
        return 0.0
    bdays = business_days_between(st, en)
    if bdays <= 0:
        return 0.0
    return float(series_total_units(s)) / float(bdays)


def compute_units_composition(task_series: List[TaskSeries]):
    """Compute done units/day + remaining units/day over business days.

    Done units/day are derived from done_days:
    each DONE day contributes (total_units / business_days_in_task).

    Returns:
      all_days (business days), done_units_day, remaining_units_day, comp_by_day
    """
    tasks = [s for s in (task_series or []) if is_task(s)]
    if not tasks:
        return [], [], [], []

    min_day = min(s.start for s in tasks)
    max_day = max(s.end for s in tasks)

    all_days = business_day_range(min_day, max_day)
    if not all_days:
        return [], [], [], []

    done_units_day: List[float] = []
    remaining_units_day: List[float] = []
    comp_by_day: List[List[Dict[str, Any]]] = []

    # precompute per-series constants
    per_day: Dict[str, float] = {}
    total_units: Dict[str, float] = {}
    for s in tasks:
        per_day[s.series_id] = series_units_per_business_day(s)
        total_units[s.series_id] = float(series_total_units(s))

    for day in all_days:
        done_today = 0.0
        comp: List[Dict[str, Any]] = []
        rem_total = 0.0

        for s in tasks:
            st = s.start
            en = s.end
            if day < st:
                rem_total += total_units[s.series_id]
                continue
            if day > en:
                continue

            # done units up to day
            dd = getattr(s, "done_days", set()) or set()
            done_upto = 0.0
            for x in dd:
                if not isinstance(x, date):
                    continue
                if st <= x <= day and x.weekday() < 5:
                    done_upto += per_day[s.series_id]

            # clamp to total
            done_upto = min(done_upto, total_units[s.series_id])
            rem_total += max(0.0, total_units[s.series_id] - done_upto)

            if day in dd and day.weekday() < 5:
                done_today += per_day[s.series_id]
                comp.append(
                    {
                        "series_id": s.series_id,
                        "owner": getattr(s, "owner", ""),
                        "project": getattr(s, "project", ""),
                        "task": getattr(s, "title", ""),
                        "units": per_day[s.series_id],
                    }
                )

        done_units_day.append(float(done_today))
        remaining_units_day.append(float(rem_total))
        comp_by_day.append(comp)

    return all_days, done_units_day, remaining_units_day, comp_by_day


def avg_last_window_float(values: List[float], end_idx: int, window: int) -> float:
    if not values:
        return 0.0
    end_idx = max(0, min(end_idx, len(values) - 1))
    window = max(1, int(window))
    start = max(0, end_idx - window + 1)
    chunk = values[start : end_idx + 1]
    return float(sum(chunk)) / float(len(chunk)) if chunk else 0.0


def forecast_eta_units(
    all_days: List[date],
    done_units_day: List[float],
    remaining_units_day: List[float],
    today_: date,
    window_business_days: int = 10,
):
    """Forecast finish date based on remaining units and average done units/day.

    - Anchor = last business day <= today with any done, else today
    - Velocity = avg done units/day over last N business days ending at anchor
    - ETA = anchor + ceil(remaining/velocity) business days
    - Data quality = days with done>0 within window
    """
    if not all_days:
        return None, 0.0, None, 0

    # locate anchor index
    last_le_today = 0
    last_done = None
    for i, dd in enumerate(all_days):
        if dd <= today_:
            last_le_today = i
            if i < len(done_units_day) and float(done_units_day[i]) > 0.0:
                last_done = i
        else:
            break

    anchor_idx = last_done if last_done is not None else last_le_today
    anchor_day = all_days[anchor_idx]
    rem_anchor = float(remaining_units_day[anchor_idx]) if anchor_idx < len(remaining_units_day) else 0.0
    vel = avg_last_window_float(done_units_day, anchor_idx, int(window_business_days))

    # data quality
    w = max(1, int(window_business_days))
    start = max(0, anchor_idx - w + 1)
    dq = sum(1 for v in done_units_day[start : anchor_idx + 1] if float(v) > 0.0)

    if rem_anchor <= 0:
        return anchor_day, 0.0, anchor_day, dq
    if vel <= 0:
        return anchor_day, 0.0, None, dq

    import math

    days_needed = int(math.ceil(rem_anchor / vel))
    eta = add_business_days(anchor_day, days_needed)
    return anchor_day, float(vel), eta, dq
