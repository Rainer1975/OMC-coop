# core.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


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
        done_in_range = [
            d for d in dd if getattr(s, "start") <= d <= getattr(s, "end") and d <= today
        ]
        return max(0.0, min(1.0, float(len(done_in_range)) / float(td)))
    except Exception:
        return 0.0


def remaining_days(s: TaskSeries, today: Optional[date] = None) -> int:
    """
    Backward compatible helper (older UI modules import this).
    Remaining calendar days (inclusive) from max(today,start) until end,
    subtracting days already marked DONE in that window. Never negative.
    """
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
def would_create_cycle(series_list: List[TaskSeries], src_series_id: str, dst_series_id: str) -> bool:
    if not src_series_id or not dst_series_id:
        return False
    if src_series_id == dst_series_id:
        return True

    succ: Dict[str, List[str]] = {}
    for s in series_list or []:
        sid = s.series_id
        preds = list(getattr(s, "predecessors", []) or [])
        for p in preds:
            succ.setdefault(p, []).append(sid)

    succ.setdefault(src_series_id, []).append(dst_series_id)

    seen = set()
    q = [dst_series_id]
    while q:
        x = q.pop(0)
        if x == src_series_id:
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
# Persistence
# =========================
def save_state(path: Union[str, Path], series_list: List[TaskSeries]) -> None:
    p = Path(path)
    data = [s.to_dict() for s in (series_list or [])]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(path: Union[str, Path]) -> List[TaskSeries]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        out: List[TaskSeries] = []
        for x in data:
            if isinstance(x, dict):
                try:
                    out.append(TaskSeries.from_dict(x))
                except Exception:
                    continue
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
) -> Dict[str, Any]:
    w = (window or "week").lower().strip()
    if w == "month":
        win_start, win_end = _month_window(today)
    else:
        win_start, win_end = _week_window(today)

    emp_by_id = {}
    for e in employees or []:
        eid = str(e.get("id") or "").strip()
        if not eid:
            continue
        emp_by_id[eid] = {
            "id": eid,
            "display_name": str(e.get("display_name") or "").strip(),
        }

    def units_for_task(s: TaskSeries) -> float:
        try:
            wv = float(getattr(s, "weight", 0.0) or 0.0)
            if wv > 0:
                return wv
        except Exception:
            pass
        return 1.0

    by_emp: Dict[str, Dict[str, Any]] = {}
    for eid, e in emp_by_id.items():
        cap_days = _business_days_between(win_start, win_end)
        cap = float(cap_days) * float(default_capacity_per_day)
        by_emp[eid] = {
            "id": eid,
            "name": e.get("display_name") or eid,
            "window_start": win_start,
            "window_end": win_end,
            "capacity": cap,
            "planned": 0.0,
            "done": 0.0,
            "remaining": 0.0,
            "overdue": 0,
            "utilization": 0.0,
            "status": "green",
            "tasks": [],
        }

    for s in series_list or []:
        try:
            if not is_task(s):
                continue
            eid = str(getattr(s, "owner_id", "") or "").strip()
            if not eid or eid not in by_emp:
                continue
            if getattr(s, "end") < win_start or getattr(s, "start") > win_end:
                continue

            u = units_for_task(s)
            if not is_completed(s, today):
                by_emp[eid]["planned"] += u

            dd = getattr(s, "done_days", set()) or set()
            if isinstance(dd, set) and dd:
                done_in_win = 0
                for d in dd:
                    if not isinstance(d, date):
                        continue
                    if d < win_start or d > win_end:
                        continue
                    if d < getattr(s, "start") or d > getattr(s, "end"):
                        continue
                    if d.weekday() < 5:
                        done_in_win += 1
                if done_in_win > 0:
                    biz_days_series = _business_days_between(getattr(s, "start"), getattr(s, "end"))
                    if biz_days_series <= 0:
                        biz_days_series = 1
                    by_emp[eid]["done"] += float(u) * (float(done_in_win) / float(biz_days_series))

            if is_overdue(s, today):
                by_emp[eid]["overdue"] += 1

            by_emp[eid]["tasks"].append(s)
        except Exception:
            continue

    for eid, row in by_emp.items():
        planned = float(row["planned"])
        done = float(row["done"])
        remaining = max(0.0, planned - done)
        row["remaining"] = remaining
        cap = float(row["capacity"]) if float(row["capacity"]) > 0 else 1.0
        util = planned / cap
        row["utilization"] = util

        if util < 0.85:
            row["status"] = "green"
        elif util < 1.10:
            row["status"] = "yellow"
        else:
            row["status"] = "red"

    return {
        "window": w,
        "window_start": win_start,
        "window_end": win_end,
        "by_employee": by_emp,
    }


# =========================
# Velocity / Burndown helpers (units-based)
# =========================
def business_days_between_inclusive(start: date, end: date) -> List[date]:
    out: List[date] = []
    if start > end:
        return out
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def add_business_days(d: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    left = abs(int(n))
    cur = d
    while left > 0:
        cur = cur + timedelta(days=step)
        if cur.weekday() < 5:
            left -= 1
    return cur


def series_units(s: TaskSeries) -> float:
    try:
        wv = float(getattr(s, "weight", 0.0) or 0.0)
        if wv > 0:
            return float(wv)
    except Exception:
        pass
    return 1.0


def compute_units_composition(task_series: List[TaskSeries]):
    tasks = [s for s in (task_series or []) if is_task(s) and not getattr(s, "is_meta", False)]
    if not tasks:
        return [], [], [], []

    min_d = min(getattr(s, "start") for s in tasks)
    max_d = max(getattr(s, "end") for s in tasks)
    all_days = business_days_between_inclusive(min_d, max_d)
    day_index = {d: i for i, d in enumerate(all_days)}

    done_units_day = [0.0 for _ in all_days]
    total_units = 0.0
    comp_by_day: List[List[Dict[str, Any]]] = [[] for _ in all_days]

    per_task_total: Dict[str, float] = {}
    per_task_daily: Dict[str, float] = {}

    for s in tasks:
        u = series_units(s)
        total_units += u
        biz_days = business_days_between_inclusive(getattr(s, "start"), getattr(s, "end"))
        denom = max(1, len(biz_days))
        per_day = float(u) / float(denom)

        per_task_total[s.series_id] = float(u)
        per_task_daily[s.series_id] = per_day

        dd = getattr(s, "done_days", set()) or set()
        if not isinstance(dd, set):
            dd = set()

        for d in biz_days:
            i = day_index.get(d)
            if i is None:
                continue
            units_done = per_day if d in dd else 0.0
            done_units_day[i] += units_done

    remaining_units_day: List[float] = []
    cum_done = 0.0
    for i, _d in enumerate(all_days):
        cum_done += float(done_units_day[i])
        remaining = max(0.0, float(total_units) - float(cum_done))
        remaining_units_day.append(remaining)

    for s in tasks:
        u_total = float(per_task_total.get(s.series_id, 0.0))
        per_day = float(per_task_daily.get(s.series_id, 0.0))
        biz_days = business_days_between_inclusive(getattr(s, "start"), getattr(s, "end"))
        dd = getattr(s, "done_days", set()) or set()
        if not isinstance(dd, set):
            dd = set()

        done_so_far = 0.0
        for d in biz_days:
            i = day_index.get(d)
            if i is None:
                continue
            units_done = per_day if d in dd else 0.0
            done_so_far += units_done
            units_remaining = max(0.0, u_total - done_so_far)
            comp_by_day[i].append(
                {
                    "series_id": s.series_id,
                    "title": s.title,
                    "units_done": float(units_done),
                    "units_remaining": float(units_remaining),
                }
            )

    return all_days, done_units_day, remaining_units_day, comp_by_day


def avg_last_window_float(values: List[float], anchor_idx: int, window_business_days: int) -> float:
    w = max(1, int(window_business_days))
    start = max(0, int(anchor_idx) - w + 1)
    win = [float(x) for x in values[start : anchor_idx + 1]]
    if not win:
        return 0.0
    return float(sum(win)) / float(len(win))


def forecast_eta_units(
    all_days: List[date],
    done_units_day: List[float],
    remaining_units_day: List[float],
    today_: date,
    window_business_days: int = 10,
):
    if not all_days:
        return None, 0.0, None, 0

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


# =========================
# Compatibility layer for app.py imports
# =========================
def build_burndown_series(
    series_list: List[TaskSeries],
    today: Optional[date] = None,
    window_business_days: int = 10,
) -> Dict[str, Any]:
    if today is None:
        today = date.today()

    all_days, done_units_day, remaining_units_day, comp_by_day = compute_units_composition(series_list)

    anchor_day, velocity, eta, dq = forecast_eta_units(
        all_days=all_days,
        done_units_day=done_units_day,
        remaining_units_day=remaining_units_day,
        today_=today,
        window_business_days=int(window_business_days),
    )

    return {
        "all_days": all_days,
        "done_units_day": done_units_day,
        "remaining_units_day": remaining_units_day,
        "comp_by_day": comp_by_day,
        "anchor_day": anchor_day,
        "velocity": float(velocity or 0.0),
        "eta": eta,
        "data_quality_days": int(dq or 0),
    }


def calc_velocity(series_list: List[TaskSeries], today: Optional[date] = None, window_business_days: int = 10) -> float:
    if today is None:
        today = date.today()
    pkg = build_burndown_series(series_list, today=today, window_business_days=window_business_days)
    return float(pkg.get("velocity", 0.0) or 0.0)


def forecast_finish_date(series_list: List[TaskSeries], today: Optional[date] = None, window_business_days: int = 10) -> Optional[date]:
    if today is None:
        today = date.today()
    pkg = build_burndown_series(series_list, today=today, window_business_days=window_business_days)
    return pkg.get("eta")
