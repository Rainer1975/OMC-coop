# core.py
from __future__ import annotations

import json

__version__ = "2026.03.03.1"
STATE_SCHEMA_VERSION = 2

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class TaskPart:
    day: date
    units: float = 1.0
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "units": float(self.units),
            "done": bool(self.done),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskPart":
        return TaskPart(
            day=datetime.fromisoformat(str(d.get("day"))).date(),
            units=float(d.get("units", 1.0)),
            done=bool(d.get("done", False)),
        )


@dataclass
class TaskSeries:
    series_id: str
    title: str
    start: date
    end: date
    kind: str = "task"  # "task" | "appointment" | "meta"
    owner: str = ""
    project: str = ""
    theme: str = ""
    portfolio: str = ""
    is_meta: bool = False

    parts: List[TaskPart] = field(default_factory=list)
    done_days: Set[str] = field(default_factory=set)

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "kind": self.kind,
            "owner": self.owner,
            "project": self.project,
            "theme": self.theme,
            "portfolio": self.portfolio,
            "is_meta": bool(self.is_meta),
            "parts": [p.to_dict() for p in (self.parts or [])],
            "done_days": sorted(list(self.done_days or set())),
            "meta": dict(self.meta or {}),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TaskSeries":
        s = TaskSeries(
            series_id=str(d.get("series_id", "")),
            title=str(d.get("title", "")),
            start=datetime.fromisoformat(str(d.get("start"))).date(),
            end=datetime.fromisoformat(str(d.get("end"))).date(),
            kind=str(d.get("kind", "task")),
            owner=str(d.get("owner", "")),
            project=str(d.get("project", "")),
            theme=str(d.get("theme", "")),
            portfolio=str(d.get("portfolio", "")),
            is_meta=bool(d.get("is_meta", False)),
            parts=[],
            done_days=set(d.get("done_days", []) or []),
            meta=dict(d.get("meta", {}) or {}),
        )
        parts = d.get("parts", []) or []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict):
                    try:
                        s.parts.append(TaskPart.from_dict(p))
                    except Exception:
                        continue
        return s


# =========================
# Core utils (unchanged)
# =========================
def total_days(s: TaskSeries) -> int:
    return max(1, (s.end - s.start).days + 1)


def is_task(s: TaskSeries) -> bool:
    return str(getattr(s, "kind", "task") or "task").lower() == "task"


def is_appointment(s: TaskSeries) -> bool:
    return str(getattr(s, "kind", "") or "").lower() == "appointment"


def is_completed(s: TaskSeries, today: date) -> bool:
    if not is_task(s):
        return False
    return progress_percent(s) >= 100.0


def progress_percent(s: TaskSeries) -> float:
    # If parts exist, compute completion by parts; else fallback to done_days coverage.
    if s.parts:
        total = sum(float(p.units) for p in s.parts)
        done = sum(float(p.units) for p in s.parts if p.done)
        return 100.0 * done / max(1e-9, total)
    # done_days fallback: if all days in range are done, 100%
    days = [(s.start + timedelta(days=i)).isoformat() for i in range(total_days(s))]
    done = sum(1 for d in days if d in (s.done_days or set()))
    return 100.0 * done / max(1, len(days))


def mark_done(s: TaskSeries, day: date) -> None:
    s.done_days.add(day.isoformat())


def unmark_done(s: TaskSeries, day: date) -> None:
    try:
        s.done_days.remove(day.isoformat())
    except Exception:
        pass


def bulk_set_done(series_list: List[TaskSeries], day: date, done: bool) -> None:
    for s in series_list or []:
        if not is_task(s):
            continue
        if done:
            mark_done(s, day)
        else:
            unmark_done(s, day)


def is_overdue(s: TaskSeries, today: date) -> bool:
    return is_task(s) and (s.end < today) and not is_completed(s, today)


def is_active(s: TaskSeries, today: date) -> bool:
    return s.start <= today <= s.end


def can_depend_series(series_list: List[TaskSeries], src_id: str, tgt_id: str) -> bool:
    # Simple check (no self), deeper cycle check elsewhere
    return bool(src_id) and bool(tgt_id) and src_id != tgt_id


def would_create_cycle(series_list: List[TaskSeries], src_id: str, tgt_id: str) -> bool:
    # Conservative: detect if tgt reaches src already
    by_id = {s.series_id: s for s in (series_list or []) if getattr(s, "series_id", "")}
    if src_id not in by_id or tgt_id not in by_id:
        return False

    # Build adjacency: succ -> preds list (as stored on succ)
    def preds_of(s: TaskSeries) -> List[str]:
        m = getattr(s, "meta", {}) or {}
        if isinstance(m, dict):
            v = m.get("predecessors", [])
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        v2 = getattr(s, "predecessors", None)
        if isinstance(v2, list):
            return [str(x).strip() for x in v2 if str(x).strip()]
        return []

    # If we add edge src -> tgt (meaning: tgt has predecessor src),
    # that creates a cycle if src is reachable from tgt via predecessor links.
    seen: Set[str] = set()
    stack = [src_id]
    # Traverse reverse edges: from node, go to its predecessors (because stored that way)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur == tgt_id:
            return True
        cur_s = by_id.get(cur)
        if not cur_s:
            continue
        for p in preds_of(cur_s):
            if p and p not in seen:
                stack.append(p)
    return False


def gantt_items(series_list: List[TaskSeries], today: Optional[date] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in series_list or []:
        out.append(
            {
                "id": s.series_id,
                "title": s.title,
                "start": s.start,
                "end": s.end,
                "kind": s.kind,
                "owner": s.owner,
                "project": s.project,
                "theme": s.theme,
                "portfolio": s.portfolio,
                "is_meta": bool(getattr(s, "is_meta", False)),
                "progress": progress_percent(s),
            }
        )
    return out


def critical_path_series(series_list: List[TaskSeries]) -> List[str]:
    # If you already have a real implementation elsewhere, keep it.
    # Here: conservative placeholder (no-op)
    return []


# =========================
# Persistence (v2, backward compatible loader)
# =========================
def save_state(path: Union[str, Path], series_list: List[TaskSeries], *, app_version: Optional[str] = None) -> None:
    """Persist state to disk.

    Backward compatible:
    - v1 format: JSON list of series dicts
    - v2+ format: JSON object with metadata + 'series' list
    """
    p = Path(path)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "app_version": app_version or __version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "series": [s.to_dict() for s in (series_list or [])],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(path: Union[str, Path]) -> List[TaskSeries]:
    """Load state from disk (supports both legacy list and new dict format)."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))

        # v2+ format
        if isinstance(data, dict):
            series_data = data.get("series", [])
        else:
            # legacy v1 format
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
        return out
    except Exception:
        return []
