# ui_gantt.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st

from core import TaskSeries, would_create_cycle

try:
    # exists in your core.py
    from core import critical_path_series  # type: ignore
except Exception:
    critical_path_series = None  # type: ignore


# =========================
# Helpers
# =========================
def _safe_date(x: Any) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    try:
        return datetime.fromisoformat(str(x)).date()
    except Exception:
        return None


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def _series_label(s: TaskSeries) -> str:
    title = (getattr(s, "title", "") or "").strip()
    project = (getattr(s, "project", "") or "").strip()
    owner = (getattr(s, "owner", "") or "").strip()
    bits = [b for b in [title, project, owner] if b]
    return " — ".join(bits) if bits else (getattr(s, "series_id", "") or "")


def _get_series_list(ctx: dict) -> List[TaskSeries]:
    # Prefer ctx.visible_series() (respects focus mode), else session_state.series
    vs = ctx.get("visible_series")
    if callable(vs):
        try:
            return list(vs())
        except Exception:
            pass
    return list(st.session_state.get("series", []) or [])


def _get_today(ctx: dict) -> date:
    t = ctx.get("today")
    if isinstance(t, date):
        return t
    return date.today()


def _get_preds(s: TaskSeries) -> List[str]:
    # Unified read of predecessor IDs (supports both s.predecessors and s.meta["predecessors"])
    preds: List[str] = []
    try:
        v = getattr(s, "predecessors", None)
        if isinstance(v, list):
            preds = [str(x) for x in v if str(x).strip()]
    except Exception:
        pass

    if not preds:
        try:
            m = getattr(s, "meta", {}) or {}
            if isinstance(m, dict):
                v = m.get("predecessors", [])
                if isinstance(v, list):
                    preds = [str(x) for x in v if str(x).strip()]
        except Exception:
            pass

    # de-dup, keep order
    out: List[str] = []
    seen = set()
    for p in preds:
        p = str(p).strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _set_preds(s: TaskSeries, pred_ids: List[str]) -> None:
    pred_ids = [str(x).strip() for x in (pred_ids or []) if str(x).strip()]
    # write to s.predecessors if it exists
    if hasattr(s, "predecessors"):
        try:
            setattr(s, "predecessors", list(pred_ids))
        except Exception:
            pass
    # always mirror into meta["predecessors"] for robustness / export
    try:
        m = getattr(s, "meta", {}) or {}
        if not isinstance(m, dict):
            m = {}
        m = dict(m)
        m["predecessors"] = list(pred_ids)
        setattr(s, "meta", m)
    except Exception:
        pass


def _is_task(s: TaskSeries, ctx: dict) -> bool:
    fn = ctx.get("is_task")
    if callable(fn):
        try:
            return bool(fn(s))
        except Exception:
            pass
    return str(getattr(s, "kind", "task") or "task").lower() == "task"


def _is_completed(s: TaskSeries, ctx: dict, today: date) -> bool:
    fn = ctx.get("is_completed")
    if callable(fn):
        try:
            return bool(fn(s, today))
        except Exception:
            pass
    # fallback: 100% progress if available
    fnp = ctx.get("progress_percent")
    if callable(fnp):
        try:
            return float(fnp(s) or 0.0) >= 100.0
        except Exception:
            pass
    return False


def _progress(s: TaskSeries, ctx: dict) -> float:
    fn = ctx.get("progress_percent")
    if callable(fn):
        try:
            return float(fn(s) or 0.0)
        except Exception:
            pass
    return 0.0


def _persist(ctx: dict) -> None:
    fn = ctx.get("persist")
    if callable(fn):
        try:
            fn()
            return
        except Exception:
            pass


# =========================
# Dependency editor
# =========================
def _render_dependency_editor(ctx: dict, series_filtered: List[TaskSeries]) -> None:
    if not series_filtered:
        st.info("Keine Tasks im aktuellen Filter.")
        return

    by_id = {str(getattr(s, "series_id", "") or "").strip(): s for s in series_filtered}
    by_id = {k: v for k, v in by_id.items() if k}

    succ = st.selectbox(
        "Nachfolger (Succ)",
        options=series_filtered,
        format_func=_series_label,
        key="gantt_dep_succ",
    )
    if not succ:
        return

    succ_id = str(getattr(succ, "series_id", "") or "").strip()
    if not succ_id:
        st.error("Succ hat keine series_id.")
        return

    current = _get_preds(succ)
    available_preds = [s for s in series_filtered if str(getattr(s, "series_id", "") or "").strip() and s is not succ]

    picked = st.multiselect(
        "Vorgänger (Pred) – mehrere möglich",
        options=available_preds,
        default=[by_id[p] for p in current if p in by_id],
        format_func=_series_label,
        key="gantt_dep_preds",
    )

    pred_ids = [str(getattr(p, "series_id", "") or "").strip() for p in picked]
    pred_ids = [p for p in pred_ids if p and p != succ_id]

    # validation: no cycles
    for pid in pred_ids:
        try:
            if would_create_cycle(series_filtered, src_id=pid, tgt_id=succ_id):
                st.error("Ungültig: Diese Abhängigkeit würde einen Zyklus erzeugen.")
                return
        except Exception:
            # if core changes, fail safe: allow save but warn
            st.warning("Zyklus-Check nicht verfügbar/fehlgeschlagen – bitte vorsichtig.")
            break

    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("💾 Speichern", key="gantt_dep_save", width="stretch"):
        _set_preds(succ, pred_ids)
        _persist(ctx)
        st.success("Dependencies gespeichert.")
        st.rerun()

    if c2.button("🧹 Leeren", key="gantt_dep_clear", width="stretch"):
        _set_preds(succ, [])
        _persist(ctx)
        st.success("Dependencies gelöscht.")
        st.rerun()

    c3.caption("Dependencies werden als predecessor-IDs am Nachfolger gespeichert.")


# =========================
# Gantt plotting
# =========================
def _compute_window_defaults(items: List[Dict[str, Any]], today: date) -> Tuple[date, date]:
    starts = [_safe_date(it.get("start")) for it in items]
    ends = [_safe_date(it.get("end")) for it in items]
    starts = [d for d in starts if d]
    ends = [d for d in ends if d]
    if not starts or not ends:
        return today, today + timedelta(days=14)
    return min(starts), max(ends)


def _plot_gantt(
    items: List[Dict[str, Any]],
    series_by_id: Dict[str, TaskSeries],
    ctx: dict,
    today: date,
    win_from: date,
    win_to: date,
    show_deps: bool,
    show_critical: bool,
) -> None:
    # sort for stable layout
    def sort_key(it: Dict[str, Any]) -> Tuple[str, str, str]:
        s = series_by_id.get(str(it.get("id") or "").strip())
        project = (getattr(s, "project", "") or "").strip() if s else (str(it.get("project") or "").strip())
        owner = (getattr(s, "owner", "") or "").strip() if s else (str(it.get("owner") or "").strip())
        title = (getattr(s, "title", "") or "").strip() if s else (str(it.get("title") or "").strip())
        return (_norm(project), _norm(owner), _norm(title))

    items_sorted = sorted(items, key=sort_key)

    # filter by window overlap
    filtered: List[Dict[str, Any]] = []
    for it in items_sorted:
        stt = _safe_date(it.get("start"))
        en = _safe_date(it.get("end"))
        if not stt or not en:
            continue
        if en < win_from or stt > win_to:
            continue
        filtered.append(it)

    if not filtered:
        st.info("Keine Tasks im gewählten Zeitraum.")
        return

    # critical path ids
    crit_ids = set()
    if show_critical and callable(critical_path_series):
        try:
            crit_ids = set(critical_path_series(list(series_by_id.values())))
        except Exception:
            crit_ids = set()

    # map y positions
    ymap: Dict[str, int] = {}
    labels: List[str] = []
    for i, it in enumerate(filtered):
        sid = str(it.get("id") or "").strip()
        ymap[sid] = i
        s = series_by_id.get(sid)
        labels.append(_series_label(s) if s else sid)

    fig_h = max(4.0, min(0.45 * len(filtered) + 2.5, 18.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # draw bars
    for it in filtered:
        sid = str(it.get("id") or "").strip()
        s = series_by_id.get(sid)
        stt = _safe_date(it.get("start"))
        en = _safe_date(it.get("end"))
        if not stt or not en:
            continue

        y = ymap[sid]
        left = mdates.date2num(stt)
        right = mdates.date2num(en + timedelta(days=1))  # inclusive end
        width = max(0.5, right - left)

        is_meta = bool(getattr(s, "is_meta", False)) if s else bool(it.get("is_meta", False))
        prog = _progress(s, ctx) if s else float(it.get("progress", 0.0) or 0.0)
        done = prog >= 100.0

        # style via alpha/hatch (no fixed colors)
        alpha = 0.35 if is_meta else (0.20 if done else 0.85)
        hatch = "///" if (sid in crit_ids and not done) else ("")

        ax.barh(
            y=y,
            width=width,
            left=left,
            height=0.6,
            alpha=alpha,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
        )

        # progress marker (thin bar inside)
        if prog > 0.0:
            inner_w = width * min(1.0, max(0.0, prog / 100.0))
            ax.barh(
                y=y,
                width=inner_w,
                left=left,
                height=0.18,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.3,
            )

    # today line
    ax.axvline(mdates.date2num(today), linestyle="--", linewidth=1.0)

    # deps arrows
    if show_deps:
        for it in filtered:
            sid = str(it.get("id") or "").strip()
            s = series_by_id.get(sid)
            if not s:
                continue
            succ_y = ymap.get(sid)
            if succ_y is None:
                continue

            succ_start = _safe_date(getattr(s, "start", None))
            if not succ_start:
                continue

            preds = _get_preds(s)
            for pid in preds:
                if pid not in ymap:
                    continue
                pred = series_by_id.get(pid)
                if not pred:
                    continue
                pred_end = _safe_date(getattr(pred, "end", None))
                if not pred_end:
                    continue

                # draw from pred_end to succ_start
                x0 = mdates.date2num(pred_end + timedelta(days=1))
                x1 = mdates.date2num(succ_start)
                y0 = ymap[pid]
                y1 = succ_y

                # only if in window-ish
                if max(x0, x1) < mdates.date2num(win_from) or min(x0, x1) > mdates.date2num(win_to + timedelta(days=1)):
                    continue

                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=0.8),
                )

    # axes formatting
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=9)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    ax.set_xlim(mdates.date2num(win_from), mdates.date2num(win_to + timedelta(days=1)))
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Tasks")

    fig.tight_layout()
    st.pyplot(fig)


# =========================
# Main render
# =========================
def render(ctx: dict) -> None:
    st.title("Gantt")

    open_detail = ctx.get("open_detail")

    today = _get_today(ctx)

    # load series
    series_all = _get_series_list(ctx)
    if not series_all:
        st.info("Noch keine Tasks vorhanden.")
        return

    # keep only tasks/appointments that have start/end and id
    cleaned: List[TaskSeries] = []
    for s in series_all:
        sid = str(getattr(s, "series_id", "") or "").strip()
        stt = _safe_date(getattr(s, "start", None))
        en = _safe_date(getattr(s, "end", None))
        if not sid or not stt or not en:
            continue
        cleaned.append(s)

    if not cleaned:
        st.info("Keine gültigen Tasks (fehlende series_id/start/end).")
        return

    # build items via core.gantt_items if available in ctx, else minimal fallback
    gantt_fn = ctx.get("gantt_items")
    items: List[Dict[str, Any]] = []
    if callable(gantt_fn):
        try:
            items = list(gantt_fn(cleaned, today=today) or [])
        except TypeError:
            items = list(gantt_fn(cleaned) or [])
        except Exception:
            items = []
    if not items:
        # fallback
        for s in cleaned:
            items.append(
                {
                    "id": str(getattr(s, "series_id", "") or "").strip(),
                    "title": str(getattr(s, "title", "") or ""),
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "project": str(getattr(s, "project", "") or ""),
                    "owner": str(getattr(s, "owner", "") or ""),
                    "theme": str(getattr(s, "theme", "") or ""),
                    "is_meta": bool(getattr(s, "is_meta", False)),
                    "kind": str(getattr(s, "kind", "task") or "task"),
                    "progress": _progress(s, ctx),
                }
            )

    series_by_id = {str(getattr(s, "series_id", "") or "").strip(): s for s in cleaned}
    series_by_id = {k: v for k, v in series_by_id.items() if k}

    # filters
    with st.expander("Filters & Zeitraum", expanded=True):
        portfolios = sorted({(getattr(s, "portfolio", "") or "").strip() for s in cleaned if (getattr(s, "portfolio", "") or "").strip()})
        projects = sorted({(getattr(s, "project", "") or "").strip() for s in cleaned if (getattr(s, "project", "") or "").strip()})
        themes = sorted({(getattr(s, "theme", "") or "").strip() for s in cleaned if (getattr(s, "theme", "") or "").strip()})
        owners = sorted({(getattr(s, "owner", "") or "").strip() for s in cleaned if (getattr(s, "owner", "") or "").strip()})

        c1, c2, c3 = st.columns(3)
        pf = c1.selectbox("Portfolio", options=["All"] + portfolios, index=0, key="gantt_f_pf")
        pr = c2.selectbox("Project", options=["All"] + projects, index=0, key="gantt_f_pr")
        th = c3.selectbox("Theme", options=["All"] + themes, index=0, key="gantt_f_th")

        c4, c5, c6 = st.columns(3)
        ow = c4.selectbox("Owner", options=["All"] + owners, index=0, key="gantt_f_ow")
        meta_mode = c5.selectbox("META", options=["All", "Normal", "META"], index=0, key="gantt_f_meta")
        hide_done = c6.checkbox("Hide completed", value=True, key="gantt_f_hide_done")

        show_deps = st.checkbox("Dependencies-Pfeile anzeigen", value=True, key="gantt_show_deps")
        show_crit = st.checkbox("Kritischen Pfad markieren (Hatch)", value=True, key="gantt_show_crit")

        # compute window defaults based on currently filtered set later; use broad defaults first
        win_def_from, win_def_to = _compute_window_defaults(items, today)
        rng = st.date_input(
            "Zeitraum (From / To)",
            value=(win_def_from, win_def_to),
            key="gantt_window",
        )
        if isinstance(rng, tuple) and len(rng) == 2:
            win_from, win_to = rng[0], rng[1]
        else:
            win_from, win_to = win_def_from, win_def_to

    def _match(s: TaskSeries) -> bool:
        if pf != "All" and (getattr(s, "portfolio", "") or "").strip() != pf:
            return False
        if pr != "All" and (getattr(s, "project", "") or "").strip() != pr:
            return False
        if th != "All" and (getattr(s, "theme", "") or "").strip() != th:
            return False
        if ow != "All" and (getattr(s, "owner", "") or "").strip() != ow:
            return False
        is_meta = bool(getattr(s, "is_meta", False))
        if meta_mode == "Normal" and is_meta:
            return False
        if meta_mode == "META" and not is_meta:
            return False
        if hide_done and _is_task(s, ctx) and _is_completed(s, ctx, today):
            return False
        return True

    series_filtered = [s for s in cleaned if _match(s)]
    if not series_filtered:
        st.info("Keine Tasks nach Filter.")
        return

    allowed = set(str(getattr(s, "series_id", "") or "").strip() for s in series_filtered)
    items_filtered = [it for it in items if str(it.get("id") or it.get("series_id") or "").strip() in allowed]

    # normalize item keys minimally (id/start/end)
    norm_items: List[Dict[str, Any]] = []
    for it in items_filtered:
        sid = str(it.get("id") or it.get("series_id") or "").strip()
        stt = _safe_date(it.get("start"))
        en = _safe_date(it.get("end"))
        if not sid or not stt or not en:
            continue
        norm_items.append(
            {
                "id": sid,
                "title": it.get("title") or it.get("series_title") or "",
                "start": stt,
                "end": en,
                "project": it.get("project") or "",
                "owner": it.get("owner") or "",
                "theme": it.get("theme") or "",
                "is_meta": bool(it.get("is_meta", False)),
                "kind": it.get("kind") or it.get("series_kind") or "task",
                "progress": float(it.get("progress", 0.0) or 0.0),
            }
        )

    if win_from > win_to:
        st.error("Zeitraum ungültig: From > To.")
        return

    # dependency editor
    with st.expander("➕ Dependencies bearbeiten", expanded=False):
        _render_dependency_editor(ctx, series_filtered)

    # plot
    st.markdown("---")
    st.caption("Tipp: Wenn du Pfeile nicht siehst, liegt’s fast immer daran, dass Pred oder Succ außerhalb des Zeitfensters gefiltert ist.")
    _plot_gantt(
        items=norm_items,
        series_by_id={str(getattr(s, "series_id", "") or "").strip(): s for s in series_filtered if str(getattr(s, "series_id", "") or "").strip()},
        ctx=ctx,
        today=today,
        win_from=win_from,
        win_to=win_to,
        show_deps=show_deps,
        show_critical=show_crit,
    )


    with st.expander('Open details (klick auf Titel)', expanded=False):
        if callable(open_detail):
            q = st.text_input('Filter', placeholder='search title…', key='gantt_open_filter')
            shown = list(series_filtered)
            if q:
                qq = q.lower()
                shown = [s for s in shown if qq in ((getattr(s,'title','') or '') + ' ' + (getattr(s,'project','') or '') + ' ' + (getattr(s,'owner','') or '')).lower()]
            for s in shown[:50]:
                sid = str(getattr(s,'series_id','') or '').strip()
                if not sid:
                    continue
                if st.button(str(getattr(s,'title','') or ''), key=f'gantt_open_{sid}', use_container_width=True):
                    open_detail(sid)
                st.caption(f"{getattr(s,'project','')} · {getattr(s,'owner','')} · {getattr(s,'start','')}→{getattr(s,'end','')}")
