# ui_gantt.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st

from core import TaskSeries, would_create_cycle

__version__ = "2026.03.03.2"


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


# ---------- flash messages survive st.rerun ----------
def _flash_set(level: str, msg: str) -> None:
    st.session_state["_gantt_flash"] = (str(level), str(msg))


def _flash_render() -> None:
    v = st.session_state.pop("_gantt_flash", None)
    if not v:
        return
    level, msg = v
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.error(msg)


def _get_today(ctx: dict) -> date:
    t = ctx.get("today")
    return t if isinstance(t, date) else date.today()


def _get_series_list(ctx: dict) -> List[TaskSeries]:
    vs = ctx.get("visible_series")
    if callable(vs):
        try:
            return list(vs())
        except Exception:
            pass
    return list(st.session_state.get("series", []) or [])


def _persist(ctx: dict) -> Tuple[bool, str]:
    fn = ctx.get("persist")
    if not callable(fn):
        return False, "persist() fehlt im ctx."
    try:
        fn()
        return True, ""
    except Exception as e:
        return False, f"Persist fehlgeschlagen: {e!r}"


# ---------- predecessors: read robustly from BOTH fields ----------
def _get_preds_from_series(s: TaskSeries) -> List[str]:
    preds: List[str] = []
    try:
        v = getattr(s, "predecessors", None)
        if isinstance(v, list):
            preds = [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        preds = []

    try:
        m = getattr(s, "meta", {}) or {}
        if isinstance(m, dict):
            v2 = m.get("predecessors", [])
            if isinstance(v2, list):
                for x in v2:
                    xs = str(x).strip()
                    if xs and xs not in preds:
                        preds.append(xs)
    except Exception:
        pass

    # de-dup keep order
    out: List[str] = []
    seen = set()
    for p in preds:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _set_preds_to_series(s: TaskSeries, pred_ids: List[str]) -> None:
    pred_ids = [str(x).strip() for x in (pred_ids or []) if str(x).strip()]

    # write BOTH: attribute + meta
    try:
        if hasattr(s, "predecessors"):
            setattr(s, "predecessors", list(pred_ids))
    except Exception:
        pass

    try:
        m = getattr(s, "meta", {}) or {}
        if not isinstance(m, dict):
            m = {}
        m = dict(m)
        m["predecessors"] = list(pred_ids)
        setattr(s, "meta", m)
    except Exception:
        pass


def _compute_window_defaults(items: List[Dict[str, Any]], today: date) -> Tuple[date, date]:
    starts = [_safe_date(it.get("start")) for it in items]
    ends = [_safe_date(it.get("end")) for it in items]
    starts = [d for d in starts if d]
    ends = [d for d in ends if d]
    if not starts or not ends:
        return today, today + timedelta(days=14)
    return min(starts), max(ends)


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

    current = _get_preds_from_series(succ)
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

    # cycle check
    for pid in pred_ids:
        try:
            if would_create_cycle(series_filtered, src_id=pid, tgt_id=succ_id):
                st.error("Ungültig: Diese Abhängigkeit würde einen Zyklus erzeugen.")
                return
        except Exception:
            st.warning("Zyklus-Check nicht verfügbar/fehlgeschlagen – bitte vorsichtig.")
            break

    c1, c2, c3 = st.columns([1, 1, 6])

    if c1.button("💾 Speichern", key="gantt_dep_save", use_container_width=True):
        _set_preds_to_series(succ, pred_ids)
        ok, err = _persist(ctx)
        _flash_set("success", "Dependencies gespeichert.") if ok else _flash_set("error", err or "Save fehlgeschlagen.")
        st.rerun()

    if c2.button("🧹 Leeren", key="gantt_dep_clear", use_container_width=True):
        _set_preds_to_series(succ, [])
        ok, err = _persist(ctx)
        _flash_set("success", "Dependencies gelöscht.") if ok else _flash_set("error", err or "Clear fehlgeschlagen.")
        st.rerun()

    c3.caption("Dependencies werden als predecessor-IDs am Nachfolger gespeichert.")


def _plot_gantt(
    items: List[Dict[str, Any]],
    series_by_id: Dict[str, TaskSeries],
    today: date,
    win_from: date,
    win_to: date,
    show_deps: bool,
) -> None:
    # Sort stable
    def sort_key(it: Dict[str, Any]) -> Tuple[str, str, str]:
        sid = str(it.get("id") or it.get("series_id") or "").strip()
        s = series_by_id.get(sid)
        project = (getattr(s, "project", "") or "").strip() if s else str(it.get("project") or "").strip()
        owner = (getattr(s, "owner", "") or "").strip() if s else str(it.get("owner") or "").strip()
        title = (getattr(s, "title", "") or "").strip() if s else str(it.get("title") or "").strip()
        return (_norm(project), _norm(owner), _norm(title))

    # normalize items ids
    norm_items: List[Dict[str, Any]] = []
    for it in items or []:
        sid = str(it.get("id") or it.get("series_id") or "").strip()
        stt = _safe_date(it.get("start"))
        en = _safe_date(it.get("end"))
        if not sid or not stt or not en:
            continue
        norm_items.append(
            {
                "id": sid,
                "title": it.get("title") or "",
                "project": it.get("project") or "",
                "owner": it.get("owner") or "",
                "start": stt,
                "end": en,
                # optional: predecessors may exist on items
                "predecessors": it.get("predecessors", []),
            }
        )

    items_sorted = sorted(norm_items, key=sort_key)

    # filter by time overlap
    filtered: List[Dict[str, Any]] = []
    for it in items_sorted:
        stt = it["start"]
        en = it["end"]
        if en < win_from or stt > win_to:
            continue
        filtered.append(it)

    plot_from = win_from
    plot_to = win_to

    # include predecessor items when deps enabled
    if show_deps and filtered:
        item_by_id = {it["id"]: it for it in items_sorted}
        include = {it["id"] for it in filtered}

        for it in list(filtered):
            sid = it["id"]
            s = series_by_id.get(sid)
            preds = []

            # read preds from series robustly
            if s:
                preds = _get_preds_from_series(s)

            # also read preds from items if present
            v = it.get("predecessors", [])
            if isinstance(v, list):
                for x in v:
                    xs = str(x).strip()
                    if xs and xs not in preds:
                        preds.append(xs)

            for pid in preds:
                if pid in item_by_id:
                    include.add(pid)

        filtered = [it for it in items_sorted if it["id"] in include]

        starts = [it["start"] for it in filtered]
        ends = [it["end"] for it in filtered]
        plot_from = min(plot_from, min(starts))
        plot_to = max(plot_to, max(ends))

    if not filtered:
        st.info("Keine Tasks im Zeitraum.")
        return

    # y mapping
    ymap: Dict[str, int] = {it["id"]: i for i, it in enumerate(filtered)}
    labels: List[str] = []
    for it in filtered:
        sid = it["id"]
        s = series_by_id.get(sid)
        labels.append(_series_label(s) if s else sid)

    item_dates = {it["id"]: (it["start"], it["end"]) for it in filtered}

    fig_h = max(4.0, min(0.45 * len(filtered) + 2.5, 18.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # bars
    for it in filtered:
        sid = it["id"]
        stt, en = it["start"], it["end"]
        y = ymap[sid]

        left = mdates.date2num(stt)
        right = mdates.date2num(en + timedelta(days=1))  # inclusive end
        width = max(0.5, right - left)

        ax.barh(
            y=y,
            width=width,
            left=left,
            height=0.6,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.axvline(mdates.date2num(today), linestyle="--", linewidth=1.0)

    # arrows (thicker, no clip)
    if show_deps:
        for it in filtered:
            sid = it["id"]
            succ = series_by_id.get(sid)

            preds: List[str] = []
            if succ:
                preds = _get_preds_from_series(succ)

            v = it.get("predecessors", [])
            if isinstance(v, list):
                for x in v:
                    xs = str(x).strip()
                    if xs and xs not in preds:
                        preds.append(xs)

            succ_start, _ = item_dates.get(sid, (None, None))
            if not succ_start:
                continue

            for pid in preds:
                if pid not in ymap:
                    continue
                _ps, pe = item_dates.get(pid, (None, None))
                if not pe:
                    continue

                # inside bars + ensure visible horizontal length
                pred_right = mdates.date2num(pe) + 1.0
                succ_left = mdates.date2num(succ_start)
                x0 = pred_right - 0.12
                x1 = succ_left + 0.12
                if x1 <= x0:
                    x1 = x0 + 0.35

                y0 = ymap[pid]
                y1 = ymap[sid]

                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        linewidth=1.6,
                        shrinkA=0,
                        shrinkB=0,
                        connectionstyle="angle3,angleA=0,angleB=90",
                    ),
                    zorder=50,
                    clip_on=False,
                )

    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=9)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    ax.set_xlim(mdates.date2num(plot_from), mdates.date2num(plot_to + timedelta(days=1)))
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Tasks")

    fig.tight_layout()
    st.pyplot(fig)


def render(ctx: dict) -> None:
    st.title("Gantt")
    st.caption(f"ui_gantt v{__version__}")

    _flash_render()

    today = _get_today(ctx)

    series_all = _get_series_list(ctx)
    if not series_all:
        st.info("Noch keine Tasks vorhanden.")
        return

    # full mapping (not filtered away)
    series_by_id_all = {str(getattr(s, "series_id", "") or "").strip(): s for s in series_all}
    series_by_id_all = {k: v for k, v in series_by_id_all.items() if k}

    gantt_fn = ctx.get("gantt_items")
    items_raw: List[Dict[str, Any]] = []
    if callable(gantt_fn):
        try:
            items_raw = list(gantt_fn(series_all, today=today) or [])
        except TypeError:
            items_raw = list(gantt_fn(series_all) or [])
        except Exception:
            items_raw = []
    else:
        # fallback
        for s in series_all:
            sid = str(getattr(s, "series_id", "") or "").strip()
            if not sid:
                continue
            items_raw.append(
                {
                    "id": sid,
                    "title": getattr(s, "title", ""),
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "project": getattr(s, "project", ""),
                    "owner": getattr(s, "owner", ""),
                    "predecessors": _get_preds_from_series(s),
                }
            )

    # filters
    with st.expander("Filters & Zeitraum", expanded=True):
        win_def_from, win_def_to = _compute_window_defaults(items_raw, today)
        rng = st.date_input("Zeitraum (From / To)", value=(win_def_from, win_def_to), key="gantt_window")
        if isinstance(rng, tuple) and len(rng) == 2:
            win_from, win_to = rng[0], rng[1]
        else:
            win_from, win_to = win_def_from, win_def_to

        show_deps = st.checkbox("Dependencies-Pfeile anzeigen", value=True, key="gantt_show_deps")

    with st.expander("➕ Dependencies bearbeiten", expanded=False):
        # editor uses filtered list? keep it simple: all series
        _render_dependency_editor(ctx, list(series_all))

    st.markdown("---")
    st.caption("Hinweis: Vorgänger außerhalb des Zeitfensters werden automatisch mitgezeichnet, wenn Pfeile aktiv sind.")
    _plot_gantt(
        items=items_raw,
        series_by_id=series_by_id_all,
        today=today,
        win_from=win_from,
        win_to=win_to,
        show_deps=show_deps,
    )
