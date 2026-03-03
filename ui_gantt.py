# ui_gantt.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import streamlit as st

from core import TaskSeries, would_create_cycle

__version__ = "2026.03.03.4"


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


def _series_id(s: TaskSeries) -> str:
    return str(getattr(s, "series_id", "") or "").strip()


def _series_label(s: TaskSeries) -> str:
    title = (getattr(s, "title", "") or "").strip()
    project = (getattr(s, "project", "") or "").strip()
    owner = (getattr(s, "owner", "") or "").strip()
    bits = [b for b in [title, project, owner] if b]
    return " — ".join(bits) if bits else _series_id(s)


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


def _get_all_series(ctx: dict) -> List[TaskSeries]:
    # IMPORTANT: always use the canonical list as source of truth
    # (visible_series can be a filtered view)
    ss = st.session_state.get("series", [])
    if isinstance(ss, list) and ss:
        return ss
    # fallback
    vs = ctx.get("visible_series")
    if callable(vs):
        try:
            return list(vs())
        except Exception:
            pass
    return []


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

    out: List[str] = []
    seen = set()
    for p in preds:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _write_preds(s: TaskSeries, pred_ids: List[str]) -> None:
    pred_ids = [str(x).strip() for x in (pred_ids or []) if str(x).strip()]
    # write attribute
    try:
        setattr(s, "predecessors", list(pred_ids))
    except Exception:
        pass
    # write meta
    try:
        m = getattr(s, "meta", {}) or {}
        if not isinstance(m, dict):
            m = {}
        m = dict(m)
        m["predecessors"] = list(pred_ids)
        setattr(s, "meta", m)
    except Exception:
        pass


def _apply_preds_to_canonical_state(succ_id: str, pred_ids: List[str]) -> bool:
    """Critical fix: always update the TaskSeries inside st.session_state.series."""
    succ_id = str(succ_id or "").strip()
    if not succ_id:
        return False
    series_list = st.session_state.get("series", [])
    if not isinstance(series_list, list):
        return False
    for s in series_list:
        if _series_id(s) == succ_id:
            _write_preds(s, pred_ids)
            return True
    return False


def _compute_window_defaults(items: List[Dict[str, Any]], today: date) -> Tuple[date, date]:
    starts = [_safe_date(it.get("start")) for it in items]
    ends = [_safe_date(it.get("end")) for it in items]
    starts = [d for d in starts if d]
    ends = [d for d in ends if d]
    if not starts or not ends:
        return today, today + timedelta(days=14)
    return min(starts), max(ends)


def _render_dependency_editor(ctx: dict, series_all: List[TaskSeries]) -> None:
    if not series_all:
        st.info("Keine Tasks vorhanden.")
        return

    succ = st.selectbox(
        "Nachfolger (Succ)",
        options=series_all,
        format_func=_series_label,
        key="gantt_dep_succ",
    )
    if not succ:
        return

    succ_id = _series_id(succ)
    if not succ_id:
        st.error("Succ hat keine series_id.")
        return

    # IMPORTANT: load current preds from canonical object, not from succ instance
    canonical = None
    for s in st.session_state.get("series", []) or []:
        if _series_id(s) == succ_id:
            canonical = s
            break
    current = _get_preds_from_series(canonical) if canonical else _get_preds_from_series(succ)

    available_preds = [s for s in series_all if _series_id(s) and _series_id(s) != succ_id]

    # build default list by id match
    by_id = {_series_id(s): s for s in available_preds}
    default_objs = [by_id[p] for p in current if p in by_id]

    picked = st.multiselect(
        "Vorgänger (Pred) – mehrere möglich",
        options=available_preds,
        default=default_objs,
        format_func=_series_label,
        key="gantt_dep_preds",
    )

    pred_ids = [_series_id(p) for p in picked]
    pred_ids = [p for p in pred_ids if p and p != succ_id]

    # cycle check using canonical list
    for pid in pred_ids:
        try:
            if would_create_cycle(series_all, src_id=pid, tgt_id=succ_id):
                st.error("Ungültig: Diese Abhängigkeit würde einen Zyklus erzeugen.")
                return
        except Exception:
            st.warning("Zyklus-Check nicht verfügbar/fehlgeschlagen – bitte vorsichtig.")
            break

    c1, c2, c3 = st.columns([1, 1, 6])

    if c1.button("💾 Speichern", key="gantt_dep_save", use_container_width=True):
        ok_update = _apply_preds_to_canonical_state(succ_id, pred_ids)
        if not ok_update:
            _flash_set("error", "Save fehlgeschlagen: Succ nicht im session_state gefunden.")
            st.rerun()

        ok, err = _persist(ctx)
        _flash_set("success", "Dependencies gespeichert.") if ok else _flash_set("error", err or "Save fehlgeschlagen.")
        st.rerun()

    if c2.button("🧹 Leeren", key="gantt_dep_clear", use_container_width=True):
        ok_update = _apply_preds_to_canonical_state(succ_id, [])
        if not ok_update:
            _flash_set("error", "Clear fehlgeschlagen: Succ nicht im session_state gefunden.")
            st.rerun()

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
    show_debug: bool,
) -> None:
    # normalize items
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
                "predecessors": it.get("predecessors", []),
            }
        )

    def sort_key(it: Dict[str, Any]) -> Tuple[str, str, str]:
        sid = it["id"]
        s = series_by_id.get(sid)
        project = (getattr(s, "project", "") or "").strip() if s else str(it.get("project") or "").strip()
        owner = (getattr(s, "owner", "") or "").strip() if s else str(it.get("owner") or "").strip()
        title = (getattr(s, "title", "") or "").strip() if s else str(it.get("title") or "").strip()
        return (_norm(project), _norm(owner), _norm(title))

    items_sorted = sorted(norm_items, key=sort_key)

    filtered = [it for it in items_sorted if not (it["end"] < win_from or it["start"] > win_to)]
    plot_from = win_from
    plot_to = win_to

    if show_deps and filtered:
        item_by_id = {it["id"]: it for it in items_sorted}
        include = {it["id"] for it in filtered}

        def preds_for_id(sid: str) -> List[str]:
            preds: List[str] = []
            s = series_by_id.get(sid)
            if s:
                preds.extend(_get_preds_from_series(s))
            v = item_by_id.get(sid, {}).get("predecessors", [])
            if isinstance(v, list):
                for x in v:
                    xs = str(x).strip()
                    if xs and xs not in preds:
                        preds.append(xs)
            out, seen = [], set()
            for p in preds:
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
            return out

        for it in list(filtered):
            for pid in preds_for_id(it["id"]):
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

    ymap = {it["id"]: i for i, it in enumerate(filtered)}
    labels = []
    for it in filtered:
        sid = it["id"]
        s = series_by_id.get(sid)
        labels.append(_series_label(s) if s else sid)

    item_dates = {it["id"]: (it["start"], it["end"]) for it in filtered}

    edges: List[Tuple[str, str]] = []
    if show_deps:
        for it in filtered:
            sid = it["id"]
            preds: List[str] = []
            s = series_by_id.get(sid)
            if s:
                preds.extend(_get_preds_from_series(s))
            v = it.get("predecessors", [])
            if isinstance(v, list):
                for x in v:
                    xs = str(x).strip()
                    if xs and xs not in preds:
                        preds.append(xs)
            for pid in preds:
                edges.append((pid, sid))

    if show_debug:
        with st.expander("Debug: erkannte Dependencies (Pred → Succ)", expanded=True):
            if not edges:
                st.write("Kanten erkannt: 0  (im State ist nichts gespeichert)")
            else:
                st.write(f"Kanten erkannt: {len(edges)}")
                st.dataframe(
                    [{"pred": p, "succ": s, "pred_in_plot": p in ymap, "succ_in_plot": s in ymap} for p, s in edges],
                    use_container_width=True,
                )

    fig_h = max(4.0, min(0.45 * len(filtered) + 2.5, 18.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    for it in filtered:
        sid = it["id"]
        stt, en = it["start"], it["end"]
        y = ymap[sid]
        left = mdates.date2num(stt)
        right = mdates.date2num(en + timedelta(days=1))
        width = max(0.5, right - left)
        ax.barh(y=y, width=width, left=left, height=0.6, alpha=0.85, edgecolor="black", linewidth=0.6)

    ax.axvline(mdates.date2num(today), linestyle="--", linewidth=1.0)

    if show_deps:
        for pred_id, succ_id in edges:
            if pred_id not in ymap or succ_id not in ymap:
                continue
            succ_start, _ = item_dates.get(succ_id, (None, None))
            _ps, pred_end = item_dates.get(pred_id, (None, None))
            if not succ_start or not pred_end:
                continue

            pred_right = mdates.date2num(pred_end) + 1.0
            succ_left = mdates.date2num(succ_start)

            x0 = pred_right - 0.15
            x1 = succ_left + 0.15
            if x1 <= x0:
                x1 = x0 + 0.50

            y0 = ymap[pred_id]
            y1 = ymap[succ_id]

            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.8,
                    connectionstyle="angle3,angleA=0,angleB=90",
                    zorder=80,
                    clip_on=False,
                )
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
    series_all = _get_all_series(ctx)
    if not series_all:
        st.info("Noch keine Tasks vorhanden.")
        return

    series_by_id = {_series_id(s): s for s in series_all if _series_id(s)}

    gantt_fn = ctx.get("gantt_items")
    items_raw: List[Dict[str, Any]] = []
    if callable(gantt_fn):
        try:
            items_raw = list(gantt_fn(series_all, today=today) or [])
        except TypeError:
            items_raw = list(gantt_fn(series_all) or [])
        except Exception:
            items_raw = []

    with st.expander("Filters & Zeitraum", expanded=True):
        win_def_from, win_def_to = _compute_window_defaults(items_raw, today)
        rng = st.date_input("Zeitraum (From / To)", value=(win_def_from, win_def_to), key="gantt_window")
        if isinstance(rng, tuple) and len(rng) == 2:
            win_from, win_to = rng[0], rng[1]
        else:
            win_from, win_to = win_def_from, win_def_to

        show_deps = st.checkbox("Dependencies-Pfeile anzeigen", value=True, key="gantt_show_deps")
        show_debug = st.checkbox("Debug anzeigen", value=False, key="gantt_show_debug")

    with st.expander("➕ Dependencies bearbeiten", expanded=False):
        _render_dependency_editor(ctx, list(series_all))

    st.markdown("---")
    _plot_gantt(
        items=items_raw,
        series_by_id=series_by_id,
        today=today,
        win_from=win_from,
        win_to=win_to,
        show_deps=show_deps,
        show_debug=show_debug,
    )
