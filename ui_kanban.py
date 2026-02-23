# ui_kanban.py
from __future__ import annotations

from datetime import date
import streamlit as st


KANBAN_STATES = ["PLANNED", "ACTIVE", "BLOCKED", "DONE", "CANCELLED"]


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def _in_filters(s, f_portfolio, f_project, f_owner, f_theme, f_meta) -> bool:
    pf = _norm(getattr(s, "portfolio", "") or "Default") or "Default"
    pr = _norm(getattr(s, "project", "") or "")
    ow = _norm(getattr(s, "owner", "") or "")
    th = _norm(getattr(s, "theme", "") or "")
    meta = bool(getattr(s, "is_meta", False))

    if f_portfolio and pf not in f_portfolio:
        return False
    if f_project and pr not in f_project:
        return False
    if f_owner and ow not in f_owner:
        return False
    if f_theme and th not in f_theme:
        return False
    if f_meta is not None:
        if f_meta is True and not meta:
            return False
        if f_meta is False and meta:
            return False
    return True


def _today_done(s, today: date) -> bool:
    return today in (getattr(s, "done_days", set()) or set())


def _card_subline(s) -> str:
    pf = _norm(getattr(s, "portfolio", "") or "Default") or "Default"
    pr = _norm(getattr(s, "project", "") or "")
    th = _norm(getattr(s, "theme", "") or "")
    ow = _norm(getattr(s, "owner", "") or "")
    parts = [pf]
    if pr:
        parts.append(pr)
    if th:
        parts.append(th)
    if ow:
        parts.append(ow)
    return " · ".join(parts)


def _date_line(s) -> str:
    try:
        return f"{s.start.isoformat()}→{s.end.isoformat()}"
    except Exception:
        return ""


def render(ctx: dict) -> None:
    today: date = ctx["today"]
    visible_series = ctx["visible_series"]

    is_task = ctx["is_task"]
    is_completed = ctx["is_completed"]

    persist = ctx["persist"]
    sync_lists_from_data = ctx["sync_lists_from_data"]

    open_detail = ctx["open_detail"]
    request_done_single = ctx["request_done_single"]
    unmark_done = ctx["unmark_done"]

    st.title("Kanban")
    st.caption("State-based board (wie Linear/Notion light). Kein Drag&Drop – aber schneller State-Wechsel pro Karte.")

    series = [s for s in visible_series() if is_task(s)]

    # ----- filters (minimal, wie bisher über expander)
    with st.expander("Filters", expanded=False):
        portfolios = sorted({(_norm(getattr(s, "portfolio", "") or "Default") or "Default") for s in series})
        projects = sorted({_norm(getattr(s, "project", "") or "") for s in series if _norm(getattr(s, "project", "") or "")})
        owners = sorted({_norm(getattr(s, "owner", "") or "") for s in series if _norm(getattr(s, "owner", "") or "")})
        themes = sorted({_norm(getattr(s, "theme", "") or "") for s in series if _norm(getattr(s, "theme", "") or "")})

        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])
        f_portfolio = c1.multiselect("Portfolio", portfolios, default=[], key="kb_f_portfolio")
        f_project = c2.multiselect("Project", projects, default=[], key="kb_f_project")
        f_owner = c3.multiselect("Owner", owners, default=[], key="kb_f_owner")
        f_theme = c4.multiselect("Theme", themes, default=[], key="kb_f_theme")
        meta_mode = c5.selectbox("Meta", options=["All", "Normal only", "META only"], index=0, key="kb_f_meta_mode")

        f_meta = None
        if meta_mode == "Normal only":
            f_meta = False
        elif meta_mode == "META only":
            f_meta = True

    # apply filters
    filtered = [
        s for s in series
        if _in_filters(s, f_portfolio, f_project, f_owner, f_theme, f_meta)
    ]

    # group by state (fallback if missing)
    buckets = {stt: [] for stt in KANBAN_STATES}
    for s in filtered:
        stt = _norm(getattr(s, "state", "") or "").upper()
        if stt not in buckets:
            stt = "PLANNED"
        buckets[stt].append(s)

    # sort inside buckets: by end date, then owner, then title
    for stt in buckets:
        buckets[stt].sort(key=lambda x: (x.end, _norm(x.owner), _norm(x.title)))

    # ----- board columns (wie gehabt)
    cols = st.columns(len(KANBAN_STATES))
    for i, stt in enumerate(KANBAN_STATES):
        with cols[i]:
            st.subheader(f"{stt} ({len(buckets[stt])})")

            for s in buckets[stt]:
                sid = s.series_id

                # --- header: title (clickable) + meta line
                if st.button(str(s.title or ""), key=f"kb_title_{sid}", use_container_width=True):
                    open_detail(sid)
                st.caption(_card_subline(s))
                st.caption(_date_line(s))

                # --- controls row: state dropdown + actions (popover)
                c_state, c_actions = st.columns([7, 2])

                # STATE dropdown (persist on change)
                def _set_state(series_id: str, key: str):
                    new_state = st.session_state.get(key)
                    if not new_state:
                        return
                    ss = next((x for x in ctx["visible_series"]() if x.series_id == series_id), None)
                    # visible_series() may filter out, so fall back to global session list
                    if ss is None:
                        ss = next((x for x in st.session_state.series if x.series_id == series_id), None)
                    if ss is None:
                        return
                    ss.state = new_state
                    persist()
                    sync_lists_from_data()

                state_key = f"kb_state_{sid}"
                # initialize selectbox state
                if state_key not in st.session_state:
                    st.session_state[state_key] = stt

                c_state.selectbox(
                    label="",
                    options=KANBAN_STATES,
                    key=state_key,
                    label_visibility="collapsed",
                    on_change=_set_state,
                    args=(sid, state_key),
                )

                # Actions: reduce button clutter per card
                with c_actions.popover("⋯", use_container_width=True):
                    if st.button("Open details", key=f"kb_open_{sid}", use_container_width=True):
                        open_detail(sid)

                    st.divider()

                    # DONE today toggle
                    done_today = _today_done(s, today)
                    is_future = today < s.start  # your rule: future disabled for DONE
                    disabled = bool(is_future)

                    label = "Mark DONE (today)" if not done_today else "Undo DONE (today)"
                    help_txt = "Needs confirmation" if not done_today else "Remove DONE for today"
                    btn_key = f"kb_done_{sid}_{today.isoformat()}"

                    if st.button(label, key=btn_key, help=help_txt, disabled=disabled, use_container_width=True):
                        if done_today:
                            try:
                                unmark_done(s, today)
                            except TypeError:
                                unmark_done(s, today)
                            persist()
                            st.rerun()
                        else:
                            request_done_single(sid, today, reason="Kanban quick DONE")
                            st.rerun()

                # small spacer
                st.markdown("---")
