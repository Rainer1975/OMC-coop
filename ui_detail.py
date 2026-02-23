# ui_detail.py
from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x) if isinstance(x, (set, tuple)) else []


def _incoming_dependents(series_id: str) -> list:
    """Series that depend on this series_id (incoming edges).

    Supports legacy fields:
    - s.predecessors (current)
    - s.depends_on (legacy)
    - s.meta['predecessors'] / s.meta['depends_on']
    """
    out = []
    for s in st.session_state.get("series", []) or []:
        try:
            preds = []
            v = getattr(s, "predecessors", None)
            if isinstance(v, list):
                preds.extend([str(x).strip() for x in v if str(x).strip()])
            v2 = getattr(s, "depends_on", None)
            if isinstance(v2, list):
                preds.extend([str(x).strip() for x in v2 if str(x).strip()])

            m = getattr(s, "meta", {}) or {}
            if isinstance(m, dict):
                vm = m.get("predecessors", [])
                if isinstance(vm, list):
                    preds.extend([str(x).strip() for x in vm if str(x).strip()])
                vm2 = m.get("depends_on", [])
                if isinstance(vm2, list):
                    preds.extend([str(x).strip() for x in vm2 if str(x).strip()])

            if series_id in preds:
                out.append(s)
        except Exception:
            continue
    return out


def _series_label(s: Any) -> str:
    # Keep it stable and readable for dependency pickers
    pf = (getattr(s, "portfolio", "") or "").strip()
    pr = (getattr(s, "project", "") or "").strip()
    ow = (getattr(s, "owner", "") or "").strip()
    tt = (getattr(s, "title", "") or "").strip()
    return " · ".join([x for x in [pf, pr, ow, tt] if x])


def render(ctx: dict) -> None:
    today: date = ctx["today"]
    open_id = st.session_state.get("open_series")
    if not open_id:
        st.info("Kein Task ausgewählt.")
        if st.button("↩️ Zurück"):
            ctx["close_detail"]()
        return

    s = ctx["find_series"](open_id)
    sid = s.series_id

    st.markdown("## Detail")

    top_l, top_r = st.columns([6, 4])
    with top_l:
        if st.button("↩️ Zurück", key=f"detail_back_{sid}"):
            ctx["close_detail"]()

    # -------------------------
    # Delete section (hard delete)
    # -------------------------
    incoming = _incoming_dependents(sid)

    with top_r:
        with st.popover("🗑️ Löschen", use_container_width=True):
            st.error("Achtung: Löschen entfernt den Eintrag dauerhaft aus data.json.")
            if incoming:
                st.warning(
                    "Andere Tasks hängen davon ab:\n\n"
                    + "\n".join([f"- {(_series_label(x) or x.series_id)}" for x in incoming])
                )
                fix_refs = st.checkbox(
                    "Abhängigkeiten in anderen Tasks automatisch entfernen",
                    value=True,
                    key=f"detail_del_fixrefs_{sid}",
                )
            else:
                fix_refs = True

            confirm = st.checkbox(
                "Ich bestätige: wirklich löschen",
                value=False,
                key=f"detail_del_confirm_{sid}",
            )
            if st.button(
                "🗑️ Jetzt löschen",
                key=f"detail_del_do_{sid}",
                disabled=not confirm,
                use_container_width=True,
            ):
                # Remove references in other tasks if requested
                if incoming and fix_refs:
                    for other in incoming:
                        # Normalize: remove sid from predecessors/depends_on and meta mirrors
                        try:
                            preds = _as_list(getattr(other, "predecessors", []) or [])
                            if sid in preds:
                                preds = [x for x in preds if x != sid]
                                try:
                                    other.predecessors = preds
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            deps = _as_list(getattr(other, "depends_on", []) or [])
                            if sid in deps:
                                deps = [x for x in deps if x != sid]
                                try:
                                    other.depends_on = deps
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            m = getattr(other, "meta", {}) or {}
                            if isinstance(m, dict):
                                m2 = dict(m)
                                for k in ("predecessors", "depends_on"):
                                    vv = m2.get(k, [])
                                    if isinstance(vv, list) and sid in vv:
                                        m2[k] = [x for x in vv if x != sid]
                                other.meta = m2
                        except Exception:
                            pass

                ctx["delete_series"](sid)
                try:
                    ctx["sync_lists_from_data"]()
                except Exception:
                    pass
                st.success("Task gelöscht.")
                # close_detail reruns; if not available for some reason, do a rerun
                try:
                    ctx["close_detail"]()
                except Exception:
                    st.session_state.open_series = None
                    st.session_state.page = st.session_state.get("page_prev", "HOME")
                    st.rerun()

    st.markdown("---")

    # -------------------------
    # Basic fields
    # -------------------------
    kind = (getattr(s, "kind", "") or "").strip().lower()
    is_task = bool(ctx["is_task"](s)) if "is_task" in ctx else (kind == "task")
    is_appt = bool(ctx["is_appointment"](s)) if "is_appointment" in ctx else (kind == "appointment")

    left, right = st.columns([2, 1])

    with left:
        new_title = st.text_input("Title", value=(s.title or ""), key=f"detail_title_{sid}")

        # Portfolio / Project / Theme pickers
        new_portfolio = ctx["pick_from_list"](
            "Portfolio",
            key=f"detail_pf_{sid}",
            values=(ctx["lists"].get("portfolios", []) if "lists" in ctx else []),
            current=(getattr(s, "portfolio", "") or ""),
            require=True,
            list_key="portfolios",
        )
        new_project = ctx["pick_from_list"](
            "Project",
            key=f"detail_pr_{sid}",
            values=(ctx["lists"].get("projects", []) if "lists" in ctx else []),
            current=(getattr(s, "project", "") or ""),
            require=True,
            list_key="projects",
        )
        new_theme = ctx["pick_from_list"](
            "Theme",
            key=f"detail_th_{sid}",
            values=(ctx["lists"].get("themes", []) if "lists" in ctx else []),
            current=(getattr(s, "theme", "") or ""),
            require=True,
            list_key="themes",
        )

        # Owner picker
        new_owner_id, new_owner_display = ctx["pick_owner"](
            "Owner",
            key=f"detail_owner_{sid}",
            current_owner_id=(getattr(s, "owner_id", "") or ""),
            current_owner_display=(getattr(s, "owner", "") or ""),
            employees=ctx.get("employees", []),
        )

        # State + META (tasks only)
        if is_task:
            TASK_STATES = ctx.get("TASK_STATES", ["PLANNED", "ACTIVE", "BLOCKED", "DONE", "CANCELLED"])
            cur_state = (getattr(s, "state", "") or "ACTIVE").strip()
            if cur_state not in TASK_STATES:
                TASK_STATES = [cur_state] + [x for x in TASK_STATES if x != cur_state]
            new_state = st.selectbox(
                "State",
                options=TASK_STATES,
                index=TASK_STATES.index(cur_state),
                key=f"detail_state_{sid}",
            )
            new_is_meta = st.checkbox(
                "META",
                value=bool(getattr(s, "is_meta", False)),
                key=f"detail_meta_{sid}",
                disabled=bool(st.session_state.get("focus_mode", False)),
            )
        else:
            new_state = (getattr(s, "state", "") or "").strip()
            new_is_meta = bool(getattr(s, "is_meta", False))

        # Dates
        c1, c2 = st.columns(2)
        new_start = c1.date_input("Start", value=getattr(s, "start", today), key=f"detail_start_{sid}")
        new_end = c2.date_input("End", value=getattr(s, "end", today), key=f"detail_end_{sid}")

        if new_start > new_end:
            st.error("Start muss <= End sein.")

        # Appointment fields
        if is_appt:
            c3, c4 = st.columns(2)
            new_time_start = c3.text_input(
                "Time start (HH:MM)",
                value=(getattr(s, "time_start", "") or ""),
                key=f"detail_ts_{sid}",
            )
            new_time_end = c4.text_input(
                "Time end (HH:MM)",
                value=(getattr(s, "time_end", "") or ""),
                key=f"detail_te_{sid}",
            )
            new_location = st.text_input(
                "Location",
                value=(getattr(s, "location", "") or ""),
                key=f"detail_loc_{sid}",
            )
        else:
            new_time_start = getattr(s, "time_start", "") or ""
            new_time_end = getattr(s, "time_end", "") or ""
            new_location = getattr(s, "location", "") or ""

        new_notes = st.text_area(
            "Notes",
            value=(getattr(s, "notes", "") or ""),
            key=f"detail_notes_{sid}",
            height=120,
        )

        # Save changes
        save_disabled = bool(new_start > new_end) or not (new_title or "").strip()
        if st.button("💾 Speichern", key=f"detail_save_{sid}", disabled=save_disabled):
            # Apply changes
            s.title = (new_title or "").strip()
            s.portfolio = (new_portfolio or "").strip()
            s.project = (new_project or "").strip()
            s.theme = (new_theme or "").strip()

            s.owner = (new_owner_display or "").strip()
            try:
                s.owner_id = (new_owner_id or "").strip()
            except Exception:
                pass

            try:
                s.state = (new_state or "").strip()
            except Exception:
                pass
            try:
                s.is_meta = bool(new_is_meta)
            except Exception:
                pass

            s.start = new_start
            s.end = new_end

            if is_appt:
                try:
                    s.time_start = (new_time_start or "").strip()
                    s.time_end = (new_time_end or "").strip()
                    s.location = (new_location or "").strip()
                except Exception:
                    pass

            try:
                s.notes = (new_notes or "").strip()
            except Exception:
                pass

            ctx["persist"]()
            try:
                ctx["sync_lists_from_data"]()
            except Exception:
                pass
            st.success("Gespeichert.")
            st.rerun()

    # -------------------------
    # Right side: status, progress, DONE handling, dependencies
    # -------------------------
    with right:
        try:
            prog = ctx["progress_percent"](s, today)
        except Exception:
            prog = 0.0

        st.metric("Progress%", f"{prog:.0f}")

        if is_task:
            done_days = set(getattr(s, "done_days", set()) or set())
            st.caption("DONE Tage (Future disabled).")
            mark_day = st.date_input("Tag auswählen", value=today, key=f"detail_done_pick_{sid}")
            c1, c2 = st.columns(2)
            if c1.button("✅ DONE setzen", key=f"detail_done_set_{sid}"):
                if mark_day > today:
                    st.warning("Future ist deaktiviert. Kein DONE in der Zukunft.")
                else:
                    ctx["request_done_single"](sid, mark_day, reason="Detail: DONE setzen")
            if c2.button("↩️ DONE entfernen", key=f"detail_done_unset_{sid}"):
                if mark_day in done_days:
                    if "actor" in getattr(ctx["unmark_done"], "__code__", type("x", (), {"co_varnames": ()}))().co_varnames:
                        ctx["unmark_done"](s, mark_day, actor="ui")
                    else:
                        ctx["unmark_done"](s, mark_day)
                    # state refresh best-effort
                    try:
                        if ctx["is_completed"](s, today):
                            s.state = "DONE"
                        elif (getattr(s, "state", "") == "DONE") and not ctx["is_completed"](s, today):
                            s.state = "ACTIVE" if ctx["is_active"](s, today) else "PLANNED"
                    except Exception:
                        pass
                    ctx["persist"]()
                    st.success("DONE entfernt.")
                    st.rerun()
                else:
                    st.info("Der Tag ist nicht als DONE markiert.")

        st.markdown("---")
        st.subheader("Dependencies")

        all_series = st.session_state.get("series", []) or []
        selectable = [x for x in all_series if x.series_id != sid]
        selectable.sort(key=lambda x: _norm(_series_label(x)))

        # Current deps
        cur_deps = _as_list(getattr(s, "depends_on", []) or [])
        id_to_label = {x.series_id: _series_label(x) or x.series_id for x in selectable}
        label_to_id = {v: k for k, v in id_to_label.items()}

        cur_labels = [id_to_label[d] for d in cur_deps if d in id_to_label]
        pick_labels = st.multiselect(
            "Dieser Task hängt ab von …",
            options=list(label_to_id.keys()),
            default=cur_labels,
            key=f"detail_deps_pick_{sid}",
        )
        new_deps = [label_to_id[l] for l in pick_labels if l in label_to_id]

        # Validate: no cycles
        cycle = False
        if new_deps:
            for dep_id in new_deps:
                try:
                    if ctx["would_create_cycle"](sid, dep_id, st.session_state.series):
                        cycle = True
                        break
                except Exception:
                    # if helper signature differs, do a conservative check using can_depend_series if available
                    try:
                        a = s
                        b = next(x for x in st.session_state.series if x.series_id == dep_id)
                        if "can_depend_series" in ctx and not ctx["can_depend_series"](a, b):
                            cycle = True
                            break
                    except Exception:
                        pass

        if cycle:
            st.error("Ungültig: würde einen Zyklus erzeugen. Auswahl bitte ändern.")

        if st.button("🔗 Dependencies speichern", key=f"detail_deps_save_{sid}", disabled=cycle):
            try:
                s.depends_on = new_deps
            except Exception:
                # last resort: set attribute dynamically
                setattr(s, "depends_on", new_deps)
            ctx["persist"]()
            st.success("Dependencies gespeichert.")
            st.rerun()

        if incoming:
            st.caption("Tasks, die von diesem Task abhängen:")
            for x in incoming:
                st.write(f"- {(_series_label(x) or x.series_id)}")
