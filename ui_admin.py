# ui_admin.py
# Admin UI: Employees + Lists + Hygiene + Stats
# Fix: IDs are ALWAYS assigned on save (None/"None"/"" treated as missing), unique + stable.

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st


# ----------------------------
# Helpers
# ----------------------------

def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def _slugify(s: str) -> str:
    s = _norm_str(s).lower()
    if not s:
        return ""
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_aliases(s: Any) -> List[str]:
    raw = _norm_str(s)
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    out: List[str] = []
    seen = set()
    for p in parts:
        p2 = _norm_str(p)
        if not p2:
            continue
        if p2 in seen:
            continue
        seen.add(p2)
        out.append(p2)
    return out


def _aliases_to_str(aliases: Any) -> str:
    # accepts list or string
    if aliases is None:
        return ""
    if isinstance(aliases, list):
        cleaned = [_norm_str(a) for a in aliases]
        cleaned = [a for a in cleaned if a]
        return ", ".join(cleaned)
    return ", ".join(_parse_aliases(aliases))


def _make_unique_id(base: str, used: set[str]) -> str:
    base = _norm_str(base)
    if not base:
        base = "emp"
    candidate = base
    i = 2
    while candidate in used or candidate.lower() in {"none", "null"}:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _normalize_employees(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Normalizes and guarantees:
    - id exists (no None/"None"/""), unique
    - display_name exists if possible
    - aliases is a list[str]
    - drops completely empty rows
    Returns (normalized, warnings)
    """
    warnings: List[str] = []

    # first pass: collect used ids that are already valid
    used: set[str] = set()
    for r in rows:
        rid = _norm_str(r.get("id"))
        if rid:
            used.add(rid)

    normalized: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows, start=1):
        rid = _norm_str(r.get("id"))
        name = _norm_str(r.get("display_name"))
        aliases = _parse_aliases(r.get("aliases"))

        # drop fully empty rows
        if not rid and not name and not aliases:
            continue

        # If id missing -> derive from display_name (slug) or from first alias or fallback
        if not rid:
            base = _slugify(name)
            if not base and aliases:
                base = _slugify(aliases[0])
            if not base:
                base = "emp"
                warnings.append(f"Row {idx}: keine ID/kein Name → fallback ID wird vergeben.")
            rid = _make_unique_id(base, used)

        # If name missing -> use id
        if not name:
            name = rid

        # ensure aliases always include name/id if useful? -> NO, keep user intent.
        normalized.append(
            {
                "id": rid,
                "display_name": name,
                "aliases": aliases,
            }
        )

    # final: ensure uniqueness again (in case of weird input)
    seen: set[str] = set()
    fixed: List[Dict[str, Any]] = []
    for r in normalized:
        rid = _norm_str(r["id"])
        if rid in seen:
            new_id = _make_unique_id(rid, used)
            warnings.append(f"Doppelte ID '{rid}' → umbenannt auf '{new_id}'.")
            r = dict(r)
            r["id"] = new_id
            rid = new_id
        seen.add(rid)
        fixed.append(r)

    return fixed, warnings


def _df_from_employees(employees: List[Dict[str, Any]]) -> pd.DataFrame:
    data = []
    for e in employees:
        data.append(
            {
                "id": _norm_str(e.get("id")),
                "display_name": _norm_str(e.get("display_name")),
                "aliases (comma separated)": _aliases_to_str(e.get("aliases")),
            }
        )
    if not data:
        data = [{"id": "", "display_name": "", "aliases (comma separated)": ""}]
    return pd.DataFrame(data)


def _employees_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = _norm_str(row.get("id"))
        name = _norm_str(row.get("display_name"))
        aliases = row.get("aliases (comma separated)")
        rows.append({"id": rid, "display_name": name, "aliases": aliases})
    return rows


# ----------------------------
# Render
# ----------------------------

def render(ctx: Dict[str, Any]) -> None:
    st.title("Admin")
    st.caption("Stammdaten + Hygiene. Vorsicht: wirkt sofort in allen Views.")

    tab_emps, tab_lists, tab_hygiene, tab_stats = st.tabs(["Employees", "Projects/Themes", "Hygiene", "Stats"])

    # --------------------------------
    # Employees
    # --------------------------------
    with tab_emps:
        st.subheader("Employees (IDs + aliases)")
        st.info("IDs sind der Primärschlüssel. Aliases dienen als Mapping aus alten Daten/Tippos.", icon="ℹ️")

        employees: List[Dict[str, Any]] = list(ctx.get("employees") or [])
        df = _df_from_employees(employees)

        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="admin_employees_editor",
            column_config={
                "id": st.column_config.TextColumn("id", help="Primärschlüssel (wird bei Bedarf automatisch vergeben)"),
                "display_name": st.column_config.TextColumn("display_name", help="Anzeigename"),
                "aliases (comma separated)": st.column_config.TextColumn("aliases (comma separated)", help="Kommagetrennte Aliases"),
            },
        )

        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            if st.button("Save employees", type="primary", use_container_width=True):
                raw_rows = _employees_from_df(edited)
                fixed, warns = _normalize_employees(raw_rows)

                # IMPORTANT: write-back through ctx saver AND session_state
                if "save_employees" not in ctx or not callable(ctx["save_employees"]):
                    st.error("ctx['save_employees'] fehlt oder ist nicht callable. (app.py ctx wiring kaputt)")
                    st.stop()

                ctx["save_employees"](fixed)

                # keep runtime consistent: update session state too
                st.session_state["employees"] = fixed
                ctx["employees"] = fixed  # best-effort (ctx is a dict)

                if warns:
                    for w in warns:
                        st.warning(w)
                st.success(f"Employees gespeichert: {len(fixed)} (IDs wurden bei Bedarf automatisch vergeben).")
                st.rerun()

        with col_b:
            if st.button("Re-map owners from employees (safe)", use_container_width=True):
                if "series" not in ctx or "save_series" not in ctx or "resolve_owner_id" not in ctx:
                    st.error("ctx['series']/ctx['save_series']/ctx['resolve_owner_id'] fehlt – Remap nicht möglich.")
                    st.stop()

                series = list(ctx.get("series") or [])
                resolver = ctx["resolve_owner_id"]
                fixed_count = 0

                for s in series:
                    # series objects expected to have owner_id
                    old = getattr(s, "owner_id", "")
                    new = resolver(old, st.session_state.get("employees", employees))
                    if new and new != old:
                        setattr(s, "owner_id", new)
                        fixed_count += 1

                ctx["save_series"](series)
                st.session_state["series"] = series
                ctx["series"] = series

                st.success(f"Owner-IDs remapped: {fixed_count}")
                st.rerun()

        with col_c:
            # quick sanity
            st.caption("Sanity (Session)")
            sess_emps = st.session_state.get("employees", employees)
            st.write(f"Session Employees: **{len(sess_emps)}**")
            if any(_norm_str(e.get("id")) == "" for e in (sess_emps or [])):
                st.error("Session enthält Employees ohne ID → bitte 'Save employees' drücken.")
            if any(_norm_str(e.get("id")).lower() == "none" for e in (sess_emps or [])):
                st.error("Session enthält ID='None' → das ist kaputt, bitte 'Save employees' drücken.")

    # --------------------------------
    # Projects/Themes (lists.json)
    # --------------------------------
    with tab_lists:
        st.subheader("Projects / Portfolios / Themes")

        lists = dict(ctx.get("lists") or {})
        if not lists:
            lists = {"portfolios": [], "projects": [], "themes": []}

        def _edit_list(title: str, key: str) -> List[str]:
            st.markdown(f"### {title}")
            current = lists.get(key) or []
            df_l = pd.DataFrame({"value": list(current)})
            edited_l = st.data_editor(df_l, num_rows="dynamic", use_container_width=True, key=f"admin_list_{key}")
            vals = []
            for v in edited_l["value"].tolist():
                s = _norm_str(v)
                if s:
                    vals.append(s)
            # de-dupe preserve order
            out = []
            seen = set()
            for v in vals:
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
            return out

        col1, col2, col3 = st.columns(3)
        with col1:
            portfolios = _edit_list("Portfolios", "portfolios")
        with col2:
            projects = _edit_list("Projects", "projects")
        with col3:
            themes = _edit_list("Themes", "themes")

        if st.button("Save lists", type="primary"):
            if "save_lists" not in ctx or not callable(ctx["save_lists"]):
                st.error("ctx['save_lists'] fehlt oder ist nicht callable.")
                st.stop()

            new_lists = {"portfolios": portfolios, "projects": projects, "themes": themes}
            ctx["save_lists"](new_lists)
            st.session_state["lists"] = new_lists
            ctx["lists"] = new_lists
            st.success("Lists gespeichert.")
            st.rerun()

    # --------------------------------
    # Hygiene
    # --------------------------------
    with tab_hygiene:
        st.subheader("Hygiene")
        st.caption("Sichere Checks (keine Magie).")

        employees = st.session_state.get("employees", ctx.get("employees") or [])
        series = st.session_state.get("series", ctx.get("series") or [])

        # Employees hygiene
        missing_ids = [e for e in employees if not _norm_str(e.get("id"))]
        none_ids = [e for e in employees if _norm_str(e.get("id")).lower() == "none"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Employees", len(employees))
        with c2:
            st.metric("Employees w/o id", len(missing_ids))
        with c3:
            st.metric("Employees id='None'", len(none_ids))

        if len(missing_ids) or len(none_ids):
            st.error("Employees sind nicht sauber → geh zu Employees und drück 'Save employees'.")

        # Series hygiene
        if series:
            ids = [getattr(s, "series_id", None) for s in series]
            ids = [_norm_str(x) for x in ids if _norm_str(x)]
            dup = {i for i in ids if ids.count(i) > 1}
            st.metric("Series", len(series))
            st.metric("Duplicate series_id", len(dup))
            if dup:
                st.warning(f"Doppelte series_id: {', '.join(sorted(list(dup))[:20])}")

        st.markdown("---")
        st.caption("Keine destruktiven Auto-Fixes hier. Fixes laufen über die jeweiligen Views + Save-Buttons.")

    # --------------------------------
    # Stats
    # --------------------------------
    with tab_stats:
        st.subheader("Stats")
        employees = st.session_state.get("employees", ctx.get("employees") or [])
        lists = st.session_state.get("lists", ctx.get("lists") or {})
        series = st.session_state.get("series", ctx.get("series") or [])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Employees (session)", len(employees))
        with c2:
            st.metric("Portfolios", len(lists.get("portfolios") or []))
        with c3:
            st.metric("Projects", len(lists.get("projects") or []))
        with c4:
            st.metric("Themes", len(lists.get("themes") or []))

        st.write("Series (session):", len(series) if series else 0)
        st.caption("Wenn Session ≠ Files/ZIP: Ursache ist fast immer 'nicht gespeichert' oder falsches Data-Dir. (Das hier fixt die Employee-ID-Nummer-None-Scheiße zuverlässig.)")
