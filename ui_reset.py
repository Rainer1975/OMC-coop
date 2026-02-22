# ui_reset.py
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def render(ctx: dict) -> None:
    st.header("🧨 Global Reset")

    data_file: Path = ctx["DATA_FILE"]
    emp_file: Path = ctx["EMP_FILE"]
    lists_file: Path = ctx["LISTS_FILE"]

    st.error(
        "Das löscht **ALLES** (unwiderruflich):\n\n"
        "- Tasks & Termine (`data.json`)\n"
        "- Employees (`employees.json`)\n"
        "- Portfolios/Projects/Themes (`lists.json`)\n"
        "- Session-State (alles im Speicher)\n"
    )

    # Statusanzeige (was ist aktuell drin?)
    try:
        cur_series = st.session_state.get("series", []) or []
        cur_emps = st.session_state.get("employees", []) or []
        cur_lists = st.session_state.get("lists", {}) or {}
        st.caption(
            f"Aktueller Stand (Session): "
            f"Series={len(cur_series)} | Employees={len(cur_emps)} | "
            f"Portfolios={len(cur_lists.get('portfolios', []))} | "
            f"Projects={len(cur_lists.get('projects', []))} | "
            f"Themes={len(cur_lists.get('themes', []))}"
        )
    except Exception:
        pass

    st.divider()

    # 2-Step Confirm (damit man nicht versehentlich klickt)
    c1 = st.checkbox("Ich verstehe: **ALLE Daten werden gelöscht**", key="reset_confirm_1")
    c2 = st.checkbox("Ja, wirklich löschen (zweite Bestätigung)", key="reset_confirm_2", disabled=not c1)

    colA, colB = st.columns([2, 6])
    with colA:
        do_reset = st.button(
            "🧨 GLOBAL RESET",
            type="primary",
            disabled=not (c1 and c2),
            use_container_width=True,
            key="reset_do_it",
        )

    with colB:
        st.info(
            "Hinweis: In Streamlit Cloud sind Daten flüchtig. "
            "Dieser Reset setzt alles absichtlich auf **leer** zurück."
        )

    if not do_reset:
        return

    # 1) Dateien auf leer setzen (nicht nur löschen -> stabiler)
    try:
        _write_json(data_file, {"series": []})
    except Exception as e:
        st.error(f"Fehler beim Schreiben von data.json: {e}")
        return

    try:
        _write_json(emp_file, {"schema_version": 1, "employees": []})
    except Exception as e:
        st.error(f"Fehler beim Schreiben von employees.json: {e}")
        return

    try:
        _write_json(lists_file, {"portfolios": ["Default"], "projects": [], "themes": []})
    except Exception as e:
        st.error(f"Fehler beim Schreiben von lists.json: {e}")
        return

    # 2) Session State resetten (damit UI sofort leer ist)
    st.session_state.series = []
    st.session_state.employees = []
    st.session_state.lists = {"portfolios": ["Default"], "projects": [], "themes": []}

    # 3) Optional: persist() falls vorhanden
    try:
        if "persist" in ctx:
            ctx["persist"]()
    except Exception:
        pass

    st.success("GLOBAL RESET durchgeführt. App lädt neu …")
    st.rerun()
