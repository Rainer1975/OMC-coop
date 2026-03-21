from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from core import new_series

__version__ = "2026.03.21.5"

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role_at_write TEXT NOT NULL,
    created_at TEXT NOT NULL,
    work_date TEXT NOT NULL,
    project TEXT,
    title TEXT NOT NULL,
    status TEXT,
    priority TEXT,
    next_step TEXT,
    blockers TEXT,
    due_hint TEXT,
    notes TEXT,
    source_text TEXT,
    entry_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""

FIELDS = [
    ("project", "Wie heißt das Projekt?"),
    ("work_item", "Woran arbeitest du daran konkret?"),
    ("status", "Wie ist der aktuelle Status?"),
    ("next_step", "Was ist der nächste konkrete Schritt?"),
    ("blockers", "Gibt es Blocker oder Abhängigkeiten? Wenn nein, sag einfach 'keine'."),
    ("entry_kind", "Soll ich das als Status, als Aufgabe oder als beides behandeln?"),
    ("priority", "Welche Priorität hat das? Niedrig, Mittel oder Hoch?"),
    ("due_hint", "Gibt es einen Termin oder zeitlichen Hinweis?"),
    ("notes", "Gibt es noch Zusatznotizen, die ich mitnehmen soll? Wenn nicht, sag 'keine'."),
]

SUMMARY_COMMANDS = {"zeige zusammenfassung", "zusammenfassung", "anzeigen", "prüfen", "check"}
SAVE_COMMANDS = {"speichern", "übernehmen", "save"}
CANCEL_COMMANDS = {"abbrechen", "neu starten", "reset"}
NONE_WORDS = {"keine", "nein", "nichts", "n/a", "-", ""}


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _low(x: Any) -> str:
    return _norm(x).lower()


def _slugify(s: str) -> str:
    s = _low(s)
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else ".")
    v = "".join(out).strip(".")
    while ".." in v:
        v = v.replace("..", ".")
    return v or "user"


def _db_path(ctx: Dict[str, Any]) -> Path:
    return Path(str(ctx.get("AGENT_DB_FILE") or "agent_intake.db"))


def _db(ctx: Dict[str, Any]) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(ctx))
    conn.row_factory = sqlite3.Row
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


def _employees(ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for e in (ctx.get("employees") or []):
        if not isinstance(e, dict):
            continue
        dn = _norm(e.get("display_name"))
        if not dn:
            continue
        uid = _norm(e.get("id")) or _slugify(dn)
        out.append({"username": uid.lower(), "display_name": dn})
    return out


def _is_admin_username(username: str) -> bool:
    u = _low(username)
    return u == "sebastian" or u.startswith("sebastian.") or ".sebastian" in u


def _seed_users(ctx: Dict[str, Any]) -> None:
    conn = _db(ctx)
    employees = _employees(ctx)
    now = datetime.now().isoformat(timespec="seconds")
    with conn:
        for e in employees:
            role = "admin" if _is_admin_username(e["username"]) or _is_admin_username(e["display_name"]) else "employee"
            conn.execute(
                """
                INSERT INTO users(username, display_name, role, created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(username) DO UPDATE SET display_name=excluded.display_name, role=excluded.role
                """,
                (e["username"], e["display_name"], role, now),
            )
    conn.close()


def _user_record(ctx: Dict[str, Any], username: str) -> Dict[str, Any]:
    _seed_users(ctx)
    u = _low(username)
    if not u:
        return {}
    conn = _db(ctx)
    row = conn.execute("SELECT * FROM users WHERE username = ?", (u,)).fetchone()
    if row is None:
        role = "admin" if _is_admin_username(u) else "employee"
        display = username.strip() or u
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT OR REPLACE INTO users(username, display_name, role, created_at) VALUES(?,?,?,?)",
            (u, display, role, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (u,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def _current_user() -> Dict[str, Any]:
    return st.session_state.get("agent_auth_profile") or {}


def _can_read(user: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    if not user:
        return False
    if _low(user.get("role")) == "admin":
        return True
    return _low(user.get("username")) == _low(entry.get("username"))


def _visible_entries(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    user = _current_user()
    if not user:
        return []
    conn = _db(ctx)
    rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC, id DESC").fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["payload_json"] = json.loads(d.get("payload_json") or "{}")
        except Exception:
            d["payload_json"] = {}
        if _can_read(user, d):
            out.append(d)
    return out


def _visible_series(ctx: Dict[str, Any]) -> List[Any]:
    user = _current_user()
    if not user:
        return []
    series = list(st.session_state.get("series") or [])
    if _low(user.get("role")) == "admin":
        return series
    uname = _low(user.get("username"))
    return [s for s in series if _low(getattr(s, "owner_id", "")) == uname]


def _init_state() -> None:
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("agent_draft", {})
    st.session_state.setdefault("agent_mode", "idle")
    st.session_state.setdefault("agent_show_summary", False)
    st.session_state.setdefault("agent_last_search", None)
    st.session_state.setdefault("agent_last_saved", None)
    st.session_state.setdefault("agent_waiting_field", None)
    if not st.session_state["agent_messages"]:
        st.session_state["agent_messages"] = [
            {
                "role": "assistant",
                "content": "Gib deine Projektinformationen ein. Starte mit `Coop-Eingabe ...` oder suche mit `Hey Projekt ...`.",
            }
        ]


def _reset_dialog() -> None:
    st.session_state["agent_messages"] = [
        {
            "role": "assistant",
            "content": "Gib deine Projektinformationen ein. Starte mit `Coop-Eingabe ...` oder suche mit `Hey Projekt ...`.",
        }
    ]
    st.session_state["agent_draft"] = {}
    st.session_state["agent_mode"] = "idle"
    st.session_state["agent_show_summary"] = False
    st.session_state["agent_last_search"] = None
    st.session_state["agent_last_saved"] = None
    st.session_state["agent_waiting_field"] = None


def _append(role: str, content: str) -> None:
    st.session_state.agent_messages.append({"role": role, "content": content})


def _clean_value(value: str) -> str:
    v = _norm(value)
    if _low(v) in NONE_WORDS:
        return ""
    return v


def _project_options(ctx: Dict[str, Any]) -> List[str]:
    projects = []
    for p in ((ctx.get("lists") or {}).get("projects") or []):
        if _norm(p):
            projects.append(_norm(p))
    for e in _visible_entries(ctx):
        if _norm(e.get("project")):
            projects.append(_norm(e.get("project")))
    uniq = []
    seen = set()
    for p in projects:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _infer_from_text(text: str, ctx: Dict[str, Any]) -> Dict[str, str]:
    t = _norm(text)
    tl = t.lower()
    out: Dict[str, str] = {}
    if not t:
        return out

    for p in _project_options(ctx):
        if p.lower() in tl:
            out["project"] = p
            break
    m = re.search(r"projekt\s+([A-Za-z0-9ÄÖÜäöüß\-_ ]{2,60})", t, flags=re.IGNORECASE)
    if m and not out.get("project"):
        out["project"] = _norm(m.group(1))

    if any(w in tl for w in ["hoch", "dringend", "kritisch", "p1", "prio 1"]):
        out["priority"] = "Hoch"
    elif any(w in tl for w in ["mittel", "normal", "p2", "prio 2"]):
        out["priority"] = "Mittel"
    elif any(w in tl for w in ["niedrig", "später", "p3", "prio 3"]):
        out["priority"] = "Niedrig"

    if "beides" in tl or ("status" in tl and "aufgabe" in tl):
        out["entry_kind"] = "Beides"
    elif "aufgabe" in tl or "ticket" in tl or "anlegen" in tl:
        out["entry_kind"] = "Aufgabe"
    elif "status" in tl or "update" in tl:
        out["entry_kind"] = "Status"

    if any(w in tl for w in ["heute", "morgen", "woche", "deadline", "termin", "bis "]):
        out["due_hint"] = t
    if any(w in tl for w in ["blocker", "abhängig", "warte auf", "risiko", "hindernis"]):
        out["blockers"] = t
    if any(w in tl for w in ["als nächstes", "nächster schritt", "danach", "weiter mit"]):
        out["next_step"] = t
    if any(w in tl for w in ["in arbeit", "offen", "fertig", "review", "warte", "läuft", "status"]):
        out["status"] = t
    out.setdefault("work_item", t)
    return out


def _missing_fields(draft: Dict[str, str]) -> List[str]:
    return [key for key, _question in FIELDS if not _norm(draft.get(key))]


def _question_for(field: str) -> str:
    for key, question in FIELDS:
        if key == field:
            return question
    return "Bitte ergänze die fehlende Information."


def _normalize_field_value(field: str, value: str) -> str:
    v = _clean_value(value)
    if not v:
        return ""
    if field == "priority":
        vl = v.lower()
        if "hoch" in vl or "dring" in vl or "krit" in vl:
            return "Hoch"
        if "nied" in vl or "spät" in vl:
            return "Niedrig"
        return "Mittel"
    if field == "entry_kind":
        vl = v.lower()
        if "beid" in vl or ("status" in vl and "aufgabe" in vl):
            return "Beides"
        if "aufgabe" in vl or "ticket" in vl:
            return "Aufgabe"
        return "Status"
    return v


def _show_summary() -> None:
    draft = st.session_state.agent_draft
    if not draft:
        _append("assistant", "Es gibt noch keine erfassten Daten. Starte mit `Coop-Eingabe ...`.")
        return
    st.session_state.agent_show_summary = True
    _append("assistant", "Ich blende die Zusammenfassung ein.")


def _ensure_project_in_lists(ctx: Dict[str, Any], project: str) -> None:
    p = _norm(project)
    if not p:
        return
    lists = ctx.get("lists") or {}
    projects = list(lists.get("projects") or [])
    if p not in projects:
        projects.append(p)
        projects = sorted(set(projects), key=lambda x: x.lower())
        lists["projects"] = projects
        save_lists = ctx.get("save_lists")
        if callable(save_lists):
            save_lists(lists.get("portfolios", []), lists.get("projects", []), lists.get("themes", []))


def _save_to_tool(ctx: Dict[str, Any], payload: Dict[str, str], entry_id: int) -> Optional[str]:
    if payload.get("entry_kind") not in {"Aufgabe", "Beides"}:
        return None
    user = _current_user()
    today = ctx.get("today") or date.today()
    due_text = _low(payload.get("due_hint"))
    end = today
    if "morgen" in due_text:
        end = today + timedelta(days=1)
    elif "woche" in due_text:
        end = today + timedelta(days=5)
    elif "heute" in due_text:
        end = today
    title = _norm(payload.get("work_item")) or _norm(payload.get("project")) or "Neuer Eintrag"
    s = new_series(
        title=title,
        portfolio="Default",
        project=_norm(payload.get("project")),
        theme="General",
        owner=_norm(user.get("display_name")) or _norm(user.get("username")),
        owner_id=_norm(user.get("username")),
        start=today,
        end=end,
        is_meta=False,
        kind="task",
        state="ACTIVE",
    )
    s.meta = {
        "agent_entry_id": entry_id,
        "source": "Coop-Eingabe",
        "status": payload.get("status", ""),
        "next_step": payload.get("next_step", ""),
        "blockers": payload.get("blockers", ""),
        "priority": payload.get("priority", ""),
        "due_hint": payload.get("due_hint", ""),
        "notes": payload.get("notes", ""),
    }
    series = list(st.session_state.get("series") or [])
    series.append(s)
    save_series = ctx.get("save_series")
    if callable(save_series):
        save_series(series)
    sync_lists = ctx.get("sync_lists_from_data")
    if callable(sync_lists):
        sync_lists()
    return getattr(s, "series_id", None)


def _store_entry(ctx: Dict[str, Any], draft: Dict[str, str]) -> int:
    user = _current_user()
    payload = {
        "project": _norm(draft.get("project")),
        "title": (_norm(draft.get("work_item")) or "Projektupdate")[:160],
        "status": _norm(draft.get("status")),
        "priority": _norm(draft.get("priority")),
        "next_step": _norm(draft.get("next_step")),
        "blockers": _norm(draft.get("blockers")),
        "due_hint": _norm(draft.get("due_hint")),
        "notes": _norm(draft.get("notes")),
        "source_text": _norm(draft.get("source_text")),
        "entry_kind": _norm(draft.get("entry_kind")) or "Status",
        "work_date": date.today().isoformat(),
    }
    conn = _db(ctx)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO entries(
                username, display_name, role_at_write, created_at, work_date,
                project, title, status, priority, next_step, blockers, due_hint,
                notes, source_text, entry_kind, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                _norm(user.get("username")),
                _norm(user.get("display_name")) or _norm(user.get("username")),
                _norm(user.get("role")) or "employee",
                datetime.now().isoformat(timespec="seconds"),
                payload["work_date"],
                payload["project"],
                payload["title"],
                payload["status"],
                payload["priority"],
                payload["next_step"],
                payload["blockers"],
                payload["due_hint"],
                payload["notes"],
                payload["source_text"],
                payload["entry_kind"],
                json.dumps(payload, ensure_ascii=False),
            ),
        )
    entry_id = int(cur.lastrowid)
    conn.close()
    _ensure_project_in_lists(ctx, payload["project"])
    task_id = _save_to_tool(ctx, payload, entry_id)
    st.session_state.agent_last_saved = {"entry_id": entry_id, "task_id": task_id}
    return entry_id


def _render_summary_box() -> None:
    if not st.session_state.get("agent_show_summary"):
        return
    draft = st.session_state.get("agent_draft") or {}
    if not draft:
        return
    with st.container(border=True):
        st.markdown("### Zusammenfassung")
        rows = [
            ("Projekt", draft.get("project")),
            ("Arbeitsinhalt", draft.get("work_item")),
            ("Status", draft.get("status")),
            ("Nächster Schritt", draft.get("next_step")),
            ("Blocker", draft.get("blockers")),
            ("Art des Eintrags", draft.get("entry_kind")),
            ("Priorität", draft.get("priority")),
            ("Termin", draft.get("due_hint")),
            ("Notizen", draft.get("notes")),
        ]
        for label, value in rows:
            st.markdown(f"**{label}:** {_norm(value) or '—'}")


def _render_search_result(ctx: Dict[str, Any]) -> None:
    payload = st.session_state.get("agent_last_search")
    if not payload:
        return
    query = payload.get("query") or ""
    db_hits = payload.get("db_hits") or []
    task_hits = payload.get("task_hits") or []
    with st.container(border=True):
        st.markdown(f"### Projektsicht: {query}")
        if not db_hits and not task_hits:
            st.info("Keine sichtbaren Informationen gefunden.")
            return
        if db_hits:
            st.markdown("**Gespeicherte Eingaben**")
            for hit in db_hits[:10]:
                st.markdown(
                    f"- **{_norm(hit.get('project')) or 'Ohne Projekt'}** · {_norm(hit.get('title'))} · {_norm(hit.get('status')) or 'ohne Status'}"
                )
        if task_hits:
            st.markdown("**Einträge im Tool**")
            for s in task_hits[:10]:
                st.markdown(
                    f"- **{_norm(getattr(s, 'project', '')) or 'Ohne Projekt'}** · {_norm(getattr(s, 'title', ''))} · {_norm(getattr(s, 'state', ''))}"
                )


def _search_project(ctx: Dict[str, Any], query: str) -> None:
    q = _norm(query)
    ql = q.lower()
    db_hits = []
    for e in _visible_entries(ctx):
        hay = " ".join(
            [
                _norm(e.get("project")),
                _norm(e.get("title")),
                _norm(e.get("status")),
                _norm(e.get("next_step")),
                _norm(e.get("blockers")),
                _norm(e.get("notes")),
                _norm(e.get("source_text")),
            ]
        ).lower()
        if ql in hay:
            db_hits.append(e)
    task_hits = []
    for s in _visible_series(ctx):
        hay = " ".join(
            [
                _norm(getattr(s, "project", "")),
                _norm(getattr(s, "title", "")),
                _norm(getattr(s, "state", "")),
                _norm(getattr(s, "owner", "")),
            ]
        ).lower()
        if ql in hay:
            task_hits.append(s)
    st.session_state.agent_last_search = {"query": q, "db_hits": db_hits, "task_hits": task_hits}
    _append("assistant", f"Ich habe nach `{q}` gesucht. Die sichtbaren Treffer blende ich unten ein.")


def _start_capture(ctx: Dict[str, Any], text: str) -> None:
    st.session_state.agent_mode = "capture"
    st.session_state.agent_show_summary = False
    st.session_state.agent_last_search = None
    draft = {"source_text": _norm(text)}
    inferred = _infer_from_text(text, ctx)
    for k, v in inferred.items():
        if _norm(v):
            draft[k] = v
    st.session_state.agent_draft = draft
    missing = _missing_fields(draft)
    if missing:
        st.session_state.agent_waiting_field = missing[0]
        _append("assistant", _question_for(missing[0]))
    else:
        st.session_state.agent_waiting_field = None
        _append("assistant", "Ich habe alle Felder erfasst. Sage `zeige zusammenfassung` oder direkt `speichern`.")


def _continue_capture(ctx: Dict[str, Any], text: str) -> None:
    draft = dict(st.session_state.get("agent_draft") or {})
    field = st.session_state.get("agent_waiting_field")
    if field:
        draft[field] = _normalize_field_value(field, text)
    else:
        inferred = _infer_from_text(text, ctx)
        for k, v in inferred.items():
            if _norm(v) and not _norm(draft.get(k)):
                draft[k] = v
    st.session_state.agent_draft = draft
    missing = _missing_fields(draft)
    if missing:
        st.session_state.agent_waiting_field = missing[0]
        _append("assistant", _question_for(missing[0]))
    else:
        st.session_state.agent_waiting_field = None
        _append("assistant", "Ich habe alles. Sage `zeige zusammenfassung` oder `speichern`.")


def _handle_prompt(ctx: Dict[str, Any], prompt: str) -> None:
    text = _norm(prompt)
    if not text:
        return
    _append("user", text)
    tl = text.lower()

    if tl in CANCEL_COMMANDS:
        _reset_dialog()
        return

    if tl in SUMMARY_COMMANDS:
        _show_summary()
        return

    if tl in SAVE_COMMANDS:
        draft = st.session_state.get("agent_draft") or {}
        missing = _missing_fields(draft)
        if missing:
            st.session_state.agent_mode = "capture"
            st.session_state.agent_waiting_field = missing[0]
            _append("assistant", f"Zum Speichern fehlt noch etwas. {_question_for(missing[0])}")
            return
        entry_id = _store_entry(ctx, draft)
        st.session_state.agent_mode = "idle"
        st.session_state.agent_show_summary = False
        st.session_state.agent_waiting_field = None
        _append("assistant", f"Gespeichert. Eintrag #{entry_id} ist in der separaten Datenbank verfügbar. Mit `Hey Projekt {draft.get('project','')}` kannst du ihn wieder finden.")
        st.session_state.agent_draft = {}
        return

    if tl.startswith("hey projekt"):
        query = _norm(text[11:])
        if not query:
            _append("assistant", "Nenne nach `Hey Projekt` bitte den Projektnamen oder Suchbegriff.")
            return
        st.session_state.agent_mode = "search"
        _search_project(ctx, query)
        return

    if tl.startswith("coop-eingabe"):
        body = _norm(text[len("coop-eingabe"):])
        _start_capture(ctx, body)
        return

    if st.session_state.get("agent_mode") == "capture":
        _continue_capture(ctx, text)
        return

    _append("assistant", "Ich habe den Befehl nicht als Intake erkannt. Starte mit `Coop-Eingabe ...` oder suche mit `Hey Projekt ...`.")


def _render_auth(ctx: Dict[str, Any]) -> None:
    _seed_users(ctx)
    st.sidebar.subheader("Login")
    users = _employees(ctx)
    options = [u["display_name"] for u in users]
    display_to_username = {u["display_name"]: u["username"] for u in users}
    picked = st.sidebar.selectbox("Bekannte Nutzer", ["Manuell eingeben"] + options, key="agent_login_pick")
    default = display_to_username.get(picked, st.session_state.get("agent_login_username", ""))
    username = st.sidebar.text_input("Benutzername", value=default, key="agent_login_username")
    cols = st.sidebar.columns(2)
    if cols[0].button("Anmelden", use_container_width=True):
        profile = _user_record(ctx, username)
        st.session_state.agent_auth_profile = profile
        _reset_dialog()
        st.rerun()
    if cols[1].button("Abmelden", use_container_width=True):
        st.session_state.agent_auth_profile = None
        _reset_dialog()
        st.rerun()
    profile = _current_user()
    if profile:
        label = "Admin" if _low(profile.get("role")) == "admin" else "Mitarbeiter:in"
        st.sidebar.success(f"{_norm(profile.get('display_name'))} · {label}")
        if not users:
            st.sidebar.caption("Im ZIP sind keine gepflegten Nutzer hinterlegt. Für den Test wird der eingegebene Benutzername verwendet. Sebastian erhält Admin-Rechte.")
    else:
        st.sidebar.info("Anmeldung per Benutzername. Sebastian hat Admin-Rechte, alle anderen sind gleichberechtigt.")


def render(ctx: Dict[str, Any]) -> None:
    _init_state()
    _render_auth(ctx)

    st.markdown(
        """
        <style>
        .agent-title {text-align:center; margin-top: 4vh; margin-bottom: 1rem;}
        .agent-sub {text-align:center; color:#6b7280; margin-bottom: 2rem;}
        div[data-testid="stChatMessage"] {max-width: 900px; margin-left: auto; margin-right: auto;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="agent-title"><h1>Coop Agent</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="agent-sub">Einfacher Intake in natürlicher Sprache. Start mit <code>Coop-Eingabe ...</code> oder Suche mit <code>Hey Projekt ...</code>.</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    _render_summary_box()
    _render_search_result(ctx)

    if st.session_state.get("agent_last_saved"):
        info = st.session_state.agent_last_saved
        with st.container(border=True):
            st.markdown("### Letzter Schreibvorgang")
            st.markdown(f"**Datenbank-Eintrag:** {info.get('entry_id')}")
            st.markdown(f"**Tool-Eintrag:** {_norm(info.get('task_id')) or 'kein Aufgabenobjekt erzeugt'}")

    disabled = not bool(_current_user())
    placeholder = "Gib deine Projektinformationen ein …" if not disabled else "Zuerst links mit Benutzername anmelden …"
    prompt = st.chat_input(placeholder, disabled=disabled)
    if prompt:
        _handle_prompt(ctx, prompt)
        st.rerun()
