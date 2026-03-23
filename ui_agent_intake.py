from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from core import new_series

__version__ = "2026.03.23.19"

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

FIELDS: List[Tuple[str, str]] = [
    ("project", "Wie heißt das Projekt genau?"),
    ("work_item", "Woran arbeitest du konkret?"),
    ("status", "Wie ist der aktuelle Status?"),
    ("next_step", "Was ist der nächste konkrete Schritt?"),
    ("blockers", "Gibt es Blocker oder Abhängigkeiten? Wenn nein, sag einfach 'keine'."),
    ("entry_kind", "Soll ich das als Status, als Aufgabe oder als beides behandeln?"),
    ("priority", "Welche Priorität hat das? Niedrig, Mittel oder Hoch?"),
    ("due_hint", "Gibt es einen Termin oder zeitlichen Hinweis? Wenn nein, sag 'keine'."),
    ("notes", "Gibt es noch Zusatznotizen? Wenn nein, sag 'keine'."),
]

FIELD_LABELS = {
    "project": "Projekt",
    "work_item": "Arbeitsinhalte",
    "status": "Status",
    "next_step": "Nächster Schritt",
    "blockers": "Blocker/Abhängigkeiten",
    "entry_kind": "Art des Eintrags",
    "priority": "Priorität",
    "due_hint": "Termin",
    "notes": "Notizen",
}

SUMMARY_COMMANDS = {"zusammenfassung", "zeige zusammenfassung"}
SAVE_COMMANDS = {"speichern", "übernehmen", "save"}
BACK_COMMANDS = {"zurück", "zurueck", "back"}
CORRECT_COMMANDS = {"korrigieren", "ändern", "aendern", "edit"}
RESTART_COMMANDS = {"neu starten", "restart", "abbrechen", "reset", "neu"}
NONE_WORDS = {"keine", "kein", "nein", "nichts", "keine blocker", "keine abhängigkeiten", "keine abhaengigkeiten", "-", ""}
VOICE_PROMPT = (
    "Das Audio enthält ein deutschsprachiges Projekt-Update für ein Projekt-Intake-Tool. "
    "Schreibe den gesprochenen Inhalt sauber auf Deutsch mit Satzzeichen. Bewahre Projektnamen, Zahlen und konkrete nächste Schritte."
)


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


def _api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def _openai_available() -> bool:
    return bool(_api_key())


def _init_state() -> None:
    defaults = {
        "agent_messages": [],
        "agent_draft": {},
        "agent_mode": "home",
        "agent_login_username": "",
        "agent_help_open": False,
        "agent_prompt_value": "",
        "agent_prompt_rev": 0,
        "agent_voice_open": False,
        "agent_voice_text_value": "",
        "agent_voice_error": "",
        "agent_voice_status": "idle",
        "agent_audio_digest": "",
        "agent_voice_take": 0,
        "agent_voice_diag": {},
        "agent_voice_debug": True,
        "agent_current_field": None,
        "agent_confirm_field": None,
        "agent_confirm_value": None,
        "agent_summary_offer": False,
        "agent_save_offer": False,
        "agent_last_search": None,
        "agent_last_saved": None,
        "agent_menu": "home",
        "agent_history_fields": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _append(role: str, content: str) -> None:
    st.session_state.agent_messages.append({"role": role, "content": content})


def _clear_prompt() -> None:
    st.session_state["agent_prompt_value"] = ""
    st.session_state["agent_prompt_rev"] = int(st.session_state.get("agent_prompt_rev", 0)) + 1


def _voice_reset(increment_take: bool = False) -> None:
    st.session_state["agent_voice_text_value"] = ""
    st.session_state["agent_voice_error"] = ""
    st.session_state["agent_voice_status"] = "idle"
    st.session_state["agent_audio_digest"] = ""
    st.session_state["agent_voice_diag"] = {}
    if increment_take:
        st.session_state["agent_voice_take"] = int(st.session_state.get("agent_voice_take", 0)) + 1


def _reset_dialog(keep_user: bool = True) -> None:
    profile = st.session_state.get("agent_auth_profile") if keep_user else None
    # Do NOT write back to widget-bound keys like agent_login_username here.
    # Streamlit forbids mutating them after instantiation in the same run.
    for key in [
        "agent_messages", "agent_draft", "agent_mode", "agent_help_open", "agent_prompt_value", "agent_prompt_rev",
        "agent_voice_open", "agent_voice_text_value", "agent_voice_error", "agent_voice_status", "agent_audio_digest",
        "agent_voice_take", "agent_voice_diag", "agent_current_field", "agent_confirm_field", "agent_confirm_value",
        "agent_summary_offer", "agent_save_offer", "agent_last_search", "agent_last_saved", "agent_menu",
        "agent_history_fields"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    _init_state()
    st.session_state["agent_auth_profile"] = profile


def _question_for(field: str) -> str:
    for key, question in FIELDS:
        if key == field:
            return question
    return "Bitte ergänze die Information."


def _field_order() -> List[str]:
    return [k for k, _ in FIELDS]


def _missing_fields(draft: Dict[str, Any]) -> List[str]:
    out = []
    for field in _field_order():
        if not _norm(draft.get(field)):
            out.append(field)
    return out


def _clean_value(value: str) -> str:
    v = _norm(value)
    return "" if _low(v) in NONE_WORDS else v


def _normalize_field_value(field: str, value: str) -> str:
    v = _clean_value(value)
    if field == "priority":
        vl = _low(v)
        if not vl:
            return ""
        if any(x in vl for x in ["hoch", "krit", "dring", "p1"]):
            return "Hoch"
        if any(x in vl for x in ["nied", "spät", "spaet", "p3"]):
            return "Niedrig"
        return "Mittel"
    if field == "entry_kind":
        vl = _low(v)
        if not vl:
            return ""
        if "beid" in vl or ("status" in vl and "aufgabe" in vl):
            return "Beides"
        if "aufgabe" in vl or "ticket" in vl or "anlegen" in vl:
            return "Aufgabe"
        return "Status"
    if field == "blockers" and _low(value) in NONE_WORDS:
        return ""
    if field == "due_hint" and _low(value) in NONE_WORDS:
        return ""
    if field == "notes" and _low(value) in NONE_WORDS:
        return ""
    return v




def _menu_steps() -> List[Tuple[str, str]]:
    return [
        ("home", "1. Hauptmenü"),
        ("capture", "2. Neue Eingabe"),
        ("summary", "3. Zusammenfassung prüfen"),
        ("search", "4. Projektinfos suchen"),
    ]


def _menu_guide_text() -> str:
    mode = st.session_state.get("agent_mode") or "home"
    current_field = st.session_state.get("agent_current_field")
    confirm_field = st.session_state.get("agent_confirm_field")
    if not _current_user():
        return "Melde dich zuerst mit deinem Benutzernamen an."
    if mode != "capture":
        return "Starte im Hauptmenü mit **Neue Eingabe** oder sage direkt `Coop-Eingabe ...`."
    if confirm_field:
        return f"Bestätige jetzt **{FIELD_LABELS.get(confirm_field, confirm_field)}** per Ja/Nein oder korrigiere den Wert."
    if current_field:
        return f"Du bist in **{FIELD_LABELS.get(current_field, current_field)}**. Antworte mündlich oder schriftlich in einem ganzen Satz."
    if st.session_state.get("agent_summary_offer"):
        return "Die KI wartet auf deine Entscheidung zur **ZUSAMMENFASSUNG**."
    if st.session_state.get("agent_save_offer"):
        return "Die KI wartet auf deine Entscheidung zum **Speichern**."
    return "Die Eingabe läuft. Du kannst jeden Unterpunkt auch direkt über das Menü anwählen."


def _latest_user_utterance() -> str:
    msgs = st.session_state.get("agent_messages") or []
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            return _norm(msg.get("content"))
    return ""


def _render_menu_guide() -> None:
    mode = st.session_state.get("agent_mode") or "home"
    labels = dict(_menu_steps())
    with st.container(border=True):
        st.markdown("### Menüführung")
        cols = st.columns(len(labels))
        for i, (key, label) in enumerate(labels.items()):
            prefix = "➡️ " if ((key == "home" and mode != "capture" and st.session_state.get("agent_menu") == "home") or key == st.session_state.get("agent_menu") or (key == "capture" and mode == "capture")) else ""
            cols[i].markdown(f"{prefix}{label}")
        st.info(_menu_guide_text())


def _render_dialog_views() -> None:
    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        with st.container(border=True):
            st.markdown("### Sichtbarer Dialog")
            user_sentence = _latest_user_utterance()
            if user_sentence:
                st.markdown(f"**Dein letzter Satz:** {user_sentence}")
            else:
                st.caption("Hier erscheint dein letzter gesprochener oder geschriebener Satz.")
            msgs = st.session_state.get("agent_messages") or []
            for msg in msgs[-6:]:
                speaker = "Du" if msg.get("role") == "user" else "KI"
                st.markdown(f"**{speaker}:** {msg.get('content','')}")
    with col2:
        with st.container(border=True):
            st.markdown("### Transkript")
            vt = _norm(st.session_state.get("agent_voice_text_value"))
            if vt:
                st.write(vt)
            else:
                st.caption("Nach der Sprachaufnahme erscheint hier das Transkript.")
            draft = st.session_state.get("agent_draft") or {}
            if draft:
                st.markdown("### Bisher notiert")
                for field in _field_order():
                    val = _norm(draft.get(field))
                    if val:
                        st.markdown(f"- **{FIELD_LABELS.get(field, field)}:** {val}")

def _project_options(ctx: Dict[str, Any]) -> List[str]:
    projects = []
    for p in ((ctx.get("lists") or {}).get("projects") or []):
        if _norm(p):
            projects.append(_norm(p))
    for e in _visible_entries(ctx):
        if _norm(e.get("project")):
            projects.append(_norm(e.get("project")))
    uniq, seen = [], set()
    for p in projects:
        key = p.lower()
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _infer_from_text(text: str, ctx: Dict[str, Any], draft: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    t = _norm(text)
    tl = _low(text)
    out: Dict[str, str] = {}
    draft = draft or {}
    if not t:
        return out

    for p in _project_options(ctx):
        if p.lower() in tl:
            out["project"] = p
            break
    m = re.search(r"(?:projekt|für|fuer)\s+([A-Za-z0-9ÄÖÜäöüß\-_/ ]{2,60})", t, flags=re.IGNORECASE)
    if m and not out.get("project"):
        out["project"] = _norm(m.group(1))

    if any(w in tl for w in ["hoch", "dringend", "kritisch", "p1", "prio 1"]):
        out["priority"] = "Hoch"
    elif any(w in tl for w in ["mittel", "normal", "p2", "prio 2"]):
        out["priority"] = "Mittel"
    elif any(w in tl for w in ["niedrig", "später", "spaeter", "p3", "prio 3"]):
        out["priority"] = "Niedrig"

    if "beides" in tl or ("status" in tl and "aufgabe" in tl):
        out["entry_kind"] = "Beides"
    elif any(w in tl for w in ["aufgabe", "ticket", "anlegen"]):
        out["entry_kind"] = "Aufgabe"
    elif any(w in tl for w in ["status", "update"]):
        out["entry_kind"] = "Status"

    if re.search(r"\b(in arbeit|läuft|laeuft|offen|fertig|done|review|wartet|warte|gestartet)\b", tl):
        out["status"] = t
    if re.search(r"\b(als nächstes|als naechstes|nächster schritt|naechster schritt|weiter mit|danach)\b", tl):
        out["next_step"] = t
    if re.search(r"\b(blocker|abhäng|abhaeng|warte auf|hindernis|risiko)\b", tl):
        out["blockers"] = t
    if any(w in tl for w in ["heute", "morgen", "deadline", "bis", "diese woche", "nächste woche", "naechste woche", "termin"]):
        out["due_hint"] = t
    if _low(t) in NONE_WORDS:
        if not _norm(draft.get("blockers")):
            out["blockers"] = ""
        elif not _norm(draft.get("due_hint")):
            out["due_hint"] = ""
        elif not _norm(draft.get("notes")):
            out["notes"] = ""

    if not out.get("work_item") and len(t.split()) > 2:
        out["work_item"] = t
    return out


def _set_current_field(field: Optional[str]) -> None:
    st.session_state["agent_current_field"] = field


def _ask_next_field() -> None:
    missing = _missing_fields(st.session_state.get("agent_draft") or {})
    if missing:
        field = missing[0]
        _set_current_field(field)
        _append("assistant", f"Okay. {_question_for(field)}")
        return
    _set_current_field(None)
    st.session_state["agent_summary_offer"] = True
    _append("assistant", "Ich habe alles notiert. Möchtest du die ZUSAMMENFASSUNG ansehen? Ja oder nein?")


def _begin_new_capture(prefill: str = "", ctx: Optional[Dict[str, Any]] = None) -> None:
    st.session_state["agent_mode"] = "capture"
    st.session_state["agent_menu"] = "capture"
    st.session_state["agent_draft"] = {"source_text": _norm(prefill)}
    st.session_state["agent_confirm_field"] = None
    st.session_state["agent_confirm_value"] = None
    st.session_state["agent_summary_offer"] = False
    st.session_state["agent_save_offer"] = False
    st.session_state["agent_last_search"] = None
    st.session_state["agent_last_saved"] = None
    st.session_state["agent_history_fields"] = []
    if prefill and ctx is not None:
        inferred = _infer_from_text(prefill, ctx, st.session_state["agent_draft"])
        for k, v in inferred.items():
            if _norm(v) or k in {"blockers", "notes", "due_hint"}:
                st.session_state["agent_draft"][k] = _normalize_field_value(k, v)
    _append("assistant", "Wir erfassen jetzt dein Projektupdate. Die mündliche Eingabe ist der Hauptweg. Du kannst parallel aber auch Menü und Text nutzen.")
    _ask_next_field()


def _propose_value(field: str, value: str) -> None:
    norm_value = _normalize_field_value(field, value)
    st.session_state["agent_confirm_field"] = field
    st.session_state["agent_confirm_value"] = norm_value
    pretty = norm_value if _norm(norm_value) else "keine"
    _append("assistant", f"Ich habe für **{FIELD_LABELS.get(field, field)}** notiert: **{pretty}**. Stimmt das? Ja oder nein.")


def _accept_confirmed_value() -> None:
    field = st.session_state.get("agent_confirm_field")
    value = st.session_state.get("agent_confirm_value")
    if not field:
        return
    draft = dict(st.session_state.get("agent_draft") or {})
    draft[field] = value
    st.session_state["agent_draft"] = draft
    history = list(st.session_state.get("agent_history_fields") or [])
    history.append(field)
    st.session_state["agent_history_fields"] = history
    st.session_state["agent_confirm_field"] = None
    st.session_state["agent_confirm_value"] = None
    _append("assistant", "Verstanden.")
    _ask_next_field()


def _reject_confirmed_value() -> None:
    field = st.session_state.get("agent_confirm_field")
    st.session_state["agent_confirm_field"] = None
    st.session_state["agent_confirm_value"] = None
    if field:
        _set_current_field(field)
        _append("assistant", f"Okay, dann korrigieren wir **{FIELD_LABELS.get(field, field)}**. {_question_for(field)}")


def _step_back() -> None:
    history = list(st.session_state.get("agent_history_fields") or [])
    draft = dict(st.session_state.get("agent_draft") or {})
    if st.session_state.get("agent_confirm_field"):
        _reject_confirmed_value()
        return
    if not history:
        _append("assistant", "Wir sind schon am Anfang der Eingabe.")
        return
    last = history.pop()
    draft[last] = ""
    st.session_state["agent_draft"] = draft
    st.session_state["agent_history_fields"] = history
    _set_current_field(last)
    st.session_state["agent_summary_offer"] = False
    st.session_state["agent_save_offer"] = False
    _append("assistant", f"Ich bin einen Schritt zurückgegangen. {_question_for(last)}")


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
    st.session_state["agent_last_saved"] = {"entry_id": entry_id, "task_id": task_id}
    return entry_id


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


def _search_project(ctx: Dict[str, Any], query: str) -> None:
    q = _norm(query)
    ql = q.lower()
    db_hits = []
    for e in _visible_entries(ctx):
        hay = " ".join([
            _norm(e.get("project")), _norm(e.get("title")), _norm(e.get("status")), _norm(e.get("next_step")),
            _norm(e.get("blockers")), _norm(e.get("notes")), _norm(e.get("source_text"))
        ]).lower()
        if ql in hay:
            db_hits.append(e)
    task_hits = []
    for s in _visible_series(ctx):
        hay = " ".join([
            _norm(getattr(s, "project", "")), _norm(getattr(s, "title", "")), _norm(getattr(s, "state", "")), _norm(getattr(s, "owner", ""))
        ]).lower()
        if ql in hay:
            task_hits.append(s)
    st.session_state["agent_last_search"] = {"query": q, "db_hits": db_hits, "task_hits": task_hits}
    st.session_state["agent_menu"] = "search"
    _append("assistant", f"Ich habe nach **{q}** gesucht.")


def _show_summary() -> None:
    draft = st.session_state.get("agent_draft") or {}
    if not draft:
        _append("assistant", "Es gibt noch keine erfassten Daten.")
        return
    st.session_state["agent_menu"] = "summary"
    _append("assistant", "Hier ist die ZUSAMMENFASSUNG.")


def _handle_yes_no(answer: str, ctx: Dict[str, Any]) -> bool:
    al = _low(answer)
    yes = al in {"ja", "j", "yes", "ok", "okay", "passt", "stimmt"}
    no = al in {"nein", "n", "no"}
    if st.session_state.get("agent_confirm_field"):
        if yes:
            _accept_confirmed_value()
            return True
        if no:
            _reject_confirmed_value()
            return True
    if st.session_state.get("agent_summary_offer"):
        st.session_state["agent_summary_offer"] = False
        if yes:
            _show_summary()
        st.session_state["agent_save_offer"] = True
        _append("assistant", "Möchtest du den Eintrag jetzt speichern? Ja oder nein. Alternativ kannst du auch SPEICHERN sagen.")
        return True
    if st.session_state.get("agent_save_offer"):
        st.session_state["agent_save_offer"] = False
        if yes:
            entry_id = _store_entry(ctx, st.session_state.get("agent_draft") or {})
            project = _norm((st.session_state.get("agent_draft") or {}).get("project"))
            _append("assistant", f"Gespeichert. Eintrag #{entry_id}. Mit `Hey Projekt {project}` findest du ihn wieder.")
            st.session_state["agent_draft"] = {}
            st.session_state["agent_mode"] = "home"
            st.session_state["agent_menu"] = "home"
        else:
            _append("assistant", "Okay, ich speichere noch nicht. Du kannst korrigieren, ZUSAMMENFASSUNG sagen oder später SPEICHERN.")
        return True
    return False


def _handle_capture_input(ctx: Dict[str, Any], text: str) -> None:
    if _handle_yes_no(text, ctx):
        return
    tl = _low(text)
    if tl in BACK_COMMANDS:
        _step_back()
        return
    if tl in CORRECT_COMMANDS:
        field = st.session_state.get("agent_current_field") or (_missing_fields(st.session_state.get("agent_draft") or {}) or [None])[0]
        if field:
            _append("assistant", f"Okay, wir korrigieren **{FIELD_LABELS.get(field, field)}**. {_question_for(field)}")
        return
    if tl in RESTART_COMMANDS:
        _begin_new_capture(ctx=ctx)
        return
    if tl in SUMMARY_COMMANDS:
        _show_summary()
        return
    if tl in SAVE_COMMANDS:
        missing = _missing_fields(st.session_state.get("agent_draft") or {})
        if missing:
            _append("assistant", f"Zum Speichern fehlt noch etwas. {_question_for(missing[0])}")
            _set_current_field(missing[0])
            return
        st.session_state["agent_save_offer"] = True
        _append("assistant", "Alles da. Möchtest du jetzt speichern? Ja oder nein.")
        return

    current = st.session_state.get("agent_current_field")
    draft = st.session_state.get("agent_draft") or {}
    inferred = _infer_from_text(text, ctx, draft)

    if current and current in inferred:
        _propose_value(current, inferred[current])
        return
    if current:
        _propose_value(current, text)
        return

    missing = _missing_fields(draft)
    if missing:
        _set_current_field(missing[0])
        _propose_value(missing[0], text)
        return
    _append("assistant", "Ich habe alles notiert. Sage ZUSAMMENFASSUNG oder SPEICHERN.")


def _handle_prompt(ctx: Dict[str, Any], prompt: str) -> None:
    text = _norm(prompt)
    if not text:
        return
    _append("user", text)
    tl = _low(text)

    if tl.startswith("hey projekt"):
        query = _norm(text[11:])
        if not query:
            _append("assistant", "Nenne nach `Hey Projekt` bitte den Projektnamen oder Suchbegriff.")
            return
        _search_project(ctx, query)
        return

    if tl.startswith("coop-eingabe"):
        body = _norm(text[len("coop-eingabe"):])
        _begin_new_capture(body, ctx)
        return

    if st.session_state.get("agent_mode") == "capture":
        _handle_capture_input(ctx, text)
        return

    if tl in {"neue eingabe", "neu"}:
        _begin_new_capture(ctx=ctx)
        return
    if tl in {"suche", "projekt suchen"}:
        st.session_state["agent_menu"] = "search"
        _append("assistant", "Okay. Suche mit `Hey Projekt ...` nach bestehenden Informationen.")
        return
    if tl in SUMMARY_COMMANDS:
        _show_summary()
        return
    if tl in {"hilfe"}:
        st.session_state["agent_help_open"] = True
        return

    _append("assistant", "Wähle im Hauptmenü einen Punkt oder starte direkt mit `Coop-Eingabe ...`.")


def _transcribe_audio_bytes(audio_bytes: bytes, mime_type: str = "audio/wav") -> tuple[str, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=_api_key())
    model = "gpt-4o-mini-transcribe"
    suffix = ".wav"
    if "webm" in mime_type:
        suffix = ".webm"
    elif "mpeg" in mime_type or "mp3" in mime_type:
        suffix = ".mp3"
    elif "mp4" in mime_type or "m4a" in mime_type:
        suffix = ".m4a"

    temp_path = None
    diag = {
        "model": model,
        "mime_type": mime_type,
        "audio_bytes": len(audio_bytes or b""),
        "suffix": suffix,
        "request_sent": False,
        "response_type": "",
        "response_preview": "",
    }
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        with open(temp_path, "rb") as audio_file:
            diag["request_sent"] = True
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                prompt=VOICE_PROMPT,
                language="de",
            )
        diag["response_type"] = type(transcript).__name__
        text_value = _norm(getattr(transcript, "text", ""))
        diag["response_preview"] = text_value[:300]
        return text_value, diag
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _maybe_transcribe_audio(audio) -> None:
    diag = {
        "key_present": bool(_api_key()),
        "audio_present": audio is not None,
        "audio_name": getattr(audio, "name", ""),
        "mime_type": getattr(audio, "type", "audio/wav") or "audio/wav",
        "audio_bytes": 0,
        "digest_prefix": "",
        "request_sent": False,
        "model": "gpt-4o-mini-transcribe",
        "response_type": "",
        "response_preview": "",
        "error": "",
        "status": "idle",
    }
    st.session_state["agent_voice_diag"] = diag
    if audio is None:
        return
    audio_bytes = audio.getvalue()
    diag["audio_bytes"] = len(audio_bytes or b"")
    if not audio_bytes:
        diag["error"] = "Audioobjekt vorhanden, aber ohne Daten."
        diag["status"] = "error"
        st.session_state["agent_voice_error"] = diag["error"]
        st.session_state["agent_voice_status"] = "error"
        st.session_state["agent_voice_diag"] = diag
        return
    digest = hashlib.sha256(audio_bytes).hexdigest()
    diag["digest_prefix"] = digest[:12]
    if digest == st.session_state.get("agent_audio_digest"):
        diag["status"] = "cached"
        st.session_state["agent_voice_diag"] = diag
        return
    if not _openai_available():
        diag["error"] = "OPENAI_API_KEY fehlt oder wird nicht geladen."
        diag["status"] = "error"
        st.session_state["agent_voice_error"] = diag["error"]
        st.session_state["agent_voice_status"] = "error"
        st.session_state["agent_voice_diag"] = diag
        return
    try:
        st.session_state["agent_voice_status"] = "transcribing"
        text_value, transcribe_diag = _transcribe_audio_bytes(audio_bytes, diag["mime_type"])
        diag.update(transcribe_diag)
        diag["status"] = "ready"
        st.session_state["agent_voice_text_value"] = text_value
        st.session_state["agent_audio_digest"] = digest
        st.session_state["agent_voice_error"] = ""
        st.session_state["agent_voice_status"] = "ready"
    except Exception as exc:
        diag["error"] = str(exc)
        diag["status"] = "error"
        st.session_state["agent_voice_error"] = f"Transkription fehlgeschlagen: {exc}"
        st.session_state["agent_voice_status"] = "error"
    st.session_state["agent_voice_diag"] = diag


def _render_summary_box() -> None:
    if st.session_state.get("agent_menu") != "summary":
        return
    draft = st.session_state.get("agent_draft") or {}
    with st.container(border=True):
        st.markdown("### ZUSAMMENFASSUNG")
        if not draft:
            st.info("Noch keine Eingabe im Entwurf.")
            return
        for field in _field_order():
            label = FIELD_LABELS.get(field, field)
            value = _norm(draft.get(field)) or "—"
            st.markdown(f"**{label}:** {value}")


def _render_search_result() -> None:
    payload = st.session_state.get("agent_last_search")
    if st.session_state.get("agent_menu") != "search":
        return
    with st.container(border=True):
        st.markdown("### Suche")
        if not payload:
            st.info("Nutze `Hey Projekt ...`, um nach bestehenden Informationen zu suchen.")
            return
        db_hits = payload.get("db_hits") or []
        task_hits = payload.get("task_hits") or []
        if not db_hits and not task_hits:
            st.info("Keine sichtbaren Informationen gefunden.")
            return
        if db_hits:
            st.markdown("**Gespeicherte Eingaben**")
            for hit in db_hits[:10]:
                st.markdown(f"- **{_norm(hit.get('project')) or 'Ohne Projekt'}** · {_norm(hit.get('title'))} · {_norm(hit.get('status')) or 'ohne Status'}")
        if task_hits:
            st.markdown("**Einträge im Tool**")
            for s in task_hits[:10]:
                st.markdown(f"- **{_norm(getattr(s, 'project', '')) or 'Ohne Projekt'}** · {_norm(getattr(s, 'title', ''))} · {_norm(getattr(s, 'state', ''))}")


def _render_help_box() -> None:
    with st.container(border=True):
        st.markdown("### Hilfe")
        st.info(
            "\n".join([
                "Sprachmodus ist der Primärmodus. Bitte sprich dein Update zuerst ein.",
                "Wichtige Befehle:",
                "- Coop-Eingabe ...",
                "- ZUSAMMENFASSUNG",
                "- SPEICHERN",
                "- Hey Projekt ...",
                "- zurück",
                "- korrigieren",
                "- neu starten",
            ])
        )


def _render_voice_capture() -> None:
    if not st.session_state.get("agent_voice_open"):
        return
    with st.container(border=True):
        st.markdown("### 🎙 Mündliche Dateneingabe")
        st.caption("Bitte sprich deine Projektdaten ein. Nach dem zweiten Klick wird die Aufnahme direkt über OpenAI transkribiert.")
        audio = st.audio_input(
            "Projektupdate aufnehmen",
            key=f"agent_voice_audio_{st.session_state.get('agent_voice_take', 0)}",
            label_visibility="collapsed",
            sample_rate=16000,
        )
        if audio is not None:
            st.audio(audio)
            _maybe_transcribe_audio(audio)

        status = st.session_state.get("agent_voice_status")
        if status == "ready":
            st.success("Transkript bereit. Prüfe es kurz und übernimm es dann in den Dialog.")
        elif status == "error":
            st.error(st.session_state.get("agent_voice_error") or "Transkription fehlgeschlagen.")
        elif status == "cached":
            st.info("Diese Aufnahme wurde bereits transkribiert.")
        else:
            st.info("Noch keine Aufnahme vorhanden.")

        if st.session_state.get("agent_voice_debug"):
            with st.expander("Diagnose (Testmodus)", expanded=False):
                st.json(st.session_state.get("agent_voice_diag") or {})

        voice_text = st.text_area(
            "Transkript",
            value=st.session_state.get("agent_voice_text_value", ""),
            height=180,
            placeholder="Hier erscheint das echte Transkript aus der OpenAI-Transkription.",
        )
        st.session_state["agent_voice_text_value"] = voice_text

        c1, c2, c3 = st.columns(3)
        if c1.button("In Eingabefeld übernehmen", type="primary", use_container_width=True):
            current = _norm(st.session_state.get("agent_prompt_value"))
            tr = _norm(voice_text)
            st.session_state["agent_prompt_value"] = ((current + " " + tr).strip() if current and tr else tr or current)
            st.session_state["agent_prompt_rev"] = int(st.session_state.get("agent_prompt_rev", 0)) + 1
            st.session_state["agent_voice_open"] = False
            _voice_reset(increment_take=True)
            st.rerun()
        if c2.button("Neu aufnehmen", use_container_width=True):
            _voice_reset(increment_take=True)
            st.session_state["agent_voice_open"] = True
            st.rerun()
        if c3.button("Schließen", use_container_width=True):
            st.session_state["agent_voice_open"] = False
            _voice_reset()
            st.rerun()


def _render_login(ctx: Dict[str, Any]) -> None:
    st.markdown('<div class="coop-center">', unsafe_allow_html=True)
    st.markdown("<h2>Anmelden</h2>", unsafe_allow_html=True)
    st.text_input("Benutzername", key="agent_login_username", label_visibility="collapsed", placeholder="Benutzername")
    cols = st.columns([1, 1, 1])
    if cols[1].button("Weiter", use_container_width=True, type="primary"):
        profile = _user_record(ctx, st.session_state.get("agent_login_username", ""))
        if profile:
            st.session_state["agent_auth_profile"] = profile
            _reset_dialog()
            st.rerun()
    st.markdown("<div class='coop-muted'>Sebastian hat Admin-Rechte. Alle anderen sind gleichberechtigt.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_menu_buttons(ctx: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Neue Eingabe", use_container_width=True, type="primary"):
        _begin_new_capture(ctx=ctx)
        st.rerun()
    if c2.button("Suche", use_container_width=True):
        st.session_state["agent_menu"] = "search"
        st.rerun()
    if c3.button("Zusammenfassung", use_container_width=True):
        st.session_state["agent_menu"] = "summary"
        st.rerun()
    if c4.button("Hilfe", use_container_width=True):
        st.session_state["agent_help_open"] = not bool(st.session_state.get("agent_help_open"))
        st.rerun()
    if c5.button("Abmelden", use_container_width=True):
        st.session_state["agent_auth_profile"] = None
        _reset_dialog(keep_user=False)
        st.rerun()


def _render_submenu_buttons() -> None:
    if st.session_state.get("agent_mode") != "capture":
        return
    labels = [(field, FIELD_LABELS.get(field, field)) for field in _field_order()]
    cols = st.columns(3)
    for idx, (field, label) in enumerate(labels):
        value = _norm((st.session_state.get("agent_draft") or {}).get(field))
        style = "✅ " if value or field in {"blockers", "due_hint", "notes"} and field in (st.session_state.get("agent_history_fields") or []) else "• "
        if cols[idx % 3].button(style + label, key=f"submenu_{field}", use_container_width=True):
            st.session_state["agent_current_field"] = field
            st.session_state["agent_confirm_field"] = None
            st.session_state["agent_confirm_value"] = None
            _append("assistant", f"Okay, wir springen zu **{label}**. {_question_for(field)}")
            st.rerun()


def render(ctx: Dict[str, Any]) -> None:
    _init_state()
    _seed_users(ctx)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], header[data-testid="stHeader"] {display:none !important;}
        .block-container {padding-top: 1.4rem; max-width: 980px;}
        .coop-center {max-width: 760px; margin: 12vh auto 0 auto; text-align:center;}
        .coop-muted {color:#6b7280; margin-top:.75rem;}
        .coop-shell {max-width: 920px; margin: 0 auto;}
        .coop-headline {text-align:center; margin-bottom:.3rem;}
        .coop-sub {text-align:center; color:#6b7280; margin-bottom:1rem;}
        .voice-hint {text-align:center; font-weight:700; color:#0f172a; margin-bottom:.8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not _current_user():
        _render_login(ctx)
        return

    st.markdown('<div class="coop-shell">', unsafe_allow_html=True)
    st.markdown('<div class="coop-headline"><h1>Coop Agent v19</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="coop-sub">Hauptmenü · natürlicher KI-Dialog · Menüführung parallel nutzbar</div>', unsafe_allow_html=True)
    st.markdown('<div class="voice-hint">Bitte sprich deine Projektdaten ein. Alternativ kannst du einen Menüpunkt wählen.</div>', unsafe_allow_html=True)

    _render_menu_buttons(ctx)
    _render_menu_guide()

    if st.session_state.get("agent_mode") == "capture":
        _render_submenu_buttons()

    if st.session_state.get("agent_help_open"):
        _render_help_box()

    _render_dialog_views()

    if st.session_state.get("agent_last_saved"):
        info = st.session_state["agent_last_saved"]
        st.success(f"Gespeichert · DB #{info.get('entry_id')} · Tool: {_norm(info.get('task_id')) or 'kein Aufgabenobjekt'}")

    if st.session_state.get("agent_mode") == "home" and not st.session_state.get("agent_messages"):
        _append("assistant", "Willkommen im Hauptmenü. Ich führe dich Schritt für Schritt durch das Menü. Starte mit **Neue Eingabe** oder sprich direkt mit **Coop-Eingabe ...**.")

    for msg in st.session_state.get("agent_messages") or []:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    _render_summary_box()
    _render_search_result()

    prompt_value = st.text_input(
        "Coop Prompt",
        key=f"agent_prompt_input_{st.session_state.get('agent_prompt_rev', 0)}",
        value=st.session_state.get("agent_prompt_value", ""),
        label_visibility="collapsed",
        placeholder="Sprich oder schreibe jetzt deinen nächsten Satz im Gespräch mit der KI",
    )
    st.session_state["agent_prompt_value"] = prompt_value

    c1, c2 = st.columns([1, 1.2])
    if c1.button("Senden", use_container_width=True, type="primary"):
        prompt = _norm(prompt_value)
        if prompt:
            _handle_prompt(ctx, prompt)
        _clear_prompt()
        st.rerun()
    if c2.button("🎙 Mündliche Eingabe starten", use_container_width=True):
        st.session_state["agent_voice_open"] = not bool(st.session_state.get("agent_voice_open"))
        if not st.session_state["agent_voice_open"]:
            _voice_reset()
        st.rerun()

    _render_voice_capture()

    if st.session_state.get("agent_confirm_field"):
        y1, y2 = st.columns(2)
        if y1.button("Ja, stimmt", use_container_width=True):
            _accept_confirmed_value()
            st.rerun()
        if y2.button("Nein, korrigieren", use_container_width=True):
            _reject_confirmed_value()
            st.rerun()

    if not _openai_available():
        st.caption("OpenAI-Sprachtranskription ist noch nicht aktiv. Hinterlege OPENAI_API_KEY in den Streamlit-Secrets.")

    st.markdown('</div>', unsafe_allow_html=True)
