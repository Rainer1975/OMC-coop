from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

__version__ = "2026.03.21.4"

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT,
    role TEXT NOT NULL,
    project_scope TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    display_name TEXT,
    role_at_write TEXT,
    created_at TEXT NOT NULL,
    work_date TEXT NOT NULL,
    portfolio TEXT,
    project TEXT,
    theme TEXT,
    title TEXT NOT NULL,
    status TEXT,
    priority TEXT,
    next_step TEXT,
    blockers TEXT,
    notes TEXT,
    source_text TEXT,
    entry_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(username) REFERENCES users(username)
);
"""

FIELD_ORDER = [
    "work_item",
    "project",
    "status",
    "next_step",
    "blockers",
    "need_task",
    "priority",
    "due_hint",
    "notes",
]

FIELD_LABELS = {
    "work_item": "Arbeitspunkt",
    "project": "Projekt",
    "status": "Status",
    "next_step": "Nächster Schritt",
    "blockers": "Blocker",
    "need_task": "Eintragsart",
    "priority": "Priorität",
    "due_hint": "Termin",
    "notes": "Notizen",
}

FIELD_QUESTIONS = {
    "work_item": "Woran arbeitest du gerade? Sag es einfach in einem Satz.",
    "project": "Zu welchem Projekt gehört das?",
    "status": "Wie ist der aktuelle Stand?",
    "next_step": "Was ist der nächste konkrete Schritt?",
    "blockers": "Gibt es Blocker, Abhängigkeiten oder Risiken? Wenn nichts offen ist, sag einfach 'keine'.",
    "need_task": "Soll ich das als Status-Update, als neue Aufgabe oder als beides speichern?",
    "priority": "Welche Priorität hat das? Niedrig, Mittel oder Hoch?",
    "due_hint": "Gibt es einen Termin oder eine zeitliche Erwartung dazu?",
    "notes": "Gibt es noch etwas, das ich für das Tool mitnehmen soll?",
}

COMMAND_CAPTURE = "coop-eingabe"
COMMAND_LOOKUP = "hey projekt"


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _norm_l(x: Any) -> str:
    return _norm(x).lower()


def _slugify(s: str) -> str:
    s = _norm(s).lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else ".")
    v = "".join(out).strip(".")
    while ".." in v:
        v = v.replace("..", ".")
    return v or "user"


def _is_sebastian(name_or_username: str) -> bool:
    t = _norm_l(name_or_username)
    return bool(re.search(r"(^|[ ._-])sebastian($|[ ._-])", t)) or t == "sebastian"


def get_db_path(ctx: Dict[str, Any]) -> Path:
    return Path(str(ctx.get("AGENT_DB_FILE") or "agent_intake.db"))


def get_roles_path(ctx: Dict[str, Any]) -> Path:
    return Path(str(ctx.get("ROLES_FILE") or "access_roles.json"))


def _db(ctx: Dict[str, Any]) -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(ctx))
    conn.row_factory = sqlite3.Row
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


# ---------- users / auth ----------
def _employees_from_ctx(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = ctx.get("employees") or []
    out: List[Dict[str, Any]] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        dn = _norm(e.get("display_name"))
        if not dn:
            continue
        username = _norm(e.get("id")) or _slugify(dn)
        aliases = e.get("aliases") if isinstance(e.get("aliases"), list) else []
        out.append({"username": _norm_l(username), "display_name": dn, "aliases": aliases})
    return out


def _roles_file_users(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = get_roles_path(ctx)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    users = data.get("users") if isinstance(data, dict) else []
    out: List[Dict[str, Any]] = []
    for u in users or []:
        if not isinstance(u, dict):
            continue
        username = _norm_l(u.get("username"))
        display_name = _norm(u.get("display_name")) or username
        if username:
            out.append({
                "username": username,
                "display_name": display_name,
                "aliases": [],
                "project_scope": u.get("project_scope", []) if isinstance(u.get("project_scope"), list) else [],
            })
    return out


def _project_options(ctx: Dict[str, Any]) -> List[str]:
    projects = list((ctx.get("lists") or {}).get("projects", []) or [])
    visible = {_norm(p) for p in projects if _norm(p)}
    for u in _roles_file_users(ctx):
        for p in u.get("project_scope") or []:
            if _norm(p):
                visible.add(_norm(p))
    return sorted(visible)


def _default_scope(ctx: Dict[str, Any]) -> List[str]:
    projects = _project_options(ctx)
    return projects[:1] if projects else []


def _collect_actual_users(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    # first choice: employees.json from the uploaded ZIP
    candidates = _employees_from_ctx(ctx)
    # fallback: existing role file from the current project folder, if present
    if not candidates:
        candidates = _roles_file_users(ctx)

    out: List[Dict[str, Any]] = []
    seen = set()
    for c in candidates:
        username = _norm_l(c.get("username"))
        display_name = _norm(c.get("display_name")) or username
        if not username or username in seen:
            continue
        seen.add(username)
        role = "admin" if (_is_sebastian(username) or _is_sebastian(display_name)) else "employee"
        scope = c.get("project_scope") if isinstance(c.get("project_scope"), list) else _default_scope(ctx)
        out.append({
            "username": username,
            "display_name": display_name,
            "role": role,
            "project_scope": [_norm(x) for x in scope if _norm(x)],
        })
    return out


def _write_roles_snapshot(ctx: Dict[str, Any], users: List[Dict[str, Any]]) -> None:
    data = {"users": users}
    get_roles_path(ctx).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_users(ctx: Dict[str, Any]) -> None:
    actual_users = _collect_actual_users(ctx)
    _write_roles_snapshot(ctx, actual_users)
    now = datetime.now().isoformat(timespec="seconds")
    conn = _db(ctx)
    with conn:
        for u in actual_users:
            conn.execute(
                """
                INSERT INTO users(username, display_name, role, project_scope, created_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(username) DO UPDATE SET
                    display_name=excluded.display_name,
                    role=excluded.role,
                    project_scope=excluded.project_scope
                """,
                (
                    u["username"],
                    u["display_name"],
                    u["role"],
                    json.dumps(u.get("project_scope", []), ensure_ascii=False),
                    now,
                ),
            )
    conn.close()


def _known_users(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    _seed_users(ctx)
    conn = _db(ctx)
    rows = conn.execute("SELECT * FROM users ORDER BY display_name COLLATE NOCASE").fetchall()
    conn.close()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["project_scope"] = json.loads(d.get("project_scope") or "[]")
        except Exception:
            d["project_scope"] = []
        out.append(d)
    return out


def _user_record(ctx: Dict[str, Any], username: str) -> Dict[str, Any]:
    _seed_users(ctx)
    username = _norm_l(username)
    conn = _db(ctx)
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row:
        d = dict(row)
        try:
            d["project_scope"] = json.loads(d.get("project_scope") or "[]")
        except Exception:
            d["project_scope"] = []
        return d
    return {}


def _create_manual_user(ctx: Dict[str, Any], username: str) -> Dict[str, Any]:
    username = _norm_l(username)
    role = "admin" if _is_sebastian(username) else "employee"
    profile = {
        "username": username,
        "display_name": username,
        "role": role,
        "project_scope": _default_scope(ctx),
    }
    conn = _db(ctx)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO users(username, display_name, role, project_scope, created_at) VALUES(?,?,?,?,?)",
            (
                profile["username"],
                profile["display_name"],
                profile["role"],
                json.dumps(profile.get("project_scope", []), ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    conn.close()
    return _user_record(ctx, username)


def ensure_session_auth(ctx: Dict[str, Any]) -> None:
    st.session_state.setdefault("auth_username", "")
    st.session_state.setdefault("auth_profile", None)

    with st.sidebar:
        st.subheader("Anmeldung")
        known = _known_users(ctx)
        label_map = {u["display_name"]: u["username"] for u in known if _norm(u.get("display_name"))}
        options = ["Benutzername manuell eingeben"] + list(label_map.keys())
        picked = st.selectbox("Vorhandene Nutzer", options=options, index=0, key="auth_known_user")
        default_username = label_map.get(picked, st.session_state.get("auth_username", ""))
        username = st.text_input(
            "Benutzername",
            value=default_username,
            key="auth_username_input",
            placeholder="z. B. alina.punt",
        ).strip().lower()
        c1, c2 = st.columns(2)
        login_clicked = c1.button("Anmelden", use_container_width=True)
        logout_clicked = c2.button("Abmelden", use_container_width=True)

        if logout_clicked:
            st.session_state.auth_username = ""
            st.session_state.auth_profile = None
            for key in [k for k in list(st.session_state.keys()) if k.startswith("agent_")]:
                del st.session_state[key]
            st.rerun()

        if login_clicked and username:
            profile = _user_record(ctx, username) or _create_manual_user(ctx, username)
            st.session_state.auth_username = username
            st.session_state.auth_profile = profile
            _reset_conversation(force=True)
            st.rerun()

        profile = st.session_state.get("auth_profile") or {}
        if profile:
            role_label = {"employee": "Mitarbeiter:in", "admin": "Administrator"}.get(profile.get("role"), profile.get("role") or "unbekannt")
            st.success(f"{profile.get('display_name') or profile.get('username')} · {role_label}")
        else:
            st.info("Für den Test reicht der Benutzername. Alle sind gleichberechtigt. Nur Sebastian ist Administrator.")


# ---------- visibility ----------
def _current_user() -> Dict[str, Any]:
    return st.session_state.get("auth_profile") or {}


def _can_read_entry(user: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    if _norm(user.get("role")) == "admin":
        return True
    return _norm_l(entry.get("username")) == _norm_l(user.get("username"))


def _fetch_entries(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    conn = _db(ctx)
    rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC, id DESC").fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        try:
            d["payload_json"] = json.loads(d.get("payload_json") or "{}")
        except Exception:
            d["payload_json"] = {}
        out.append(d)
    return out


def _visible_entries(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    user = _current_user()
    if not user:
        return []
    return [e for e in _fetch_entries(ctx) if _can_read_entry(user, e)]


def _visible_series(ctx: Dict[str, Any]) -> List[Any]:
    user = _current_user()
    series = list(st.session_state.get("series") or [])
    if not user:
        return []
    out = []
    for s in series:
        if _norm(user.get("role")) == "admin":
            out.append(s)
        elif _norm_l(getattr(s, "owner_id", "")) == _norm_l(user.get("username")):
            out.append(s)
    return out


# ---------- persistence ----------
def _upsert_user_from_auth(ctx: Dict[str, Any]) -> Dict[str, Any]:
    user = _current_user()
    if not user:
        return {}
    conn = _db(ctx)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO users(username, display_name, role, project_scope, created_at) VALUES(?,?,?,?,?)",
            (
                user.get("username"),
                user.get("display_name") or user.get("username"),
                user.get("role") or "employee",
                json.dumps(user.get("project_scope", []), ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    conn.close()
    return user


def _store_entry(ctx: Dict[str, Any], payload: Dict[str, Any]) -> int:
    user = _upsert_user_from_auth(ctx)
    conn = _db(ctx)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO entries(
                username, display_name, role_at_write, created_at, work_date,
                portfolio, project, theme, title, status, priority, next_step,
                blockers, notes, source_text, entry_kind, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user.get("username"),
                user.get("display_name") or user.get("username"),
                user.get("role") or "employee",
                datetime.now().isoformat(timespec="seconds"),
                payload.get("work_date") or date.today().isoformat(),
                payload.get("portfolio") or "Default",
                payload.get("project") or "",
                payload.get("theme") or "General",
                payload.get("title") or "Unbenannter Eintrag",
                payload.get("status") or "",
                payload.get("priority") or "",
                payload.get("next_step") or "",
                payload.get("blockers") or "",
                payload.get("notes") or "",
                payload.get("source_text") or "",
                payload.get("entry_kind") or "status_update",
                json.dumps(payload, ensure_ascii=False),
            ),
        )
    last_id = int(cur.lastrowid)
    conn.close()
    return last_id


# ---------- command parsing ----------
def _extract_project(text: str, ctx: Dict[str, Any]) -> str:
    t = _norm(text)
    tl = t.lower()
    for project in _project_options(ctx):
        if project.lower() in tl:
            return project
    m = re.search(r"projekt\s+([A-Za-z0-9ÄÖÜäöüß\-_ ]{2,60})", t, flags=re.IGNORECASE)
    return _norm(m.group(1)) if m else ""


def _extract_priority(text: str) -> str:
    tl = _norm_l(text)
    if any(w in tl for w in ["hoch", "kritisch", "dringend", "prio 1", "p1"]):
        return "Hoch"
    if any(w in tl for w in ["niedrig", "später", "nicht dringend", "prio 3", "p3"]):
        return "Niedrig"
    if any(w in tl for w in ["mittel", "normal", "prio 2", "p2"]):
        return "Mittel"
    return ""


def _extract_entry_kind(text: str) -> str:
    tl = _norm_l(text)
    if not tl:
        return ""
    if "beides" in tl or ("status" in tl and "aufgabe" in tl):
        return "task_and_status"
    if any(w in tl for w in ["aufgabe", "anlegen", "neu", "ticket"]):
        return "task"
    if "status" in tl or "update" in tl:
        return "status_update"
    return ""


def _extract_due_hint(text: str) -> str:
    t = _norm(text)
    tl = t.lower()
    patterns = ["heute", "morgen", "diese woche", "nächste woche", "bis", "deadline", "frist", "termin"]
    return t if any(p in tl for p in patterns) else ""


def _infer_from_text(text: str, ctx: Dict[str, Any]) -> Dict[str, str]:
    t = _norm(text)
    tl = t.lower()
    inferred: Dict[str, str] = {}
    if not t:
        return inferred
    project = _extract_project(t, ctx)
    if project:
        inferred["project"] = project
    prio = _extract_priority(t)
    if prio:
        inferred["priority"] = prio
    ek = _extract_entry_kind(t)
    if ek:
        inferred["need_task"] = ek
    due_hint = _extract_due_hint(t)
    if due_hint:
        inferred["due_hint"] = due_hint
    if any(w in tl for w in ["blocker", "abhängig", "warte auf", "risiko", "hindernis"]):
        inferred["blockers"] = t
    if any(w in tl for w in ["nächster schritt", "als nächstes", "ich mache jetzt", "danach", "weiter mit"]):
        inferred["next_step"] = t
    if any(w in tl for w in ["fertig", "in arbeit", "offen", "läuft", "im review", "warte", "abgestimmt", "status"]):
        inferred["status"] = t
    if not inferred.get("work_item"):
        inferred["work_item"] = t
    return inferred


def _smart_followup(answers: Dict[str, Any], missing_field: str) -> str:
    if missing_field == "status" and _norm(answers.get("work_item")):
        return f"Verstanden. Wie ist bei '{_norm(answers.get('work_item'))}' der aktuelle Stand?"
    if missing_field == "next_step" and _norm(answers.get("project")):
        return f"Gut. Was ist im Projekt {_norm(answers.get('project'))} der nächste konkrete Schritt?"
    if missing_field == "need_task":
        return "Soll ich das nur als Status-Update speichern, als neue Aufgabe anlegen oder beides machen?"
    return FIELD_QUESTIONS.get(missing_field, "Kannst du das noch kurz ergänzen?")


def _normalize_special_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(answers or {})
    nt = _norm_l(out.get("need_task"))
    if nt and nt not in {"task", "status_update", "task_and_status"}:
        out["need_task"] = _extract_entry_kind(nt) or out.get("need_task")
    if _norm_l(out.get("blockers")) in {"keine", "kein", "none", "nichts"}:
        out["blockers"] = ""
    prio = _extract_priority(_norm(out.get("priority")))
    if prio:
        out["priority"] = prio
    elif not _norm(out.get("priority")) and _norm(out.get("need_task")) in {"task", "task_and_status"}:
        out["priority"] = "Mittel"
    return out


def _conversation_complete(answers: Dict[str, Any]) -> bool:
    for key in ["work_item", "project", "status", "next_step", "need_task"]:
        if not _norm(answers.get(key)):
            return False
    if _norm(answers.get("need_task")) in {"task", "task_and_status"} and not _norm(answers.get("priority")):
        return False
    return True


def _next_missing_field(answers: Dict[str, Any]) -> Optional[str]:
    for key in FIELD_ORDER:
        if key == "priority" and _norm(answers.get("need_task")) not in {"task", "task_and_status"}:
            continue
        if not _norm(answers.get(key)):
            return key
    return None


def _append_chat(role: str, content: str) -> None:
    st.session_state.agent_chat.append({"role": role, "content": content})


def _bootstrap_chat(force: bool = False) -> None:
    if force or "agent_chat" not in st.session_state:
        st.session_state.agent_chat = [{
            "role": "assistant",
            "content": (
                "Hallo. Gib deinen Eintrag einfach in natürlicher Sprache ein. "
                "Für neue Daten mit 'Coop-Eingabe ...'. Für gespeicherte Infos mit 'Hey Projekt ...'."
            ),
        }]
    st.session_state.setdefault("agent_answers", {})
    st.session_state.setdefault("agent_saved_id", None)
    st.session_state.setdefault("agent_mode", None)
    st.session_state.setdefault("agent_lookup_results", [])


def _reset_conversation(force: bool = False) -> None:
    for k in [k for k in list(st.session_state.keys()) if k.startswith("agent_")]:
        del st.session_state[k]
    _bootstrap_chat(force=True)


def _merge_answers(existing: Dict[str, Any], inferred: Dict[str, str]) -> Dict[str, Any]:
    out = dict(existing or {})
    for key, value in inferred.items():
        value = _norm(value)
        if not value:
            continue
        if key == "need_task":
            out[key] = value
        elif not _norm(out.get(key)):
            out[key] = value
        elif key in {"status", "blockers", "notes"} and value != _norm(out.get(key)):
            out[key] = value
    return out


# ---------- summary / save ----------
def _conversation_summary() -> Dict[str, Any]:
    answers = _normalize_special_answers(dict(st.session_state.get("agent_answers") or {}))
    user = _current_user()
    source_lines = []
    for m in st.session_state.get("agent_chat", []):
        role = "Agent" if m.get("role") == "assistant" else (user.get("display_name") or user.get("username") or "User")
        source_lines.append(f"{role}: {m.get('content')}")
    return {
        "work_date": date.today().isoformat(),
        "portfolio": "Default",
        "project": answers.get("project", ""),
        "theme": "General",
        "title": answers.get("work_item", "") or "Neuer Arbeitseintrag",
        "status": answers.get("status", ""),
        "priority": answers.get("priority", ""),
        "next_step": answers.get("next_step", ""),
        "blockers": answers.get("blockers", ""),
        "due_hint": answers.get("due_hint", ""),
        "notes": answers.get("notes", ""),
        "entry_kind": answers.get("need_task", "status_update") or "status_update",
        "source_text": "\n".join(source_lines),
        "answers": answers,
    }


def _write_to_planning_tool(ctx: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("entry_kind") not in {"task", "task_and_status"}:
        return None
    new_series = ctx.get("new_series")
    save_series = ctx.get("save_series")
    sync_lists_from_data = ctx.get("sync_lists_from_data")
    if not callable(new_series) or not callable(save_series):
        return None
    user = _current_user()
    start = date.today()
    end = start + timedelta(days=2)
    s = new_series(
        title=payload.get("title") or "Neuer Arbeitseintrag",
        portfolio=payload.get("portfolio") or "Default",
        project=payload.get("project") or "",
        theme=payload.get("theme") or "General",
        owner=user.get("display_name") or user.get("username") or "",
        owner_id=user.get("username") or "",
        start=start,
        end=end,
        is_meta=False,
        kind="task",
        state="ACTIVE",
    )
    s.meta = dict(getattr(s, "meta", {}) or {})
    s.meta["agent_entry"] = {
        "status": payload.get("status"),
        "next_step": payload.get("next_step"),
        "blockers": payload.get("blockers"),
        "notes": payload.get("notes"),
        "due_hint": payload.get("due_hint"),
        "priority": payload.get("priority"),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    series = list(st.session_state.get("series") or [])
    series.append(s)
    save_series(series)
    if callable(sync_lists_from_data):
        sync_lists_from_data()
    return getattr(s, "series_id", None)


def _save_current_summary(ctx: Dict[str, Any]) -> None:
    summary = _conversation_summary()
    entry_id = _store_entry(ctx, summary)
    series_id = _write_to_planning_tool(ctx, summary)
    st.session_state.agent_saved_id = {"entry_id": entry_id, "series_id": series_id}


# ---------- lookup ----------
def _lookup_query_from_prompt(prompt: str) -> str:
    q = _norm(prompt)
    if _norm_l(q).startswith(COMMAND_LOOKUP):
        q = _norm(q[len(COMMAND_LOOKUP):])
    return q


def _lookup_entries(ctx: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
    query = _lookup_query_from_prompt(prompt)
    entries = _visible_entries(ctx)
    if not query:
        return entries[:8]
    ql = query.lower()
    matches: List[Dict[str, Any]] = []
    for e in entries:
        hay = " | ".join([
            _norm(e.get("project")),
            _norm(e.get("title")),
            _norm(e.get("status")),
            _norm(e.get("next_step")),
            _norm(e.get("notes")),
            _norm((e.get("payload_json") or {}).get("due_hint")),
        ]).lower()
        if ql in hay:
            matches.append(e)
    return matches[:12]


# ---------- chat flow ----------
def _start_capture_mode(prompt: str, ctx: Dict[str, Any]) -> None:
    payload = _norm(prompt[len(COMMAND_CAPTURE):]) if _norm_l(prompt).startswith(COMMAND_CAPTURE) else prompt
    st.session_state.agent_mode = "capture"
    inferred = _infer_from_text(payload, ctx)
    answers = _normalize_special_answers(_merge_answers(st.session_state.get("agent_answers") or {}, inferred))
    st.session_state.agent_answers = answers
    missing = _next_missing_field(answers)
    if missing:
        _append_chat("assistant", _smart_followup(answers, missing))
    else:
        _save_current_summary(ctx)
        sid = st.session_state.agent_saved_id or {}
        msg = f"Danke. Gespeichert als Datenbank-Eintrag #{sid.get('entry_id')}."
        if sid.get("series_id"):
            msg += f" Zusätzlich als Aufgabe im Tool angelegt: {sid.get('series_id')}."
        _append_chat("assistant", msg)


def _continue_capture_mode(prompt: str, ctx: Dict[str, Any]) -> None:
    inferred = _infer_from_text(prompt, ctx)
    if not inferred:
        missing = _next_missing_field(st.session_state.get("agent_answers") or {}) or "notes"
        inferred = {missing: prompt}
    answers = _merge_answers(st.session_state.get("agent_answers") or {}, inferred)
    answers = _normalize_special_answers(answers)
    st.session_state.agent_answers = answers
    missing = _next_missing_field(answers)
    if missing:
        _append_chat("assistant", _smart_followup(answers, missing))
    else:
        _save_current_summary(ctx)
        sid = st.session_state.agent_saved_id or {}
        st.session_state.agent_mode = None
        msg = f"Danke. Gespeichert als Datenbank-Eintrag #{sid.get('entry_id')}."
        if sid.get("series_id"):
            msg += f" Zusätzlich als Aufgabe im Tool angelegt: {sid.get('series_id')}."
        _append_chat("assistant", msg)


def _handle_prompt(ctx: Dict[str, Any]) -> None:
    prompt = st.chat_input("Coop-Eingabe ... oder Hey Projekt ...")
    if not prompt:
        return
    _append_chat("user", prompt)
    lp = _norm_l(prompt)
    st.session_state.agent_saved_id = None
    st.session_state.agent_lookup_results = []

    if lp.startswith(COMMAND_CAPTURE):
        _start_capture_mode(prompt, ctx)
    elif lp.startswith(COMMAND_LOOKUP):
        st.session_state.agent_mode = None
        results = _lookup_entries(ctx, prompt)
        st.session_state.agent_lookup_results = results
        if results:
            _append_chat("assistant", f"Ich habe {len(results)} passende Projektinformationen gefunden.")
        else:
            _append_chat("assistant", "Dazu habe ich in deinem sichtbaren Bereich nichts gefunden.")
    elif st.session_state.get("agent_mode") == "capture":
        _continue_capture_mode(prompt, ctx)
    else:
        _append_chat(
            "assistant",
            "Starte mit 'Coop-Eingabe ...' zum Erfassen oder mit 'Hey Projekt ...' zum Suchen. Ohne diesen Befehl speichere oder suche ich nichts.",
        )
    st.rerun()


# ---------- rendering ----------
def _render_progressive_fields() -> None:
    answers = _normalize_special_answers(st.session_state.get("agent_answers") or {})
    if not answers:
        return
    st.markdown("### Erkannte Informationen")
    cols = st.columns(2)
    items = [(FIELD_LABELS[k], _norm(v)) for k, v in answers.items() if _norm(v)]
    for idx, (label, value) in enumerate(items):
        cols[idx % 2].text_input(label, value=value, disabled=True, key=f"view_{label}_{idx}")
    missing = _next_missing_field(answers)
    if missing:
        st.caption(f"Es fehlt noch: {FIELD_LABELS.get(missing, missing)}")


def _render_lookup_results(ctx: Dict[str, Any]) -> None:
    results = st.session_state.get("agent_lookup_results") or []
    if not results:
        return
    st.markdown("### Gefundene Projektinformationen")
    for e in results:
        with st.expander(f"{e.get('project') or 'ohne Projekt'} · {e.get('title')}"):
            st.write(f"**Erfasst von:** {e.get('display_name')}")
            if e.get("status"):
                st.write(f"**Status:** {e.get('status')}")
            if e.get("next_step"):
                st.write(f"**Nächster Schritt:** {e.get('next_step')}")
            if e.get("blockers"):
                st.write(f"**Blocker:** {e.get('blockers')}")
            if e.get("notes"):
                st.write(f"**Notizen:** {e.get('notes')}")
            due_hint = (e.get("payload_json") or {}).get("due_hint")
            if due_hint:
                st.write(f"**Termin-Hinweis:** {due_hint}")


def _render_saved_views(ctx: Dict[str, Any]) -> None:
    sid = st.session_state.get("agent_saved_id") or {}
    if not sid:
        return
    st.markdown("### Im Hintergrund verfügbar")
    tab1, tab2 = st.tabs(["Agent-Datenbank", "Planungstool"])
    with tab1:
        entries = _visible_entries(ctx)
        for e in entries[:5]:
            st.write(f"**#{e.get('id')} · {e.get('project') or 'ohne Projekt'} · {e.get('title')}**")
            if e.get("status"):
                st.caption(e.get("status"))
    with tab2:
        items = _visible_series(ctx)
        for s in items[-5:][::-1]:
            st.write(f"**{getattr(s, 'project', '') or 'ohne Projekt'} · {getattr(s, 'title', '')}**")
            meta = dict(getattr(s, "meta", {}) or {})
            agent_meta = meta.get("agent_entry") or {}
            if agent_meta:
                st.json(agent_meta)


def _load_demo_conversation(ctx: Dict[str, Any]) -> None:
    user = _current_user()
    project = (_default_scope(ctx) or _project_options(ctx) or ["Chatcheck"])[0] if (_default_scope(ctx) or _project_options(ctx)) else ""
    _reset_conversation(force=True)
    line = (
        f"Coop-Eingabe Ich arbeite heute an {project or 'dem Projekt'}. Der aktuelle Stand ist in Arbeit. "
        "Als nächstes finalisiere ich die offenen Punkte. Ich warte noch auf eine Rückmeldung vom Fachbereich. "
        "Bitte als beides speichern, Priorität hoch, möglichst bis morgen."
    )
    st.session_state.agent_chat = [
        {"role": "assistant", "content": f"Hallo {user.get('display_name') or user.get('username')}. Gib deinen Eintrag einfach in natürlicher Sprache ein. Für neue Daten mit 'Coop-Eingabe ...'. Für gespeicherte Infos mit 'Hey Projekt ...'."},
        {"role": "user", "content": line},
        {"role": "assistant", "content": "Danke. Gespeichert. Du kannst jetzt mit 'Hey Projekt' danach suchen."},
    ]
    st.session_state.agent_answers = _normalize_special_answers(_infer_from_text(line, ctx))
    _save_current_summary(ctx)


def render(ctx: Dict[str, Any]) -> None:
    st.title("Agent Intake")
    ensure_session_auth(ctx)
    user = _current_user()
    if not user:
        st.info("Melde dich links mit dem vorhandenen Benutzernamen aus dem ZIP an. Wenn im ZIP keine Nutzer gepflegt sind, kannst du den Namen trotzdem manuell testen.")
        return

    _bootstrap_chat()

    top = st.columns([1, 1, 2])
    if top[0].button("Beispiel laden", use_container_width=True):
        _load_demo_conversation(ctx)
        st.rerun()
    if top[1].button("Neu", use_container_width=True):
        _reset_conversation(force=True)
        st.rerun()
    top[2].caption("Startzustand: nur Dialog. Weitere Informationen erscheinen erst nach einem Befehl oder nach der schrittweisen Erfassung.")

    for m in st.session_state.get("agent_chat", []):
        with st.chat_message("assistant" if m.get("role") == "assistant" else "user"):
            st.write(m.get("content"))

    _handle_prompt(ctx)
    _render_progressive_fields()
    _render_lookup_results(ctx)
    _render_saved_views(ctx)
