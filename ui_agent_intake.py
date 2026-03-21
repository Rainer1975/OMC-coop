from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

__version__ = "2026.03.21.2"

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

SMART_HINTS = [
    "Du kannst frei formulieren oder einfach diktieren.",
    "Ich frage nur nach, was noch fehlt.",
    "Am Ende bekommst du eine kompakte Zusammenfassung vor dem Speichern.",
]


# ---------- basics ----------
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


def get_db_path(ctx: Dict[str, Any]) -> Path:
    p = ctx.get("AGENT_DB_FILE") or "agent_intake.db"
    return Path(str(p))


def get_roles_path(ctx: Dict[str, Any]) -> Path:
    p = ctx.get("ROLES_FILE") or "access_roles.json"
    return Path(str(p))


def _db(ctx: Dict[str, Any]) -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(ctx))
    conn.row_factory = sqlite3.Row
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


# ---------- user sources ----------
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
        out.append({"username": username, "display_name": dn})
    return out


def _default_roles(ctx: Dict[str, Any]) -> Dict[str, Any]:
    employees = _employees_from_ctx(ctx)
    projects = list((ctx.get("lists") or {}).get("projects", []) or [])
    first_project = projects[0] if projects else "Chatcheck"

    users: List[Dict[str, Any]] = []
    seen = set()
    for e in employees:
        uname = _norm_l(e.get("username"))
        if not uname or uname in seen:
            continue
        seen.add(uname)
        users.append(
            {
                "username": uname,
                "display_name": e.get("display_name") or uname,
                "role": "employee",
                "project_scope": [first_project] if first_project else [],
            }
        )

    # Fallback only if the uploaded data does not contain people yet.
    if not users:
        users = [
            {
                "username": "alina.punt",
                "display_name": "Alina Punt",
                "role": "employee",
                "project_scope": ["Chatcheck"],
            },
            {
                "username": "projektleitung.chatcheck",
                "display_name": "Projektleitung Chatcheck",
                "role": "project_lead",
                "project_scope": ["Chatcheck"],
            },
            {
                "username": "admin",
                "display_name": "Admin",
                "role": "admin",
                "project_scope": [],
            },
        ]
    else:
        users.append(
            {
                "username": "admin",
                "display_name": "Admin",
                "role": "admin",
                "project_scope": [],
            }
        )
        if first_project:
            users.append(
                {
                    "username": f"projektleitung.{_slugify(first_project)}",
                    "display_name": f"Projektleitung {first_project}",
                    "role": "project_lead",
                    "project_scope": [first_project],
                }
            )

    return {"users": users}


def _read_roles(ctx: Dict[str, Any]) -> Dict[str, Any]:
    p = get_roles_path(ctx)
    if not p.exists():
        data = _default_roles(ctx)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("users"), list) and data.get("users"):
            return data
    except Exception:
        pass
    data = _default_roles(ctx)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _seed_users(ctx: Dict[str, Any]) -> None:
    roles = _read_roles(ctx)
    users = roles.get("users") if isinstance(roles, dict) else []
    if not isinstance(users, list):
        users = []
    now = datetime.now().isoformat(timespec="seconds")
    conn = _db(ctx)
    with conn:
        for u in users:
            username = _norm_l(u.get("username"))
            if not username:
                continue
            display_name = _norm(u.get("display_name")) or username
            role = _norm(u.get("role")) or "employee"
            project_scope = json.dumps(u.get("project_scope", []), ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO users(username, display_name, role, project_scope, created_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(username) DO UPDATE SET
                    display_name=excluded.display_name,
                    role=excluded.role,
                    project_scope=excluded.project_scope
                """,
                (username, display_name, role, project_scope, now),
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


# ---------- auth ----------
def ensure_session_auth(ctx: Dict[str, Any]) -> None:
    st.session_state.setdefault("auth_username", "")
    st.session_state.setdefault("auth_profile", None)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Zugang")
    known = _known_users(ctx)
    label_map = {
        u["display_name"]: u["username"]
        for u in known
        if _norm(u.get("display_name")) and _norm(u.get("username"))
    }
    display_options = ["Benutzername manuell eingeben"] + list(label_map.keys())
    picked_display = st.sidebar.selectbox("Bekannte Benutzer", options=display_options, index=0, key="auth_known_user")

    default_username = ""
    if picked_display in label_map:
        default_username = label_map[picked_display]

    username = st.sidebar.text_input(
        "Benutzername",
        value=default_username or st.session_state.get("auth_username", ""),
        key="auth_username_input",
        placeholder="z. B. alina.punt",
    ).strip().lower()

    login_clicked = st.sidebar.button("Anmelden", use_container_width=True, key="login_button")
    logout_clicked = st.sidebar.button("Abmelden", use_container_width=True, key="logout_button")

    if logout_clicked:
        st.session_state.auth_username = ""
        st.session_state.auth_profile = None
        for key in [k for k in list(st.session_state.keys()) if k.startswith("agent_")]:
            del st.session_state[key]
        st.rerun()

    if login_clicked and username:
        profile = _user_record(ctx, username)
        if not profile:
            conn = _db(ctx)
            now = datetime.now().isoformat(timespec="seconds")
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO users(username, display_name, role, project_scope, created_at) VALUES(?,?,?,?,?)",
                    (username, username, "employee", json.dumps([], ensure_ascii=False), now),
                )
            conn.close()
            profile = _user_record(ctx, username)
        st.session_state.auth_username = username
        st.session_state.auth_profile = profile
        _reset_conversation(force=True)
        st.rerun()

    profile = st.session_state.get("auth_profile") or {}
    if profile:
        role_label = {
            "employee": "Mitarbeiter:in",
            "project_lead": "Projektleitung",
            "admin": "Admin",
        }.get(profile.get("role"), profile.get("role") or "unbekannt")
        st.sidebar.success(f"Angemeldet als {profile.get('display_name') or profile.get('username')} · {role_label}")
        if profile.get("project_scope"):
            st.sidebar.caption("Projektbezug: " + ", ".join(profile.get("project_scope", [])))
    else:
        st.sidebar.info("Für den Test reicht ein Benutzername. Microsoft-Login ist in diesem Stand bewusst noch nicht eingebaut.")


# ---------- visibility ----------
def _current_user() -> Dict[str, Any]:
    return st.session_state.get("auth_profile") or {}


def _allowed_projects(user: Dict[str, Any]) -> List[str]:
    scope = user.get("project_scope") or []
    if isinstance(scope, list):
        return [_norm(x) for x in scope if _norm(x)]
    return []


def _can_read_entry(user: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    role = _norm(user.get("role"))
    username = _norm_l(user.get("username"))
    if role == "admin":
        return True
    if role == "project_lead":
        allowed = {_norm(x) for x in _allowed_projects(user)}
        return _norm(entry.get("project")) in allowed
    return username and _norm_l(entry.get("username")) == username


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
    role = _norm(user.get("role"))
    uname = _norm_l(user.get("username"))
    allowed_projects = {_norm(x) for x in _allowed_projects(user)}
    out = []
    for s in series:
        owner_id = _norm_l(getattr(s, "owner_id", ""))
        project = _norm(getattr(s, "project", ""))
        if role == "admin":
            out.append(s)
        elif role == "project_lead":
            if project in allowed_projects:
                out.append(s)
        else:
            if owner_id == uname:
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


# ---------- smart dialog ----------
def _project_options(ctx: Dict[str, Any]) -> List[str]:
    projects = list((ctx.get("lists") or {}).get("projects", []) or [])
    visible = {_norm(p) for p in projects if _norm(p)}
    for u in _known_users(ctx):
        for p in u.get("project_scope") or []:
            if _norm(p):
                visible.add(_norm(p))
    if not visible:
        visible.add("Chatcheck")
    return sorted(visible)


def _extract_project(text: str, ctx: Dict[str, Any]) -> str:
    t = _norm(text)
    tl = t.lower()
    for project in _project_options(ctx):
        if project.lower() in tl:
            return project
    m = re.search(r"projekt\s+([A-Za-z0-9ÄÖÜäöüß\-_ ]{2,40})", t, flags=re.IGNORECASE)
    if m:
        return _norm(m.group(1))
    return ""


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
    if any(p in tl for p in patterns):
        return t
    return ""


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


def _friendly_field_label(key: str) -> str:
    return {
        "work_item": "Arbeitspunkt",
        "project": "Projekt",
        "status": "Status",
        "next_step": "Nächster Schritt",
        "blockers": "Blocker",
        "need_task": "Eintragsart",
        "priority": "Priorität",
        "due_hint": "Termin",
        "notes": "Notizen",
    }.get(key, key)


def _smart_followup(answers: Dict[str, Any], missing_field: str) -> str:
    base = FIELD_QUESTIONS.get(missing_field, "Kannst du das noch kurz ergänzen?")
    project = _norm(answers.get("project"))
    work_item = _norm(answers.get("work_item"))
    if missing_field == "status" and work_item:
        return f"Verstanden. Wie ist bei '{work_item}' der aktuelle Stand?"
    if missing_field == "next_step" and project:
        return f"Gut. Was ist im Projekt {project} der nächste konkrete Schritt?"
    if missing_field == "need_task":
        return "Soll ich das nur als Status-Update speichern, als neue Aufgabe anlegen oder beides machen?"
    return base


def _conversation_complete(answers: Dict[str, Any]) -> bool:
    required = ["work_item", "project", "status", "next_step", "need_task"]
    for key in required:
        if not _norm(answers.get(key)):
            return False
    entry_kind = _norm(answers.get("need_task"))
    if entry_kind in {"task", "task_and_status"}:
        if not _norm(answers.get("priority")):
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
        st.session_state.agent_chat = [
            {
                "role": "assistant",
                "content": "Hallo. Ich sammle deinen Arbeitsstand für heute. Sag einfach frei oder diktiert, woran du arbeitest.",
            }
        ]
    st.session_state.setdefault("agent_answers", {})
    st.session_state.setdefault("agent_saved_id", None)
    st.session_state.setdefault("agent_demo_loaded", False)


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
            continue
        if not _norm(out.get(key)):
            out[key] = value
            continue
        if key in {"status", "blockers", "notes"} and value != out.get(key):
            out[key] = value
    return out


def _normalize_special_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(answers or {})
    need_task = _norm_l(out.get("need_task"))
    if need_task in {"task", "status_update", "task_and_status"}:
        pass
    elif need_task:
        out["need_task"] = _extract_entry_kind(need_task) or out.get("need_task")
    if _norm_l(out.get("blockers")) in {"keine", "kein", "none", "nichts"}:
        out["blockers"] = ""
    prio = _extract_priority(_norm(out.get("priority")))
    if prio:
        out["priority"] = prio
    elif not _norm(out.get("priority")):
        out["priority"] = "Mittel" if _norm(out.get("need_task")) in {"task", "task_and_status"} else ""
    return out


def _handle_chat_input(ctx: Dict[str, Any]) -> None:
    prompt = st.chat_input("Antwort eingeben oder diktierten Text hier einfügen")
    if not prompt:
        return
    _append_chat("user", prompt)
    inferred = _infer_from_text(prompt, ctx)
    answers = _merge_answers(st.session_state.get("agent_answers") or {}, inferred)
    answers = _normalize_special_answers(answers)
    st.session_state.agent_answers = answers

    missing = _next_missing_field(answers)
    if missing:
        _append_chat("assistant", _smart_followup(answers, missing))
    else:
        _append_chat(
            "assistant",
            "Danke. Ich habe die wesentlichen Informationen zusammen. Prüfe rechts die Zusammenfassung, passe bei Bedarf etwas an und speichere dann.",
        )
    st.rerun()


# ---------- summary ----------
def _conversation_summary(ctx: Dict[str, Any]) -> Dict[str, Any]:
    answers = _normalize_special_answers(dict(st.session_state.get("agent_answers") or {}))
    user = _current_user()
    title_guess = answers.get("work_item", "") or "Neuer Arbeitseintrag"
    source_lines = []
    for m in st.session_state.get("agent_chat", []):
        role = "Agent" if m.get("role") == "assistant" else (user.get("display_name") or user.get("username") or "User")
        source_lines.append(f"{role}: {m.get('content')}")

    return {
        "work_date": date.today().isoformat(),
        "portfolio": "Default",
        "project": answers.get("project", ""),
        "theme": "General",
        "title": title_guess,
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
        "db_mode": "separate_sqlite",
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


def _render_summary_and_save(ctx: Dict[str, Any]) -> None:
    summary = _conversation_summary(ctx)

    st.subheader("Zusammenfassung")
    c1, c2 = st.columns(2)
    summary["title"] = c1.text_input("Titel", value=summary.get("title", ""), key="agent_summary_title")
    all_projects = [""] + _project_options(ctx)
    project_idx = all_projects.index(summary.get("project", "")) if summary.get("project", "") in all_projects else 0
    summary["project"] = c2.selectbox("Projekt", options=all_projects, index=project_idx, key="agent_summary_project")

    c3, c4 = st.columns(2)
    summary["status"] = c3.text_area("Status", value=summary.get("status", ""), key="agent_summary_status", height=110)
    summary["next_step"] = c4.text_area("Nächster Schritt", value=summary.get("next_step", ""), key="agent_summary_next", height=110)

    c5, c6 = st.columns(2)
    summary["blockers"] = c5.text_area("Blocker/Abhängigkeiten", value=summary.get("blockers", ""), key="agent_summary_blockers", height=110)
    summary["notes"] = c6.text_area("Zusatznotizen", value=summary.get("notes", ""), key="agent_summary_notes", height=110)

    c7, c8 = st.columns(2)
    summary["entry_kind"] = c7.selectbox(
        "Art des Eintrags",
        options=["status_update", "task", "task_and_status"],
        index=["status_update", "task", "task_and_status"].index(summary.get("entry_kind", "status_update")),
        format_func=lambda x: {
            "status_update": "Nur Status-Update",
            "task": "Neue Aufgabe",
            "task_and_status": "Beides",
        }[x],
        key="agent_entry_kind",
    )
    summary["priority"] = c8.selectbox(
        "Priorität",
        options=["", "Niedrig", "Mittel", "Hoch"],
        index=["", "Niedrig", "Mittel", "Hoch"].index(summary.get("priority", "") if summary.get("priority", "") in ["", "Niedrig", "Mittel", "Hoch"] else "Mittel"),
        key="agent_priority",
    )
    summary["due_hint"] = st.text_input("Termin / zeitlicher Hinweis", value=summary.get("due_hint", ""), key="agent_due_hint")

    ready = _conversation_complete({**summary.get("answers", {}), **summary})
    if ready:
        st.success("Die Pflichtinformationen sind vollständig.")
    else:
        missing = _next_missing_field({**summary.get("answers", {}), **summary})
        if missing:
            st.warning(f"Es fehlt noch: {_friendly_field_label(missing)}")

    cols = st.columns([1, 1, 2])
    if cols[0].button("Speichern", type="primary", use_container_width=True, disabled=not ready):
        entry_id = _store_entry(ctx, summary)
        series_id = _write_to_planning_tool(ctx, summary)
        st.session_state.agent_saved_id = {"entry_id": entry_id, "series_id": series_id}
        st.success(f"Eintrag #{entry_id} gespeichert.")
        if series_id:
            st.info(f"Zusätzlich als Aufgabe im Planungstool angelegt: {series_id}")
    if cols[1].button("Neu beginnen", use_container_width=True):
        _reset_conversation(force=True)
        st.rerun()
    cols[2].caption("Speicherung erfolgt in eine separate SQLite-Datenbank. Optional wird zusätzlich ein Task im Planungstool angelegt.")


# ---------- demo + inspection ----------
def _load_demo_conversation(ctx: Dict[str, Any]) -> None:
    user = _current_user()
    display_name = user.get("display_name") or user.get("username") or "Mitarbeiter:in"
    project = (_allowed_projects(user) or _project_options(ctx) or ["Chatcheck"])[0]
    _reset_conversation(force=True)
    demo_user_line = (
        f"Ich arbeite heute an {project}. Ich bereite die Abstimmung mit dem Kunden vor, der aktuelle Stand ist in Arbeit. "
        "Als nächstes finalisiere ich die offene Rückfragenliste. Ich warte noch auf zwei Zahlen aus dem Fachbereich. "
        "Bitte als beides speichern, Priorität hoch, idealerweise bis morgen."
    )
    st.session_state.agent_chat = [
        {"role": "assistant", "content": f"Hallo {display_name}. Ich sammle deinen Arbeitsstand für heute. Sag einfach frei oder diktiert, woran du arbeitest."},
        {"role": "user", "content": demo_user_line},
        {"role": "assistant", "content": "Danke. Ich habe fast alles. Gibt es noch etwas, das ich zusätzlich für das Tool notieren soll?"},
    ]
    answers = _infer_from_text(demo_user_line, ctx)
    answers["notes"] = "Kundenabstimmung vorbereiten; offene Zahlen aus Fachbereich ausstehend"
    answers = _normalize_special_answers(answers)
    st.session_state.agent_answers = answers
    st.session_state.agent_demo_loaded = True


def _render_entry_list(ctx: Dict[str, Any]) -> None:
    user = _current_user()
    entries = _visible_entries(ctx)
    st.subheader("Agent-Datenbank")
    if not entries:
        st.info("Keine sichtbaren Einträge vorhanden.")
        return

    for e in entries[:50]:
        subtitle = f"{e.get('work_date')} · {e.get('title')} · {e.get('project') or 'ohne Projekt'}"
        with st.expander(subtitle):
            st.write(f"**Erfasst von:** {e.get('display_name')}")
            st.write(f"**Typ:** {e.get('entry_kind')}")
            if e.get("status"):
                st.write(f"**Status:** {e.get('status')}")
            if e.get("next_step"):
                st.write(f"**Nächster Schritt:** {e.get('next_step')}")
            if e.get("blockers"):
                st.write(f"**Blocker:** {e.get('blockers')}")
            if e.get("priority"):
                st.write(f"**Priorität:** {e.get('priority')}")
            payload = e.get("payload_json") or {}
            if payload.get("due_hint"):
                st.write(f"**Termin-Hinweis:** {payload.get('due_hint')}")
            if e.get("notes"):
                st.write(f"**Notizen:** {e.get('notes')}")
            if user.get("role") == "admin":
                st.caption(f"User: {e.get('username')} · Rolle beim Schreiben: {e.get('role_at_write')}")


def _render_planning_tool_preview(ctx: Dict[str, Any]) -> None:
    st.subheader("Planungstool-Ansicht")
    items = _visible_series(ctx)
    if not items:
        st.info("Keine sichtbaren Aufgaben im Planungstool vorhanden.")
        return
    for s in items[-20:][::-1]:
        title = _norm(getattr(s, "title", "")) or "Ohne Titel"
        project = _norm(getattr(s, "project", "")) or "ohne Projekt"
        owner = _norm(getattr(s, "owner", "")) or _norm(getattr(s, "owner_id", ""))
        with st.expander(f"{title} · {project}"):
            st.write(f"**Owner:** {owner}")
            st.write(f"**Zeitraum:** {getattr(s, 'start', '')} bis {getattr(s, 'end', '')}")
            meta = dict(getattr(s, "meta", {}) or {})
            agent_meta = meta.get("agent_entry") or {}
            if agent_meta:
                st.write("**Agent-Metadaten**")
                st.json(agent_meta)


def _render_usage_notes(ctx: Dict[str, Any]) -> None:
    user = _current_user()
    role = _norm(user.get("role"))
    st.subheader("So wirkt der intelligente Dialog")
    for hint in SMART_HINTS:
        st.caption("• " + hint)
    if role == "employee":
        st.info("Du siehst nur deine eigenen Einträge und deine eigenen Aufgaben im Tool.")
    elif role == "project_lead":
        scope = ", ".join(_allowed_projects(user)) or "kein Projekt zugewiesen"
        st.info(f"Du siehst projektübergreifend nur Einträge für: {scope}")
    else:
        st.info("Du siehst alle Einträge und alle erzeugten Tool-Daten.")


def render(ctx: Dict[str, Any]) -> None:
    st.title("Agent Intake")
    st.caption("Dialogbasierte Erfassung von Projektinformationen mit separater Datenhaltung und rollenabhängiger Sicht.")

    ensure_session_auth(ctx)
    user = _current_user()
    if not user:
        st.warning("Bitte links einen Benutzernamen eingeben und anmelden.")
        return

    _bootstrap_chat()

    top = st.columns([1, 1, 1])
    if top[0].button("Echte Situation laden", use_container_width=True):
        _load_demo_conversation(ctx)
        st.rerun()
    if top[1].button("Leeren Dialog starten", use_container_width=True):
        _reset_conversation(force=True)
        st.rerun()
    top[2].caption("Der Demo-Fall ist nur eine vorgefüllte Gesprächssituation im Interface. Er nutzt den aktuell angemeldeten Benutzer.")

    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Dialog")
        for m in st.session_state.get("agent_chat", []):
            with st.chat_message("assistant" if m.get("role") == "assistant" else "user"):
                st.write(m.get("content"))
        _handle_chat_input(ctx)
        if st.session_state.get("agent_saved_id"):
            sid = st.session_state.agent_saved_id
            st.success(f"Gespeichert: Datenbank-Eintrag #{sid.get('entry_id')}")
            if sid.get("series_id"):
                st.caption(f"Zusätzlich im Tool angelegt: {sid.get('series_id')}")

    with right:
        _render_summary_and_save(ctx)
        st.markdown("---")
        _render_usage_notes(ctx)

    st.markdown("---")
    tab1, tab2 = st.tabs(["Agent-Datenbank", "Planungstool"])
    with tab1:
        _render_entry_list(ctx)
    with tab2:
        _render_planning_tool_preview(ctx)
