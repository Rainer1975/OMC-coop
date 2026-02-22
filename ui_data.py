# ui_data.py
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st


# -------------------------
# Minimal atomic JSON writer (no dependency on app.py)
# -------------------------
def _acquire_lock(lock_path: Path, timeout_s: float = 5.0, poll_s: float = 0.05) -> int:
    import time as _time

    start = _time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8", errors="ignore"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            if (_time.monotonic() - start) >= timeout_s:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            _time.sleep(poll_s)


def _release_lock(fd: int, lock_path: Path) -> None:
    try:
        try:
            os.close(fd)
        finally:
            if lock_path.exists():
                lock_path.unlink()
    except Exception:
        pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = _acquire_lock(lock_path)
    try:
        fd2, tmp_path = tempfile.mkstemp(prefix="omg_", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd2, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(path))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    finally:
        _release_lock(fd, lock_path)


def _read_json_file(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


# -------------------------
# Normalizers / parsers
# -------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    v = "".join(out).strip("_")
    while "__" in v:
        v = v.replace("__", "_")
    return v or "unknown"


def _norm_str_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    for x in xs:
        s = " ".join(str(x).strip().split())
        if s and s not in out:
            out.append(s)
    return out


def _parse_series_payload(raw: Any) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Accept:
      - list[dict] (canonical core.save_state format)
      - dict with key 'series'|'data'|'items' containing list[dict]
    Return: (ok, list_of_dicts)
    """
    if isinstance(raw, list):
        items = [d for d in raw if isinstance(d, dict)]
        return True, items

    if isinstance(raw, dict):
        data = raw.get("series") or raw.get("data") or raw.get("items") or []
        if isinstance(data, list):
            items = [d for d in data if isinstance(d, dict)]
            return True, items
        return False, []

    return False, []


def _parse_employees_payload(raw: Any) -> Dict[str, Any]:
    """
    Accept:
      - ["Alice","Bob"]
      - {"employees":["Alice","Bob"]}
      - {"schema_version":1,"employees":[{...}]}
    Output (canonical):
      {"schema_version":1,"employees":[{"id":..,"display_name":..,"aliases":[..]}]}
    """
    # canonical dict
    if isinstance(raw, dict) and isinstance(raw.get("employees"), list):
        emps = raw.get("employees", [])
        if not emps:
            return {"schema_version": 1, "employees": []}
        if isinstance(emps[0], dict):
            # keep as-is (app.save_employees will clean on next sync if used)
            return {"schema_version": int(raw.get("schema_version", 1) or 1), "employees": emps}
        if isinstance(emps[0], str):
            names = _norm_str_list(emps)
            out = [{"id": _slugify(n), "display_name": n, "aliases": [n]} for n in names]
            return {"schema_version": 1, "employees": out}

    # list[str]
    if isinstance(raw, list):
        names = _norm_str_list(raw)
        out = [{"id": _slugify(n), "display_name": n, "aliases": [n]} for n in names]
        return {"schema_version": 1, "employees": out}

    # dict with employees list[str] (or empty)
    if isinstance(raw, dict):
        emps = raw.get("employees", [])
        if isinstance(emps, list) and (not emps or isinstance(emps[0], str)):
            names = _norm_str_list(emps)
            out = [{"id": _slugify(n), "display_name": n, "aliases": [n]} for n in names]
            return {"schema_version": 1, "employees": out}

    return {"schema_version": 1, "employees": []}


def _parse_lists_payload(raw: Any) -> Dict[str, List[str]]:
    """
    Accept dict with portfolios/projects/themes lists (strings).
    Return canonical dict.
    """
    def norm_list(xs: Any) -> List[str]:
        if not isinstance(xs, list):
            return []
        out: List[str] = []
        for x in xs:
            s = " ".join(str(x).strip().split())
            if s and s not in out:
                out.append(s)
        return out

    if isinstance(raw, dict):
        return {
            "portfolios": norm_list(raw.get("portfolios", [])),
            "projects": norm_list(raw.get("projects", [])),
            "themes": norm_list(raw.get("themes", [])),
        }
    return {"portfolios": [], "projects": [], "themes": []}


# -------------------------
# Session <-> Files
# -------------------------
def _write_session_to_files(ctx: dict) -> None:
    """
    Force-write everything from *current session* into json files.
    Important: do NOT import app.py here.
    """
    # series -> data.json (via core.save_state through ctx.persist)
    ctx["persist"]()

    # employees -> employees.json (via ctx.save_employees if present)
    try:
        ctx["save_employees"](st.session_state.get("employees", []))
    except Exception:
        _atomic_write_json(Path(ctx["EMP_FILE"]), {"schema_version": 1, "employees": st.session_state.get("employees", [])})

    # lists -> lists.json
    try:
        l = st.session_state.get("lists", {"portfolios": [], "projects": [], "themes": []})
        ctx["save_lists"](l.get("portfolios", []), l.get("projects", []), l.get("themes", []))
    except Exception:
        _atomic_write_json(Path(ctx["LISTS_FILE"]), st.session_state.get("lists", {"portfolios": [], "projects": [], "themes": []}))


def _reload_session_from_files(ctx: dict) -> None:
    """
    Reload session from json files, without importing app.py (no side effects).
    """
    # series
    try:
        from core import load_state  # type: ignore
        series = load_state(str(ctx["DATA_FILE"])) or []
    except Exception:
        raw = _read_json_file(Path(ctx["DATA_FILE"]))
        ok, data = _parse_series_payload(raw)
        series = []
        if ok:
            try:
                # rebuild TaskSeries objects if possible
                TaskSeries = ctx.get("TaskSeries")
                if TaskSeries is not None and hasattr(TaskSeries, "from_dict"):
                    for d in data:
                        try:
                            s = TaskSeries.from_dict(d)
                            if (getattr(s, "series_id", "") or "").strip():
                                series.append(s)
                        except Exception:
                            pass
            except Exception:
                series = []
    st.session_state.series = series

    # employees
    emp_raw = _read_json_file(Path(ctx["EMP_FILE"]))
    emp_payload = _parse_employees_payload(emp_raw)
    emps = emp_payload.get("employees", [])
    if not isinstance(emps, list):
        emps = []
    # normalize + sort
    cleaned: List[dict] = []
    seen = set()
    for e in emps:
        if not isinstance(e, dict):
            continue
        dn = str(e.get("display_name", "")).strip()
        if not dn:
            continue
        eid = str(e.get("id", "")).strip()
        if not eid:
            eid = _slugify(dn)
        if eid in seen:
            continue
        seen.add(eid)
        aliases = e.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        aliases = [str(a).strip() for a in aliases if str(a).strip()]
        if dn and dn not in aliases:
            aliases = [dn] + aliases
        cleaned.append({"id": eid, "display_name": dn, "aliases": aliases})
    cleaned.sort(key=lambda x: _norm(x.get("display_name", "")))
    st.session_state.employees = cleaned

    # lists
    lists_raw = _read_json_file(Path(ctx["LISTS_FILE"]))
    st.session_state.lists = _parse_lists_payload(lists_raw)

    # ensure data-derived lists merged
    try:
        ctx["sync_lists_from_data"]()
    except Exception:
        pass


def _global_reset_everything(ctx: dict) -> None:
    """
    Wipes:
      - data.json -> []
      - employees.json -> empty schema
      - lists.json -> default minimal lists
    Then reloads session.
    """
    _atomic_write_json(Path(ctx["DATA_FILE"]), [])
    _atomic_write_json(Path(ctx["EMP_FILE"]), {"schema_version": 1, "employees": []})
    _atomic_write_json(Path(ctx["LISTS_FILE"]), {"portfolios": ["Default"], "projects": [], "themes": ["General"]})
    _reload_session_from_files(ctx)


# -------------------------
# Export helpers
# -------------------------
def _build_export_zip_bytes(ctx: dict, include_manifest: bool = True, auto_sync: bool = True) -> bytes:
    """
    Builds a zip containing data.json, employees.json, lists.json, manifest.json.
    auto_sync=True makes export reflect current session (not stale files).
    """
    if auto_sync:
        _write_session_to_files(ctx)

    data_path = Path(ctx["DATA_FILE"])
    emp_path = Path(ctx["EMP_FILE"])
    lists_path = Path(ctx["LISTS_FILE"])

    data_raw = _read_json_file(data_path)
    emp_raw = _read_json_file(emp_path)
    lists_raw = _read_json_file(lists_path)

    ok, series_list = _parse_series_payload(data_raw)
    if not ok:
        series_list = []

    emp_payload = _parse_employees_payload(emp_raw)
    lists_payload = _parse_lists_payload(lists_raw)

    manifest = {
        "schema_version": 1,
        "export_date": datetime.now().date().isoformat(),
        "counts": {
            "series": len(series_list),
            "employees": len(emp_payload.get("employees", []) if isinstance(emp_payload.get("employees", []), list) else []),
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.json", json.dumps(series_list, ensure_ascii=False, indent=2))
        z.writestr("employees.json", json.dumps(emp_payload, ensure_ascii=False, indent=2))
        z.writestr("lists.json", json.dumps(lists_payload, ensure_ascii=False, indent=2))
        if include_manifest:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return buf.getvalue()


# -------------------------
# UI
# -------------------------
def render(ctx: dict) -> None:
    st.title("Data")

    DATA_FILE = Path(ctx["DATA_FILE"])
    EMP_FILE = Path(ctx["EMP_FILE"])
    LISTS_FILE = Path(ctx["LISTS_FILE"])

    st.caption(
        "Hier verwaltest du Export/Import/Synchronisation. "
        "Wichtig: Dieses Modul importiert NICHT app.py (keine Side-Effects, kein Duplicate-Key)."
    )

    # --- Quick status
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Series (Session)", len(st.session_state.get("series", [])))
    with c2:
        st.metric("Employees (Session)", len(st.session_state.get("employees", [])))
    with c3:
        l = st.session_state.get("lists", {"portfolios": [], "projects": [], "themes": []})
        st.metric("Lists (Portfolios/Projects/Themes)", f"{len(l.get('portfolios', []))}/{len(l.get('projects', []))}/{len(l.get('themes', []))}")

    with st.expander("🔎 Debug: aktuelle Files (Counts)", expanded=False):
        data_raw = _read_json_file(DATA_FILE)
        emp_raw = _read_json_file(EMP_FILE)
        lists_raw = _read_json_file(LISTS_FILE)

        ok, series_list = _parse_series_payload(data_raw)
        emp_payload = _parse_employees_payload(emp_raw)
        lists_payload = _parse_lists_payload(lists_raw)

        st.write(
            {
                "DATA_FILE": str(DATA_FILE),
                "EMP_FILE": str(EMP_FILE),
                "LISTS_FILE": str(LISTS_FILE),
                "file_counts": {
                    "series": len(series_list) if ok else "unreadable",
                    "employees": len(emp_payload.get("employees", [])) if isinstance(emp_payload.get("employees", []), list) else "unreadable",
                    "portfolios": len(lists_payload.get("portfolios", [])),
                    "projects": len(lists_payload.get("projects", [])),
                    "themes": len(lists_payload.get("themes", [])),
                },
            }
        )

    st.markdown("---")

    # --- Sync
    with st.container(border=True):
        st.subheader("🔄 Sync")
        st.caption("Schreibt Session → Files (damit Export/Import konsistent sind).")

        if st.button("Sync session → files", key="data_sync_btn", width="stretch"):
            try:
                _write_session_to_files(ctx)
                st.success("Sync OK: Session wurde in JSON Files geschrieben.")
            except Exception as e:
                st.error(f"Sync fehlgeschlagen: {e}")

    st.markdown("---")

    # --- Export
    with st.container(border=True):
        st.subheader("⬇️ Export")
        st.caption("Erzeugt ein ZIP mit data.json / employees.json / lists.json / manifest.json.")

        auto_sync = st.checkbox("Vor dem Export automatisch: Sync session → files", value=True, key="export_auto_sync")
        zbytes = _build_export_zip_bytes(ctx, include_manifest=True, auto_sync=auto_sync)
        st.download_button(
            "Download backup ZIP",
            data=zbytes,
            file_name="omg_coop_backup.zip",
            mime="application/zip",
            key="export_zip_download_btn",
            width="stretch",
        )

    st.markdown("---")

    # --- Import
    with st.container(border=True):
        st.subheader("⬆️ Import")
        st.caption("Import überschreibt data.json / employees.json / lists.json vollständig und lädt danach Session neu.")

        up = st.file_uploader("Upload backup ZIP", type=["zip"], key="import_zip_uploader")
        must_confirm = st.checkbox(
            "Ich bestätige: Import überschreibt data.json / employees.json / lists.json vollständig.",
            key="import_zip_confirm",
        )

        if st.button(
            "Import now (overwrite current data)",
            key="import_zip_btn",
            disabled=(up is None or not must_confirm),
            width="stretch",
        ):
            try:
                b = up.getvalue() if up is not None else b""
                zmem = io.BytesIO(b)

                with zipfile.ZipFile(zmem, "r") as z:
                    names = set(z.namelist())
                    required = {"data.json", "employees.json", "lists.json"}
                    missing = sorted(list(required - names))
                    if missing:
                        st.error(f"ZIP fehlt: {', '.join(missing)}")
                        st.stop()

                    data_raw = json.loads(z.read("data.json").decode("utf-8", errors="replace"))
                    emp_raw = json.loads(z.read("employees.json").decode("utf-8", errors="replace"))
                    lists_raw = json.loads(z.read("lists.json").decode("utf-8", errors="replace"))

                ok, series_list = _parse_series_payload(data_raw)
                if not ok:
                    st.error("data.json muss eine LISTE (Series) sein ODER ein Dict mit key 'series' (Liste).")
                    st.stop()

                emp_payload = _parse_employees_payload(emp_raw)
                lists_payload = _parse_lists_payload(lists_raw)

                # write canonical
                _atomic_write_json(DATA_FILE, series_list)     # list[dict]
                _atomic_write_json(EMP_FILE, emp_payload)      # dict schema
                _atomic_write_json(LISTS_FILE, lists_payload)  # dict

                # reload into session
                _reload_session_from_files(ctx)

                st.success(
                    f"Import OK. Series: {len(st.session_state.get('series', []))}, "
                    f"Employees: {len(st.session_state.get('employees', []))}."
                )
                st.rerun()

            except zipfile.BadZipFile:
                st.error("Ungültiges ZIP.")
            except Exception as e:
                st.error(f"Import fehlgeschlagen: {e}")

    st.markdown("---")

    # --- Global reset (only here, as requested)
    with st.container(border=True):
        st.subheader("💣 Global Reset (alles löschen)")
        st.caption("Setzt ALLES zurück: tasks/appointments, employees, lists. Nur hier im Data-Modul.")

        confirm = st.checkbox("Ich bestätige: Global Reset löscht wirklich alles.", key="global_reset_confirm")
        if st.button("GLOBAL RESET NOW", key="global_reset_btn", disabled=not confirm, width="stretch"):
            try:
                _global_reset_everything(ctx)
                st.success("Global Reset OK. Alles gelöscht, Session neu geladen.")
                st.rerun()
            except Exception as e:
                st.error(f"Global Reset fehlgeschlagen: {e}")
