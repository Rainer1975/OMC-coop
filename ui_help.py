# ui_help.py
from __future__ import annotations

import streamlit as st


# -----------------------------
# Content (Single Source of Truth)
# -----------------------------
HELP_SECTIONS = [
    ("wozu", "Wozu dient das Tool?"),
    ("grundkonzepte", "Grundkonzepte (Plan vs. Realität)"),
    ("nav", "Menüpunkte: wozu sind sie da?"),
    ("home", "Start › Home"),
    ("inbox", "Dateneingabe › Inbox"),
    ("today", "Dateneingabe › Today"),
    ("detail", "Dateneingabe › Detail"),
    ("kanban", "Reporting › Kanban"),
    ("gantt", "Reporting › Gantt"),
    ("burndown", "Reporting › Burndown"),
    ("dashboard", "Reporting › Dashboard"),
    ("meta", "Reporting › Meta"),
    ("admin", "Wartung › Admin"),
    ("data", "Wartung › Data"),
    ("faq", "FAQ (typische Fragen)"),
    ("glossar", "Glossar"),
    ("pflege", "Pflege-Regel: Anleitung immer aktuell halten"),
]

SECTION_INDEX = {k: i for i, (k, _) in enumerate(HELP_SECTIONS)}


def _h2(title: str, anchor: str) -> None:
    # Anchors via HTML (Streamlit has no native anchor navigation)
    st.markdown(f"<a id='{anchor}'></a>", unsafe_allow_html=True)
    st.subheader(title)


def _toc() -> None:
    st.markdown("## Inhaltsverzeichnis")
    for key, title in HELP_SECTIONS:
        st.markdown(f"- **{title}**  _(Anker: `{key}`)_")


def _render_full(target: str | None) -> None:
    st.title("Hilfe – OMG Coop (Vollständige Anleitung)")
    st.caption("Kontextsensitiv: wenn du aus einem Submenü kommst, wird der passende Abschnitt oben priorisiert.")

    # If target is given, show that section first
    if target and target in SECTION_INDEX:
        st.info(f"Direkt geöffnet: **{dict(HELP_SECTIONS).get(target)}**  (Kontext: `{target}`)")
        _render_section(target)
        st.divider()
        st.markdown("### Danach folgt die komplette Anleitung:")
        st.divider()

    _toc()
    st.divider()

    for key, _ in HELP_SECTIONS:
        _render_section(key)
        st.divider()


def _render_beginner(target: str | None) -> None:
    st.title("🧑‍🎓 Anfänger-Guide – 15 Minuten")
    st.caption("Ziel: du kannst danach Tasks planen, Dependencies setzen, Done korrekt tracken und Gantt/Burndown lesen.")

    # Optional: show context note
    if target and target in SECTION_INDEX:
        st.info(f"Du kommst aus: `{target}`. Ich verweise im Guide auf den passenden Bereich, wenn relevant.")

    st.markdown("## 0–2 Minuten: Was ist dieses Tool?")
    st.markdown(
        """
- Ein PM-Tool, das **Planung** und **Realität** strikt trennt.
- Planung: **Start/Ende + Abhängigkeiten**.
- Realität: **done_days** (= an welchen Tagen wurde wirklich gearbeitet).
- Dadurch sind **Burndown** und **Verzug** nicht geschätzt, sondern aus Daten ableitbar.
"""
    )

    st.markdown("## 2–6 Minuten: Schnell einen Task anlegen (Quick add)")
    st.markdown(
        """
1) Sidebar → **Quick add (task)**  
2) Pflichtfelder: **Title, Portfolio, Project, Theme, Owner, Start, End**  
3) „Create“  
Ergebnis: Task existiert und ist planbar.
"""
    )

    st.markdown("## 6–9 Minuten: Dependencies setzen (das ist der Kern)")
    st.markdown(
        """
1) Navigation → **Dateneingabe › Detail**  
2) Task öffnen  
3) Bereich **Dependencies (task → task)**  
4) Vorgänger wählen → **Save dependencies**  
Wichtig:
- Mehrere Vorgänger sind erlaubt.
- Zyklen sind verboten (A→B→A).
"""
    )

    st.markdown("## 9–12 Minuten: Realität erfassen (Done richtig nutzen)")
    st.markdown(
        """
1) Navigation → **Dateneingabe › Today**  
2) Für heute erledigte Arbeit **DONE** markieren  
Wichtig:
- DONE heißt: **an diesem Tag wurde daran gearbeitet**.
- DONE heißt **nicht** automatisch „Task fertig“.
"""
    )

    st.markdown("## 12–14 Minuten: Gantt lesen (Pfeile & kritischer Pfad)")
    st.markdown(
        """
1) Navigation → **Reporting › Gantt**  
2) Balken = Zeitraum  
3) Pfeile = Dependencies  
4) Kritischer Pfad = längste Abhängigkeitskette (bestimmt Enddatum)
"""
    )

    st.markdown("## 14–15 Minuten: Burndown pro Task (Plan vs Ist)")
    st.markdown(
        """
1) Navigation → **Reporting › Burndown**  
2) Pro Task:
- Ideal (berechnet) vs. Actual (done_days)
Interpretation:
- Actual über Ideal → Verzug
- Flatline → es wurde nicht getrackt/gebaut
"""
    )

    st.divider()
    st.markdown("## Nächste Schritte (wenn du mehr willst)")
    st.markdown(
        """
- Parts nutzen, wenn ein Task intern Phasen hat (gewichteter Fortschritt).
- META nutzen, um Koordinationsaufwand sichtbar zu halten.
- Data-Seite nutzen, wenn etwas „komisch“ wirkt (Debug/Truth).
"""
    )


def _render_section(key: str) -> None:
    if key == "wozu":
        _h2("Wozu dient das Tool?", "wozu")
        st.markdown(
            """
Dieses Tool ist ein **leichtgewichtiges Enterprise-PM-System**, das:
- Aufgaben **zeitlich** plant,
- **Abhängigkeiten** sichtbar macht,
- Verantwortlichkeit über **Owner** klärt,
- und Fortschritt als **Realität (done_days)** misst.

Kernprinzip:
> **Planung** (Start/Ende/Dependencies) und **Realität** (done_days) werden bewusst getrennt.
"""
        )

    elif key == "grundkonzepte":
        _h2("Grundkonzepte (Plan vs. Realität)", "grundkonzepte")
        st.markdown(
            """
**Planung**
- Startdatum / Enddatum
- Dependencies (Vorgänger)

**Realität**
- done_days: konkrete Tage, an denen am Task gearbeitet wurde

**DONE ist ein Ereignis, kein Magie-Status**
- DONE = „an diesem Tag gearbeitet“
- Nicht automatisch: „Task abgeschlossen“
"""
        )

    elif key == "nav":
        _h2("Menüpunkte: wozu sind sie da?", "nav")
        st.markdown(
            """
- **Home**: Überblick
- **Inbox**: Sammeln (ungeplant)
- **Today**: Realität erfassen (DONE)
- **Detail**: Task/Termin bearbeiten + Dependencies
- **Kanban**: Statussicht (PLANNED/ACTIVE/BLOCKED/DONE/CANCELLED)
- **Gantt**: Zeitplan + Pfeile + kritischer Pfad
- **Burndown**: Plan vs Ist pro Task
- **Dashboard**: Aggregation
- **Meta**: META-Tasks
- **Admin**: Pflege
- **Data**: Debug/Truth
"""
        )

    elif key == "home":
        _h2("Start › Home", "home")
        st.markdown(
            """
**Wozu?** Schnell Orientierung.  
**Was tun?** Überblick, Einstieg in die Arbeit.
"""
        )

    elif key == "inbox":
        _h2("Dateneingabe › Inbox", "inbox")
        st.markdown(
            """
**Wozu?** Aufgaben ungeplant erfassen – ohne sofort zu strukturieren.  
**Was tun?** Sammeln, später im Detail sauber machen.  
**Typischer Fehler:** Inbox als Planungsort missbrauchen.
"""
        )

    elif key == "today":
        _h2("Dateneingabe › Today", "today")
        st.markdown(
            """
**Wozu?** Realität erfassen.  
**Was tun?** DONE für heute setzen (und ggf. Vergangenheit korrigieren).  
**Wichtig:** Future-DONE ist verboten.
"""
        )

    elif key == "detail":
        _h2("Dateneingabe › Detail", "detail")
        st.markdown(
            """
**Wozu?** Zentrale Wahrheit eines Tasks/Termins.  
**Was tun?**
- Title / Project / Theme / Owner
- Start / End
- **Dependencies setzen**
- Parts pflegen (wenn nötig)
- DONE-Tage korrigieren

**Dependencies**
- mehrere Vorgänger erlaubt
- Zyklen verboten
"""
        )

    elif key == "kanban":
        _h2("Reporting › Kanban", "kanban")
        st.markdown(
            """
**Wozu?** Statussicht (Arbeitszustand).  
**Was tun?** Tasks zwischen States bewegen.

**Wichtig:** Status ersetzt keine Zeitplanung.
"""
        )

    elif key == "gantt":
        _h2("Reporting › Gantt", "gantt")
        st.markdown(
            """
**Wozu?** Zeitplan + Abhängigkeiten sichtbar.  
**Was sehen?**
- Balken = Zeiträume
- Pfeile = Dependencies
- Rot = kritischer Pfad

**Wenn keine Pfeile erscheinen:** Dependencies fehlen/werden nicht gespeichert oder Filter blenden Tasks aus.
"""
        )

    elif key == "burndown":
        _h2("Reporting › Burndown", "burndown")
        st.markdown(
            """
**Wozu?** Plan vs Ist pro Task.  
**Darstellung**
- Ideal: linear geplant
- Actual: aus done_days abgeleitet

**Flatline = keine Realität erfasst**.
"""
        )

    elif key == "dashboard":
        _h2("Reporting › Dashboard", "dashboard")
        st.markdown(
            """
**Wozu?** Aggregierter Überblick.  
**Was tun?** Trends und Engpässe erkennen.
"""
        )

    elif key == "meta":
        _h2("Reporting › Meta", "meta")
        st.markdown(
            """
**Wozu?** Koordinationsarbeit sichtbar machen (META).  
**Hinweis:** Focus-Mode kann META ausblenden.
"""
        )

    elif key == "admin":
        _h2("Wartung › Admin", "admin")
        st.markdown(
            """
**Wozu?** Systempflege.  
Typisch: Mitarbeiter/Owner, Struktur-Checks.
"""
        )

    elif key == "data":
        _h2("Wartung › Data", "data")
        st.markdown(
            """
**Wozu?** Debug/Truth.  
Wenn etwas „komisch“ ist: hier prüfen, ob Daten wirklich so sind.
"""
        )

    elif key == "faq":
        _h2("FAQ (typische Fragen)", "faq")
        st.markdown(
            """
**Warum keine Pfeile im Gantt?**  
→ Dependencies fehlen / nicht gespeichert / Vorgänger gefiltert.

**Warum Burndown flach?**  
→ done_days werden nicht gesetzt.

**Warum Task nicht „fertig“, obwohl Enddatum vorbei?**  
→ Enddatum ist Planung, Realität kommt aus done_days.

**Warum trennt ihr Plan und Ist?**  
→ Damit du nicht in Status-Märchen landest.
"""
        )

    elif key == "glossar":
        _h2("Glossar", "glossar")
        st.markdown(
            """
**Task**: zeitlich geplanter Arbeitsblock  
**Appointment**: fixer Termin ohne Fortschritt  
**Owner**: verantwortliche Person  
**Portfolio**: übergeordnete Klammer  
**Project**: Vorhaben  
**Theme**: Kategorie  
**Dependency**: Abhängigkeit zwischen Tasks  
**Critical Path**: längste Abhängigkeitskette  
**done_days**: Tage realer Arbeit  
**Burndown**: Remaining über Zeit (Plan vs Ist)  
**META**: Koordination/Overhead
"""
        )

    elif key == "pflege":
        _h2("Pflege-Regel: Anleitung immer aktuell halten", "pflege")
        st.markdown(
            """
**Verbindliche Regel**
- Jede funktionale Änderung im Tool → Anpassung der passenden Abschnitte hier.
- Neue Begriffe → ins Glossar.
- Neue Menüpunkte → eigene Sektion + Mapping (Kontext-Link).

Das hier ist die **Single Source of Truth** – nicht irgendwelche Chat-Nachrichten.
"""
        )

    else:
        _h2(f"{key}", key)
        st.markdown("Noch nicht dokumentiert.")


# -----------------------------
# UI Entry
# -----------------------------
def render(ctx: dict) -> None:
    """
    ctx expected:
      - help_mode: "full" | "beginner"
      - help_target: section key (optional)
    """
    mode = (ctx.get("help_mode") or "full").strip().lower()
    target = ctx.get("help_target")

    # Small top controls
    top1, top2, top3 = st.columns([2, 2, 6])
    if top1.button("❓ Vollhilfe", key="help_switch_full"):
        st.session_state.help_mode = "full"
        st.rerun()
    if top2.button("🧑‍🎓 Anfänger (15 Min)", key="help_switch_beginner"):
        st.session_state.help_mode = "beginner"
        st.rerun()
    top3.caption("Tipp: Kontexthilfe wird automatisch anhand der aktuellen Seite gesetzt.")

    st.divider()

    if mode == "beginner":
        _render_beginner(target)
    else:
        _render_full(target)
