# ui_glossary.py
from __future__ import annotations

from typing import Dict
import streamlit as st

# Central glossary for hover-tooltips and sidebar reference.
# Keep it short, practical, non-theoretical.
GLOSSARY: Dict[str, str] = {
    "Units": "Arbeitsmenge (Gewichtung). 1 Unit = eine frei definierte Menge Arbeit. Nutze es wie 'Aufwand'.",
    "Velocity": "Liefergeschwindigkeit: quittierte Units pro Arbeitstag (über ein Fenster, z.B. 10 Tage).",
    "Burndown": "Kurve der verbleibenden Units über die Zeit. Sinkt, wenn Arbeit quittiert wird.",
    "Runway": "Wie lange es mit aktueller Velocity noch dauert: Remaining Units + Forecast-Enddatum (ETA).",
    "ETA": "Voraussichtliches Enddatum basierend auf aktueller Velocity.",
    "Capacity": "Kapazität: wie viel Units pro Zeitraum realistisch geliefert werden können (z.B. Woche).",
    "Utilization": "Auslastung: Planned Units / Capacity. >1 bedeutet Überlast.",
    "Overdue": "Überfällig: Enddatum liegt in der Vergangenheit und es ist nicht DONE.",
    "Done day": "Ein Tag, an dem Arbeit quittiert wurde (DONE bestätigt).",
    "META": "Meta-Aufgabe: Organisation/Koordination statt Delivery. Optional ausblendbar (Focus mode).",
    "Blocked": "Blockiert: kann nicht weiter bearbeitet werden, weil etwas fehlt/abhängt.",
    "Planned": "Geplant: noch nicht aktiv gestartet.",
    "Active": "In Arbeit.",
    "Critical Path": "Kritischer Pfad: Kette von Abhängigkeiten, die den Endtermin bestimmt (light).",
}

def tip(term: str) -> str:
    """Returns a tiny hover icon with tooltip (HTML title)."""
    expl = GLOSSARY.get(term, "")
    if not expl:
        return ""
    # HTML is the simplest way to guarantee a real mouseover tooltip.
    safe = expl.replace("'", "&#39;")
    return f"<span title='{safe}' style='cursor:help; margin-left:6px; color:#888;'>ⓘ</span>"

def sidebar_glossary() -> None:
    with st.sidebar.expander("Glossar", expanded=False):
        st.caption("Mouseover ist im UI bei Begriffen mit ⓘ möglich. Hier alles gesammelt:")
        for k in sorted(GLOSSARY.keys()):
            st.markdown(f"**{k}** – {GLOSSARY[k]}")
