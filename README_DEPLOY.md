# OMG Coop – Internet Deployment

## Ziel
Diese Version ist so vorbereitet, dass sie auf einem Internet-Host (z.B. Streamlit Community Cloud, Azure App Service, Container) läuft.

## Was geändert wurde
- **Datenpfad ist konfigurierbar**: `OMG_COOP_DATA_DIR`
  - Default: gleicher Ordner wie `app.py`
  - Optional: setze `OMG_COOP_DATA_DIR` auf einen beschreibbaren Pfad (z.B. Volume), damit Daten bei Restarts persistieren.
- **Initialdaten sind leer**:
  - `data.json`: keine Tasks/Termine
  - `employees.json`: leer
  - `lists.json`: nur Portfolio `Default`

## Schnellstart lokal
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Repo (GitHub) anlegen und diesen Ordner-Inhalt committen.
2. In Streamlit Cloud: App erstellen → `app.py` auswählen.
3. Optional (empfohlen): Environment Variable setzen:
   - `OMG_COOP_DATA_DIR=/mount/data` (falls du Persistenz hast; sonst weglassen)

## Hinweise (ehrlich)
- Ohne Persistenz (Volume) sind Daten nach Restart weg. Das ist bei vielen Free-Hosts normal.
- Für echte Persistenz: Volume mounten oder auf DB (SQLite/Postgres) umbauen.
