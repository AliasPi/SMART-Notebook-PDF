# SMART Notebook → PDF

Ein schneller, robuster Desktop-Konverter für **SMART Notebook (`.notebook`) → PDF** mit automatischem Zuschnitt, konstanter Seitengröße (A4 oder Original), Fortschrittsanzeigen und zweisprachiger Oberfläche (DE/EN).  
Ideal für Batch-Konvertierung ganzer Ordner – komplett **offline**.

---

## Highlights

- 🗂️ **Batch-Konvertierung** einzelner Dateien oder ganzer Ordner
- ✂️ **2-Pass AutoCrop**: niedrige DPI zur Objekterkennung, hohe DPI fürs finale Rendering
- 🧭 **Intelligente Sicherheitsränder** (Auto-Safe) gegen abgeschnittene Inhalte
- 🖨️ **Konstante Seitengröße**: A4 Hoch/Quer oder **Notebook-Seite 1** (Original)
- 🔤 **Font-Einbindung**: optionaler Fonts-Ordner für fehlende/benutzerdefinierte Schriften
- 📊 **GUI mit Fortschritt** (Gesamt & aktuelle Seite), Log und System-Check
- 🌗 **Dark Mode** & **Deutsch/English** umschaltbar
- ⚡ **Warm-Up** für schnellere erste Seite (Pillow & resvg initialisieren im Hintergrund)
- 🧵 **Sichere Nebenläufigkeit**: Hintergrund-Worker, Abbrechen jederzeit möglich
- 🧽 **Haarlinien-Inset**: reduziert feine Antialiasing-Kanten beim PDF-Einbetten

---

## Screenshot

> `<img src="./screenshot.png" alt="SMART Notebook → PDF – Screenshot" width="900">`

---

## Systemvoraussetzungen

- **Python** ≥ **3.10** (wegen `X | Y`-Typnotation)
- **Tkinter** (für die GUI; auf Linux ggf. `python3-tk`/`tk` Paket nachinstallieren)
- **Abhängigkeiten (PyPI)**  
  - `customtkinter` – moderne Tk-Oberfläche  
  - `Pillow` – Bildverarbeitung  
  - `reportlab` – PDF-Erzeugung  
  - `resvg_py` – ultrascharfes SVG-Rendering (resvg-Engine)

> **Hinweis zu `resvg_py`:** Das System-Check-Fenster zeigt, ob die Bibliothek korrekt importiert werden kann. Falls nicht, bitte die Installation/Wheels für Ihr OS prüfen.

---

## Installation

```bash
# Projekt klonen
git clone <dein-repo-url>
cd <repo-ordner>

# (Optional) virtuelles Environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Abhängigkeiten installieren
pip install --upgrade pip
pip install customtkinter Pillow reportlab resvg_py
```

---

## Start

```bash
python notebook_to_pdf_gui.py
```


---

## Bedienung in Kürze

1. **Dateien wählen** (`.notebook`) oder **Ordner wählen** (alle `.notebook` im Ordner).  
2. **Seitengröße**:  
   - **Notebook (Seite 1)** – übernimmt das Format der ersten Notebook-Seite  
   - **A4 Hoch** / **A4 Quer**  
3. **DPI**: Ausgabeauflösung (Standard: **300**).  
4. **Fonts-Ordner** (optional): Ordner mit TTF/OTF, wenn Schriften fehlen/abweichen.  
5. **Autocrop & Ränder** (siehe unten) anpassen – Standardwerte sind praxiserprobt.  
6. **🚀 Konvertieren** – Fortschritt & Log beobachten, bei Bedarf **⏹️ Abbrechen**.  
7. **🧪 System-Check**: zeigt installierte Versionen, inkl. `resvg_py`-Importtest.  

---

## Einstellungen im Detail

### Basis
- **Seitenformat**  
  - *Notebook (Seite 1):* PDF-Seiten entsprechen der ersten SVG-Seite.  
  - *A4 Hoch / Quer:* Alle Seiten werden in A4 eingepasst.
- **Ausgabe-DPI** (`dpi`): Qualität der gerenderten Seiten (Standard: **300**).
- **Fonts-Ordner** (`fonts_dir`): Wird an den Renderer übergeben, damit fehlende Schriften aufgelöst werden können.

### Autocrop & Ränder
- **Start-Puffer (px)** (`start_pad`): Ausgangspadding für die vergrößerte `viewBox` beim Patchen.  
  *Standard: 32*
- **Fallback-Sicherheitsrand (px)** (`safety`): Zusätzlicher Rand um die erkannte Bounding-Box (falls Auto-Safe aus).  
  *Standard: 24*
- **Auto-Safe (empfohlen)**: Skaliert Sicherheitsrand & Dilatation dynamisch passend zur Ziel-DPI.
- **Dilatation (px)** (`dilate`): „Aufblähen“ der Alphamaske zur robusteren Objekterkennung an Kanten.  
  *Standard: 2*
- **Alpha-Schwelle (0–255)** (`alpha_thr`): Ab wann Pixel als „sichtbar“ gelten.  
  *Standard: 2*
- **Erkennungs-DPI** (`detect_dpi`): niedrige DPI für den schnellen Erkennungspass.  
  *Standard: 72*
- **Hairline-Inset (pt)** (`inset`): Winziger Innenversatz beim Einbetten ins PDF, um feine Rand-Haarlinien zu vermeiden.  
  *Standard: 0.5*

### Fortschritt/Logging
- **Gesamtfortschritt** (alle Seiten aller Dateien) & **Seitenfortschritt** (aktuelle Datei)
- **Log anzeigen**: Schaltet die detaillierten Meldungen ein/aus.

---

## Wie es funktioniert (Kurzfassung)

1. `.notebook` wird **entpackt** (ZIP); die **SVG-Seiten** werden sortiert.  
2. **SVG-Patching**: `viewBox` wird mit Puffer vergrößert, `overflow=visible`, `href`-Pfade werden repariert.  
3. **Pass 1 (Low-DPI)**: Rendern → Alphamaske → (optional) Schwelle + Dilatation → **Bounding-Box** + Sicherheitsrand.  
4. **Pass 2 (High-DPI)**: Exaktes Rendern in Ziel-DPI → präziser Zuschnitt → **weißer Hintergrund** → **PNG in PDF eingebettet**.  
5. **Seitengröße** bleibt konstant (A4/Original), Bild wird proportional eingepasst (kein Upscaling über Zielgröße hinaus).

---

## Best-Practice-Einstellungen

- **Standard** für die meisten Fälle:  
  - DPI **300**, Erkennungs-DPI **72**, **Auto-Safe** an  
  - Start-Puffer **32 px**, Dilatation **2 px**, Alpha-Schwelle **2**, Inset **0.5 pt**
- **Sehr feines Material** (dünne Linien): Alpha-Schwelle niedrig halten (1–4), Dilatation eher klein (1–2).
- **Viele randnahe Elemente**: Auto-Safe anlassen; ggf. Start-Puffer erhöhen (64–128 px).

---

## Grenzen & Hinweise

- **Schriften**: Ohne passenden Fonts-Ordner kann das Rendering abweichen (Fallback-Fonts).  
- **Sehr große Bilder**: `Image.MAX_IMAGE_PIXELS = None` deaktiviert Pillow-Bomb-Checks, um große Seiten zu erlauben.  
  Öffnen Sie dennoch **keine untrusted Dateien** aus unsicheren Quellen.
- **Transparenzen**: Seiten werden auf **weiß flatten** – keine PDF-Transparenz.

---

## Fehlerbehebung (Troubleshooting)

- **`resvg_py` nicht importierbar**  
  - Im **System-Check** wird der Fehler angezeigt. Prüfen Sie, ob ein passendes Wheel für Ihr OS/Python vorhanden ist:  
    `pip install --force-reinstall --no-cache-dir resvg_py`
- **Fehlende/komische Schrift**  
  - Fonts-Ordner angeben (TTF/OTF). Prüfen, welche Fontnamen die SVGs referenzieren.
- **Inhalte abgeschnitten**  
  - **Auto-Safe** aktivieren oder **Start-Puffer**/**Sicherheitsrand** erhöhen.
- **Haarlinien an den Rändern**  
  - **Inset** leicht erhöhen (z. B. 0.7–1.0 pt).
- **Tkinter fehlt** (Linux)  
  - Distribution-Pakete nachinstallieren (z. B. `sudo apt install python3-tk`).

---

## Entwicklung

- **Code-Struktur** (auszugweise):
  - **Parsing/Utils**: natürliche Sortierung, SVG-Größen aus `width`/`height`/`viewBox`
  - **SVG-Patch**: `viewBox`-Aufblähung, `overflow`, `href`-Reparatur (inkl. `file://`, absolute/relative Pfade)
  - **Renderer**: `resvg_py` → PNG (RGBA) → Weiß-Flatten
  - **AutoCrop**: 2-Pass (Low-DPI Erkennung, High-DPI finaler Zuschnitt)
  - **PDF**: ReportLab, zentriertes Einpassen, optionaler **Inset**
  - **GUI**: CustomTkinter, Worker-Thread + Queue, Fortschritt/Status/Log, **DE/EN** i18n
- **Style/UX**: System-Theme, Dark Mode, sanfte Defaults, Emojis in Labels für schnelle Orientierung
- **Build (optional)**:  
  Mit PyInstaller ein Ein-Datei-Build erstellen (Beispiel, anpassen nach OS):
  ```bash
  pyinstaller --noconfirm --onefile --windowed     --name "SMART Notebook to PDF"     <Hauptskript>.py
  ```


## Dank

- https://github.com/RazrFalcon/resvg – großartige SVG-Engine  
- Die Python-Community rund um Pillow & ReportLab  
- SMART Notebook-Nutzer:innen, die das Projekt mit Beispieldateien & Feedback unterstützen
