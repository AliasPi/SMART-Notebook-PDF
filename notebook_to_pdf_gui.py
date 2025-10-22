# -*- coding: utf-8 -*-
import os, re, sys, math, zipfile, tempfile, threading, queue, time, glob
from pathlib import Path
from io import BytesIO
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageFilter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import white
from reportlab.lib.utils import ImageReader
Image.MAX_IMAGE_PIXELS = None


# =========================================================
#                    i18n (DE / EN)
# =========================================================
LANG = "DE"
T = {
    "DE": {
        "app_title": "SMART Notebook → PDF",
        "header": "SMART .notebook → PDF",
        "darkmode": "Dark Mode",
        "overall": "Gesamt",
        "file": "Datei",
        "page": "Seite",
        "ready": "Bereit",
        "input": "Eingabe",
        "entry_hint": "Pfad(e) zu .notebook-Dateien (mit ';' trennen)",
        "choose_files": "📂 Dateien wählen",
        "choose_folder": "🗂️ Ordner wählen",
        "settings": "Einstellungen",
        "page_size": "Seitenformat:",
        "dpi": "Ausgabe-DPI:",
        "fonts_dir": "Fonts-Ordner (optional):",
        "autocrop_group": "Autocrop & Ränder",
        "start_pad": "Start-Puffer (px):",
        "safety": "Fallback Sicherheitsrand (px):",
        "auto_safe": "Auto-Safe (empfohlen)",
        "dilation": "Dilatation (px):",
        "alpha_thr": "Alpha-Schwelle (0–255):",
        "detect_dpi": "Low-DPI Erkennung:",
        "inset": "Hairline-Inset (pt):",
        "convert": "🚀 Konvertieren",
        "cancel": "⏹️ Abbrechen",
        "system_check": "🧪 System-Check",
        "log_toggle": "Details (Log) anzeigen",
        "pick_files_title": "Wähle .notebook-Dateien (Mehrfachauswahl möglich)",
        "pick_folder_title": "Ordner mit .notebook-Dateien auswählen",
        "no_files_in_folder": "Im gewählten Ordner wurden keine .notebook-Dateien gefunden.",
        "warn_select_first": "Bitte zuerst Datei(en) oder Ordner wählen.",
        "warn_no_valid": "Keine gültigen Pfade.",
        "info_done": "Erfolg",
        "warn": "Hinweis",
        "err": "Fehler",
        "aborting": "Abbruch angefordert",
        "aborted": "Abgebrochen",
        "starting": "Starte …",
        "language": "Sprache / Language:",
        "nb_size": "Notebook (Seite 1)",
        "a4p": "A4 Hoch",
        "a4l": "A4 Quer",
    },
    "EN": {
        "app_title": "SMART Notebook → PDF",
        "header": "SMART .notebook → PDF",
        "darkmode": "Dark Mode",
        "overall": "Overall",
        "file": "File",
        "page": "Page",
        "ready": "Ready",
        "input": "Input",
        "entry_hint": "Path(s) to .notebook files (separate with ';')",
        "choose_files": "📂 Choose files",
        "choose_folder": "🗂️ Choose folder",
        "settings": "Settings",
        "page_size": "Page size:",
        "dpi": "Output DPI:",
        "fonts_dir": "Fonts folder (optional):",
        "autocrop_group": "Autocrop & Margins",
        "start_pad": "Start padding (px):",
        "safety": "Fallback safety (px):",
        "auto_safe": "Auto-safe (recommended)",
        "dilation": "Dilation (px):",
        "alpha_thr": "Alpha threshold (0–255):",
        "detect_dpi": "Detection DPI:",
        "inset": "Hairline inset (pt):",
        "convert": "🚀 Convert",
        "cancel": "⏹️ Cancel",
        "system_check": "🧪 System check",
        "log_toggle": "Show details (log)",
        "pick_files_title": "Choose .notebook files (multi-select)",
        "pick_folder_title": "Choose folder with .notebook files",
        "no_files_in_folder": "No .notebook files found in this folder.",
        "warn_select_first": "Please choose file(s) or a folder first.",
        "warn_no_valid": "No valid paths.",
        "info_done": "Success",
        "warn": "Notice",
        "err": "Error",
        "aborting": "Abort requested",
        "aborted": "Aborted",
        "starting": "Starting …",
        "language": "Sprache / Language:",
        "nb_size": "Notebook (page 1)",
        "a4p": "A4 portrait",
        "a4l": "A4 landscape",
    }
}
def tr(key): return T[LANG][key]


# =========================================================
#                     Utility & Parsing
# =========================================================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _parse_length(s: str):
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z%]*)\s*$", s or "")
    return (float(m.group(1)), m.group(2).lower()) if m else (None, "")

def _to_inches(val: float, unit: str):
    if unit in ("", "px"): return val / 96.0
    if unit == "in": return val
    if unit == "cm": return val / 2.54
    if unit == "mm": return val / 25.4
    if unit == "pt": return val / 72.0
    if unit == "pc": return (val * 12.0) / 72.0
    return None

_SIZE_CACHE: dict[Path, tuple[float, float, float, float]] = {}
def _read_svg_size(svg_path: Path):
    if svg_path in _SIZE_CACHE:
        return _SIZE_CACHE[svg_path]
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        w_px, h_px = 1280.0, 720.0
        res = (w_px, h_px, w_px/96.0, h_px/96.0)
        _SIZE_CACHE[svg_path] = res
        return res

    w_attr = root.get("width"); h_attr = root.get("height")
    vb_attr = root.get("viewBox") or root.get("viewbox")

    w_px = h_px = w_in = h_in = None
    if w_attr and h_attr and not w_attr.endswith("%") and not h_attr.endswith("%"):
        w_val, w_unit = _parse_length(w_attr); h_val, h_unit = _parse_length(h_attr)
        if w_val is not None and h_val is not None:
            w_in = _to_inches(w_val, w_unit); h_in = _to_inches(h_val, h_unit)
            w_px = w_val if w_unit in ("", "px") else (w_in * 96.0 if w_in else None)
            h_px = h_val if h_unit in ("", "px") else (h_in * 96.0 if h_in else None)

    if (w_px is None or h_px is None) and vb_attr:
        try:
            _, _, vbw, vbh = [float(x) for x in re.split(r"[,\s]+", vb_attr.strip()) if x]
            w_px = w_px or vbw; h_px = h_px or vbh
            w_in = w_in or (vbw / 96.0); h_in = h_in or (vbh / 96.0)
        except Exception:
            pass

    if w_px is None or h_px is None: w_px, h_px = 1280.0, 720.0
    if w_in is None or h_in is None: w_in, h_in = w_px/96.0, h_px/96.0
    res = (w_px, h_px, w_in, h_in)
    _SIZE_CACHE[svg_path] = res
    return res


# =========================================================
#               SVG patch: viewBox & href Fix
# =========================================================
XLINK = "{http://www.w3.org/1999/xlink}"

def _fix_href_value(raw: str, svg_dir: Path, root_dir: Path) -> str | None:
    if not raw: return None
    s = unquote(raw.strip())
    if s.startswith(("data:", "#", "http://", "https://")):
        return raw
    if s.startswith("file://"):
        p = Path(urlparse(s).path)
        if p.exists():
            try: return os.path.relpath(p, start=svg_dir)
            except Exception: return str(p)
    if s.startswith("/"):
        cand = root_dir / s[1:]
        if cand.exists(): return os.path.relpath(cand, start=svg_dir)
    cand = (svg_dir / s).resolve()
    if cand.exists(): return os.path.relpath(cand, start=svg_dir)
    cand2 = (root_dir / s.lstrip("./")).resolve()
    if cand2.exists(): return os.path.relpath(cand2, start=svg_dir)
    return raw

def patch_svg(svg_in: Path, svg_out: Path, pad_px: int, root_dir: Path):
    """viewBox aufblasen + overflow='visible' + href reparieren."""
    try:
        tree = ET.parse(svg_in); root = tree.getroot()
    except Exception:
        svg_out.write_bytes(svg_in.read_bytes()); return

    vb_attr = root.get("viewBox") or root.get("viewbox")
    if vb_attr:
        try:
            minx, miny, vbw, vbh = [float(x) for x in re.split(r"[,\s]+", vb_attr.strip()) if x]
        except Exception:
            minx = miny = 0.0
            w_px, h_px, *_ = _read_svg_size(svg_in)
            vbw, vbh = float(w_px), float(h_px)
    else:
        w_px, h_px, *_ = _read_svg_size(svg_in)
        minx, miny, vbw, vbh = 0.0, 0.0, float(w_px), float(h_px)

    pad = float(pad_px)
    root.set("viewBox", f"{minx - pad:g} {miny - pad:g} {vbw + 2*pad:g} {vbh + 2*pad:g}")
    root.set("overflow", "visible")

    svg_dir = svg_in.parent
    root.set("xmlns:xlink", "http://www.w3.org/1999/xlink")
    for el in root.iter():
        for attr in ("href", f"{XLINK}href"):
            if attr in el.attrib:
                fixed = _fix_href_value(el.attrib.get(attr), svg_dir, root_dir)
                if fixed is not None: el.set(attr, fixed)

    tree.write(svg_out, encoding="utf-8", xml_declaration=True)


# =========================================================
#                   resvg helpers & warm-up
# =========================================================
def ensure_resvg():
    try:
        import resvg_py  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)

def _render_rgba(svg_path: Path, out_w: int, out_h: int, dpi_out: int,
                 resources_dir: Path, fonts_dir: str | None) -> Image.Image:
    import resvg_py
    png_bytes = bytes(resvg_py.svg_to_bytes(
        svg_path=str(svg_path),
        resources_dir=str(resources_dir),
        width=int(out_w), height=int(out_h), dpi=int(dpi_out),
        font_dirs=[fonts_dir] if fonts_dir else None,
    ))
    return Image.open(BytesIO(png_bytes)).convert("RGBA")

def _flatten_white(rgba: Image.Image) -> Image.Image:
    rgb = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba, mask=rgba.split()[3])
    return rgb

def _warmup_pillow():
    # PNG/JPEG Codecs laden, damit die erste Speicherung nicht bremst
    img = Image.new("RGBA", (8, 8), (255, 255, 255, 0))
    _ = img.tobytes()
    buf = BytesIO()
    img.save(buf, format="PNG"); buf.seek(0)
    Image.open(buf).load()

def _warmup_resvg(fonts_dir: str | None):
    """Einmaliger Kaltstart (Font-Scan & Renderer)."""
    try:
        import resvg_py
        tiny = "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'><text x='1' y='9'>x</text></svg>"
        resvg_py.svg_to_bytes(svg_string=tiny, width=10, height=10, dpi=72,
                              font_dirs=[fonts_dir] if fonts_dir else None)
    except Exception:
        pass

def warmup_async(get_fonts_dir_cb):
    def _run():
        try:
            _warmup_pillow()
            _warmup_resvg(get_fonts_dir_cb())
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


# =========================================================
#           2-Pass AutoCrop (schnell & robust)
# =========================================================
def detect_bbox_fast(svg_orig: Path, tmp_dir: Path, root_dir: Path, fonts_dir: str | None,
                     pad_px_init: int, dpi_detect: int, dpi_out: int,
                     safety_px_manual: int, dilate_px: int, alpha_thr: int,
                     auto_safe: bool, max_pad_px: int = 4096):
    """Low-DPI-Pass: Crop-BBox der gepatchten viewBox + Auto-Safe."""
    w0, h0, *_ = _read_svg_size(svg_orig)
    pad = int(pad_px_init)
    patched = tmp_dir / (svg_orig.stem + "_det.svg")

    ratio = max(1.0, dpi_out / float(max(1, dpi_detect)))
    base_dim = max(w0 + 2*pad, h0 + 2*pad) * (dpi_detect/96.0)
    overscan = int(max(24, round(0.03 * base_dim)))

    if auto_safe:
        auto_safety = max(overscan, int(round(48 * ratio)))
        auto_dilate = max(dilate_px, int(round(2 * ratio)))
    else:
        auto_safety = int(safety_px_manual)
        auto_dilate = int(dilate_px)

    while True:
        patch_svg(svg_orig, patched, pad, root_dir=root_dir)
        scale = max(dpi_detect, 1) / 96.0
        out_w = max(1, int(math.ceil((w0 + 2*pad) * scale)))
        out_h = max(1, int(math.ceil((h0 + 2*pad) * scale)))

        rgba = _render_rgba(patched, out_w, out_h, dpi_detect, resources_dir=svg_orig.parent, fonts_dir=fonts_dir)
        a = rgba.split()[3]
        if alpha_thr > 0:
            a = a.point(lambda x, t=alpha_thr: 255 if x >= t else 0, mode='L')
        if auto_dilate > 0:
            a = a.filter(ImageFilter.MaxFilter(2*auto_dilate + 1))

        bbox = a.getbbox()
        if not bbox:
            crop = (0, 0, out_w, out_h)
        else:
            L, T, R, B = bbox; s = auto_safety
            crop = (max(0, L - s), max(0, T - s), min(out_w, R + s), min(out_h, B + s))

        min_gap = max(12, int(round(8 * ratio)))
        gapL = crop[0]; gapT = crop[1]; gapR = (out_w - 1) - crop[2]; gapB = (out_h - 1) - crop[3]
        touches = min(gapL, gapT, gapR, gapB) < min_gap
        if not touches or pad >= max_pad_px:
            return pad, crop
        pad = min(max_pad_px, pad * 2)

def render_final_cropped(svg_orig: Path, tmp_dir: Path, root_dir: Path, fonts_dir: str | None,
                         pad_px_final: int, crop_detect: tuple, dpi_detect: int, dpi_out: int) -> Image.Image:
    """High-DPI-Pass: exakt zuschneiden (Skalierung der Detect-Koordinaten)."""
    w0, h0, *_ = _read_svg_size(svg_orig)
    patched = tmp_dir / (svg_orig.stem + "_final.svg")
    patch_svg(svg_orig, patched, pad_px_final, root_dir=root_dir)

    scale_final = max(dpi_out, 1) / 96.0
    out_w = max(1, int(math.ceil((w0 + 2*pad_px_final) * scale_final)))
    out_h = max(1, int(math.ceil((h0 + 2*pad_px_final) * scale_final)))
    rgba = _render_rgba(patched, out_w, out_h, dpi_out, resources_dir=svg_orig.parent, fonts_dir=fonts_dir)

    Ld, Td, Rd, Bd = crop_detect
    s = dpi_out / float(max(1, dpi_detect))
    L = max(0, int(round(Ld * s))); T = max(0, int(round(Td * s)))
    R = min(out_w, int(round(Rd * s))); B = min(out_h, int(round(Bd * s)))
    if R <= L: R = min(out_w, L + 1)
    if B <= T: B = min(out_h, T + 1)

    cropped = rgba.crop((L, T, R, B))
    return _flatten_white(cropped)


# =========================================================
#            Notebook → PDF (eine Datei)
# =========================================================
def convert_notebook(nb_path: Path, dpi_out: int, fonts_dir: str | None, page_mode: str,
                     start_pad_px: int, safety_px: int, dilate_px: int, alpha_thr: int,
                     inset_pt: float, dpi_detect: int, auto_safe: bool,
                     qmsg: queue.Queue, cancel_evt: threading.Event,
                     file_index: int, file_count: int,
                     pages_done_base: int, pages_total_all: int):
    out_pdf = nb_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(prefix="nb2pdf_") as tmpdir:
        tmp = Path(tempfile.mktemp(dir=tmpdir))  # unique subdir
        tmp.mkdir(parents=True, exist_ok=True)
        patched_dir = tmp / "patched"; patched_dir.mkdir(parents=True, exist_ok=True)

        qmsg.put(("log", f"📦 {tr('file')} {file_index}/{file_count}: {nb_path.name}"))
        with zipfile.ZipFile(nb_path, "r") as zf:
            zf.extractall(tmp)
            svg_rel = [n for n in zf.namelist() if n.lower().endswith(".svg")]
            if not svg_rel:
                qmsg.put(("warn", f"{tr('warn')}: Keine SVG-Seiten in {nb_path.name}.")); return 0, None
            svg_rel.sort(key=natural_key)

        total_pages_file = len(svg_rel)
        qmsg.put(("file_begin", (file_index, file_count, nb_path.name, total_pages_file)))

        # Ziel-Seitengröße (konstant)
        first_w_px, first_h_px, first_w_in, first_h_in = _read_svg_size(Path(tmp / svg_rel[0]))
        if page_mode == "NOTEBOOK": target_w_in, target_h_in = first_w_in, first_h_in
        elif page_mode == "A4_P":  target_w_in, target_h_in = 8.27, 11.69
        else:                      target_w_in, target_h_in = 11.69, 8.27
        target_w_pt, target_h_pt = target_w_in * 72.0, target_h_in * 72.0
        target_w_px, target_h_px = int(round(target_w_in * dpi_out)), int(round(target_h_in * dpi_out))

        c = rl_canvas.Canvas(str(out_pdf), pagesize=(target_w_pt, target_h_pt), pageCompression=1)

        for i, rel in enumerate(svg_rel, start=1):
            if cancel_evt.is_set(): qmsg.put(("warn", tr("aborted"))); return 0, None
            qmsg.put(("page", (i, total_pages_file)))
            qmsg.put(("log", f"🧩 {tr('page')} {i}/{total_pages_file}: {rel}"))

            svg_abs = tmp / rel
            pad_final, crop_detect = detect_bbox_fast(
                svg_orig=svg_abs, tmp_dir=patched_dir, root_dir=tmp, fonts_dir=fonts_dir,
                pad_px_init=start_pad_px, dpi_detect=dpi_detect, dpi_out=dpi_out,
                safety_px_manual=safety_px, dilate_px=dilate_px, alpha_thr=alpha_thr,
                auto_safe=auto_safe
            )
            page_rgb = render_final_cropped(
                svg_orig=svg_abs, tmp_dir=patched_dir, root_dir=tmp, fonts_dir=fonts_dir,
                pad_px_final=pad_final, crop_detect=crop_detect,
                dpi_detect=dpi_detect, dpi_out=dpi_out
            )

            # proportional auf Zielseite einpassen (kleines Fit<1)
            img_w, img_h = page_rgb.size
            fit = min(target_w_px / img_w, target_h_px / img_h, 0.999)
            new_w, new_h = max(1, int(round(img_w * fit))), max(1, int(round(img_h * fit)))
            if (new_w, new_h) != (img_w, img_h):
                page_rgb = page_rgb.resize((new_w, new_h), Image.LANCZOS)

            # Seite weiß hinterlegen
            c.setPageSize((target_w_pt, target_h_pt))
            c.setFillColor(white); c.rect(0, 0, target_w_pt, target_h_pt, fill=1, stroke=0)

            # In-Memory einbetten (kein Temp-PNG)
            buf = BytesIO(); page_rgb.save(buf, format="PNG"); buf.seek(0)
            img_rd = ImageReader(buf)

            inset = max(0.0, float(inset_pt))
            draw_w_pt = (page_rgb.size[0] / dpi_out) * 72.0
            draw_h_pt = (page_rgb.size[1] / dpi_out) * 72.0
            off_x_pt = (target_w_pt - draw_w_pt) / 2.0 + inset
            off_y_pt = (target_h_pt - draw_h_pt) / 2.0 + inset

            c.drawImage(img_rd, off_x_pt, off_y_pt,
                        width=max(1e-3, draw_w_pt - 2*inset),
                        height=max(1e-3, draw_h_pt - 2*inset),
                        preserveAspectRatio=False, anchor='sw', mask='auto')
            c.showPage()

            # Seiten-Balken
            qmsg.put(("page_progress", i/total_pages_file))
            # Gesamt-Balken (alle Dateien/Seiten)
            done = pages_done_base + i
            qmsg.put(("overall_progress", (done, pages_total_all)))

        c.save()
        return total_pages_file, out_pdf


# =========================================================
#                  Worker (Batch)
# =========================================================
def pre_scan_total_pages(paths: list[str]) -> tuple[int, list[int]]:
    total = 0; per_file = []
    for p in paths:
        cnt = 0
        try:
            with zipfile.ZipFile(p, "r") as zf:
                cnt = sum(1 for n in zf.namelist() if n.lower().endswith(".svg"))
        except Exception:
            cnt = 0
        per_file.append(cnt); total += cnt
    return total, per_file

def convert_worker(args, qmsg: queue.Queue, cancel_evt: threading.Event):
    try:
        (paths, dpi_out, fonts_dir, page_mode,
         start_pad_px, safety_px, dilate_px, alpha_thr, inset_pt,
         dpi_detect, auto_safe) = args

        # Warm-Up (zweites Mal – hier mit finalen Parametern)
        _warmup_pillow()
        _warmup_resvg(fonts_dir)

        total_pages_all, pages_per_file = pre_scan_total_pages(paths)
        qmsg.put(("overall_init", total_pages_all))

        t0 = time.time()
        successes = []
        pages_done = 0
        file_count = len(paths)

        for idx, p in enumerate(paths, start=1):
            if cancel_evt.is_set(): qmsg.put(("warn", tr("aborted"))); break
            nb_path = Path(p)
            if nb_path.suffix.lower() != ".notebook":
                qmsg.put(("warn", f"Ignoriere (kein .notebook): {nb_path}")); continue

            added_pages, out_pdf = convert_notebook(
                nb_path, dpi_out, fonts_dir, page_mode,
                start_pad_px, safety_px, dilate_px, alpha_thr,
                inset_pt, dpi_detect, auto_safe,
                qmsg, cancel_evt,
                file_index=idx, file_count=file_count,
                pages_done_base=pages_done, pages_total_all=total_pages_all
            )
            pages_done += added_pages
            if out_pdf: successes.append(out_pdf)

        dt = time.time() - t0
        if successes:
            qmsg.put(("done", f"{tr('info_done')}: {dt:.1f}s\n" + "\n".join(str(x) for x in successes)))
        else:
            qmsg.put(("warn", "Keine PDFs erzeugt."))
    except Exception as e:
        qmsg.put(("error", f"{tr('err')}: {e}"))


# =========================================================
#                     GUI / App
# =========================================================
def pump_queue():
    try:
        while True:
            typ, payload = qmsg.get_nowait()
            if typ == "log":
                if log_visible.get():
                    txt_log.configure(state="normal"); txt_log.insert("end", payload + "\n"); txt_log.see("end"); txt_log.configure(state="disabled")
                status_var.set(_truncate(payload, 90))

            elif typ == "overall_init":
                total = int(payload)
                overall_total_var.set(total)
                label_overall_var.set(f"{tr('overall')} 0/{total}")
                pct_overall_var.set("0%"); bar_overall.set(0.0)

            elif typ == "overall_progress":
                done, total = payload
                total = max(1, int(total))
                frac = max(0.0, min(1.0, done/total))
                bar_overall.set(frac)
                pct_overall_var.set(f"{int(frac*100):d}%")
                label_overall_var.set(f"{tr('overall')} {done}/{total}")

            elif typ == "file_begin":
                fi, ftot, name, pcount = payload
                file_info_var.set(f"{tr('file')} {fi}/{ftot}: {name}")
                label_page_var.set(f"{tr('page')} 1/{pcount}")
                pct_page_var.set("0%"); bar_page.set(0.0)

            elif typ == "page":
                pi, ptot = payload
                label_page_var.set(f"{tr('page')} {pi}/{ptot}")

            elif typ == "page_progress":
                v = float(payload); bar_page.set(v); pct_page_var.set(f"{int(v*100):d}%")

            elif typ == "warn":
                enable_controls(True); _reset_progress()
                messagebox.showwarning(tr("warn"), payload); status_var.set(tr("aborted"))

            elif typ == "error":
                enable_controls(True); _reset_progress()
                messagebox.showerror(tr("err"), payload); status_var.set(tr("err"))

            elif typ == "done":
                enable_controls(True)
                bar_page.set(1.0); pct_page_var.set("100%")
                bar_overall.set(1.0); pct_overall_var.set("100%")
                messagebox.showinfo(tr("info_done"), payload); status_var.set(tr("info_done"))

            qmsg.task_done()
    except queue.Empty:
        pass
    app.after(50, pump_queue)

def _truncate(s: str, n: int): return (s[:n-1] + "…") if len(s) > n else s
def _reset_progress():
    bar_overall.set(0); pct_overall_var.set("0%"); label_overall_var.set(f"{tr('overall')} —/—")
    bar_page.set(0); pct_page_var.set("0%"); label_page_var.set(f"{tr('page')} —/—")
    file_info_var.set(f"{tr('file')} —/—")
    status_var.set(tr("ready"))

def enable_controls(state: bool):
    for w in controls:
        try: w.configure(state=("normal" if state else "disabled"))
        except Exception: pass

def pick_files():
    paths = filedialog.askopenfilenames(
        title=tr("pick_files_title"),
        filetypes=[("SMART Notebook", "*.notebook"), ("Alle Dateien", "*.*")]
    )
    if paths:
        entry_files.delete(0, ctk.END); entry_files.insert(0, "; ".join(paths))

def pick_folder():
    folder = filedialog.askdirectory(title=tr("pick_folder_title"))
    if folder:
        files = sorted(glob.glob(os.path.join(folder, "*.notebook")), key=natural_key)
        entry_files.delete(0, ctk.END)
        entry_files.insert(0, "; ".join(files) if files else "")
        if not files:
            messagebox.showinfo(tr("warn"), tr("no_files_in_folder"))

def start():
    if worker_cancel.is_set(): worker_cancel.clear()
    raw = entry_files.get().strip()
    if not raw:
        messagebox.showwarning(tr("warn"), tr("warn_select_first")); return
    paths = [p.strip() for p in raw.split(";") if p.strip()]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        messagebox.showwarning(tr("warn"), tr("warn_no_valid")); return

    dpi_val   = int(entry_dpi.get().strip() or "300")
    fonts_dir = entry_fonts.get().strip() or None
    page_mode = {"Notebook (Seite 1)":"NOTEBOOK","A4 Hoch":"A4_P","A4 Quer":"A4_L",
                 "Notebook (page 1)":"NOTEBOOK","A4 portrait":"A4_P","A4 landscape":"A4_L"}[cmb_size.get()]
    start_pad = int(entry_pad.get().strip() or "32")
    safety    = int(entry_safety.get().strip() or "24")
    dilate    = int(entry_dilate.get().strip() or "2")
    alpha_thr = int(entry_alpha.get().strip() or "2")
    inset     = float(entry_inset.get().strip() or "0.5")
    dpi_detect = int(entry_dpidetect.get().strip() or "72")
    auto_safe = bool(var_autosafe.get())

    enable_controls(False); _reset_progress()
    if log_visible.get():
        txt_log.configure(state="normal"); txt_log.delete("1.0", "end"); txt_log.configure(state="disabled")
    status_var.set(tr("starting")); app.title(tr("app_title"))

    args = (paths, dpi_val, fonts_dir, page_mode, start_pad, safety, dilate, alpha_thr, inset, dpi_detect, auto_safe)
    threading.Thread(target=convert_worker, args=(args, qmsg, worker_cancel), daemon=True).start()

def cancel():
    worker_cancel.set()
    if log_visible.get():
        txt_log.configure(state="normal"); txt_log.insert("end", "⏹️ " + tr("aborting") + " …\n"); txt_log.configure(state="disabled")
    status_var.set(tr("aborting"))

def system_check():
    if log_visible.get():
        txt_log.configure(state="normal"); txt_log.delete("1.0", "end"); txt_log.configure(state="disabled")
    from importlib.metadata import version
    def ver(n):
        try: return version(n)
        except Exception: return "nicht gefunden"
    lines = [
        f"Python: {sys.version}",
        f"Interpreter: {sys.executable}",
        f"customtkinter: {ver('customtkinter')}",
        f"reportlab: {ver('reportlab')}",
        f"resvg_py: {ver('resvg_py')}",
        f"Pillow: {ver('Pillow')}",
    ]
    if log_visible.get():
        txt_log.configure(state="normal")
        for t in lines: txt_log.insert("end", t + "\n")
        ok, err = ensure_resvg()
        txt_log.insert("end", "resvg_py: ✅ importierbar\n" if ok else f"resvg_py: ❌ {err}\n")
        txt_log.configure(state="disabled")
    else:
        messagebox.showinfo("System-Check", "\n".join(lines))

def toggle_log():
    if log_visible.get():
        bottom_bar.pack(fill="x", padx=16, pady=(0, 8))
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
    else:
        log_frame.pack_forget()
        bottom_bar.pack(fill="x", padx=16, pady=(0, 8))

def set_language(new_lang: str):
    global LANG
    LANG = new_lang
    # statische Texte aktualisieren
    app.title(tr("app_title"))
    lbl_header.configure(text=tr("header"))
    lbl_input.configure(text=tr("input"))
    entry_files.configure(placeholder_text=tr("entry_hint"))
    btn_pick_files.configure(text=tr("choose_files"))
    btn_pick_folder.configure(text=tr("choose_folder"))

    lbl_settings.configure(text=tr("settings"))
    lbl_page_size.configure(text=tr("page_size"))
    lbl_dpi.configure(text=tr("dpi"))
    lbl_fonts.configure(text=tr("fonts_dir"))

    lbl_autocrop_group.configure(text=tr("autocrop_group"))
    lbl_start_pad.configure(text=tr("start_pad"))
    lbl_safety.configure(text=tr("safety"))
    chk_autosafe.configure(text=tr("auto_safe"))
    lbl_dilate.configure(text=tr("dilation"))
    lbl_alpha.configure(text=tr("alpha_thr"))
    lbl_dpidetect.configure(text=tr("detect_dpi"))
    lbl_inset.configure(text=tr("inset"))

    btn_start.configure(text=tr("convert"))
    btn_cancel.configure(text=tr("cancel"))
    toggle.configure(text=tr("log_toggle"))
    btn_check.configure(text=tr("system_check"))
    lbl_lang.configure(text=tr("language"))

    # Progress-Beschriftungen neu setzen
    label_overall_var.set(f"{tr('overall')} —/—")
    file_info_var.set(f"{tr('file')} —/—")
    label_page_var.set(f"{tr('page')} —/—")
    status_var.set(tr("ready"))

    # Seitenformat-Werte passend zur Sprache
    if LANG == "DE":
        cmb_size.configure(values=[tr("nb_size"), tr("a4p"), tr("a4l")])
        if cmb_size.get() in ["Notebook (page 1)", "A4 portrait", "A4 landscape"]:
            cmb_size.set(tr("nb_size"))
    else:
        cmb_size.configure(values=[tr("nb_size"), tr("a4p"), tr("a4l")])
        if cmb_size.get() in ["Notebook (Seite 1)", "A4 Hoch", "A4 Quer"]:
            cmb_size.set(tr("nb_size"))

# =============== Build UI ===============
ctk.set_appearance_mode("system"); ctk.set_default_color_theme("blue")

app = ctk.CTk(); app.title(tr("app_title")); app.geometry("1080x780")

# Header
header = ctk.CTkFrame(app, corner_radius=0); header.pack(fill="x")
lbl_header = ctk.CTkLabel(header, text=tr("header"), font=ctk.CTkFont(size=20, weight="bold"))
lbl_header.pack(side="left", padx=16, pady=12)

theme_switch = ctk.CTkSwitch(header, text=tr("darkmode"),
                             command=lambda: ctk.set_appearance_mode("dark" if theme_switch.get() else "light"))
theme_switch.select(); theme_switch.pack(side="right", padx=16)

# Sprache
lang_box = ctk.CTkFrame(app); lang_box.pack(fill="x", padx=16, pady=(6, 0))
lbl_lang = ctk.CTkLabel(lang_box, text=tr("language"))
lbl_lang.pack(side="left", padx=(4, 8), pady=4)
cmb_lang = ctk.CTkComboBox(lang_box, values=["Deutsch", "English"], width=140,
                            command=lambda s: set_language("DE" if s=="Deutsch" else "EN"))
cmb_lang.set("Deutsch"); cmb_lang.pack(side="left")

# Fortschritt – OBERER Balken = Gesamtfortschritt
prog = ctk.CTkFrame(app); prog.pack(fill="x", padx=16, pady=(12, 10))
overall_row = ctk.CTkFrame(prog); overall_row.pack(fill="x", padx=8, pady=(6, 2))
overall_row.grid_columnconfigure(0, weight=1); overall_row.grid_columnconfigure(1, weight=0)
label_overall_var = ctk.StringVar(value=f"{tr('overall')} —/—")
pct_overall_var = ctk.StringVar(value="0%"); overall_total_var = ctk.IntVar(value=0)
ctk.CTkLabel(overall_row, textvariable=label_overall_var, anchor="w").grid(row=0, column=0, sticky="w")
ctk.CTkLabel(overall_row, textvariable=pct_overall_var, anchor="e", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, sticky="e")
bar_overall = ctk.CTkProgressBar(prog, height=14); bar_overall.set(0.0)
bar_overall.pack(fill="x", padx=8, pady=(0, 8))

# Unterer Bereich: links aktuelle Datei, rechts aktuelle Seite
page_hdr = ctk.CTkFrame(prog); page_hdr.pack(fill="x", padx=8, pady=(4, 2))
page_hdr.grid_columnconfigure(0, weight=1); page_hdr.grid_columnconfigure(1, weight=0)
file_info_var = ctk.StringVar(value=f"{tr('file')} —/—")
label_page_var = ctk.StringVar(value=f"{tr('page')} —/—"); pct_page_var = ctk.StringVar(value="0%")
ctk.CTkLabel(page_hdr, textvariable=file_info_var, anchor="w").grid(row=0, column=0, sticky="w")
ctk.CTkLabel(page_hdr, textvariable=label_page_var, anchor="e").grid(row=0, column=1, sticky="e")
bar_page = ctk.CTkProgressBar(prog, height=14); bar_page.set(0.0)
bar_page.pack(fill="x", padx=8, pady=(0, 4))
pct_page_lbl = ctk.CTkLabel(prog, textvariable=pct_page_var, anchor="e")
pct_page_lbl.pack(fill="x", padx=8, pady=(0, 6))

# Statuszeile
status_var = ctk.StringVar(value=tr("ready"))
ctk.CTkLabel(app, textvariable=status_var, anchor="w").pack(fill="x", padx=24, pady=(0, 4))

# Hauptbereich (Eingabe & Einstellungen)
main = ctk.CTkFrame(app, corner_radius=12); main.pack(padx=16, pady=12, fill="both", expand=True)
main.grid_columnconfigure(0, weight=1); main.grid_columnconfigure(1, weight=1)

# Eingabe (links)
grp_in = ctk.CTkFrame(main, corner_radius=12); grp_in.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
lbl_input = ctk.CTkLabel(grp_in, text=tr("input"), font=ctk.CTkFont(size=16, weight="bold"))
lbl_input.pack(anchor="w", padx=12, pady=(12, 6))
entry_files = ctk.CTkEntry(grp_in, placeholder_text=tr("entry_hint"))
entry_files.pack(fill="x", padx=12, pady=6)
row_pick = ctk.CTkFrame(grp_in); row_pick.pack(fill="x", padx=12, pady=(0, 8))
btn_pick_files = ctk.CTkButton(row_pick, text=tr("choose_files"), command=lambda: pick_files(), width=160)
btn_pick_folder = ctk.CTkButton(row_pick, text=tr("choose_folder"), command=lambda: pick_folder(), width=160)
btn_pick_files.pack(side="left", padx=(0, 8))
btn_pick_folder.pack(side="left", padx=(0, 8))

# Einstellungen (rechts)
grp_opt = ctk.CTkFrame(main, corner_radius=12); grp_opt.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
lbl_settings = ctk.CTkLabel(grp_opt, text=tr("settings"), font=ctk.CTkFont(size=16, weight="bold"))
lbl_settings.grid(row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(12, 6))
for c in range(4): grp_opt.grid_columnconfigure(c, weight=1)

lbl_page_size = ctk.CTkLabel(grp_opt, text=tr("page_size"))
lbl_page_size.grid(row=1, column=0, sticky="e", padx=(12,6), pady=6)
cmb_size = ctk.CTkComboBox(grp_opt, values=[tr("nb_size"), tr("a4p"), tr("a4l")], width=200)
cmb_size.set(tr("nb_size")); cmb_size.grid(row=1, column=1, sticky="w", padx=6, pady=6)

lbl_dpi = ctk.CTkLabel(grp_opt, text=tr("dpi"))
lbl_dpi.grid(row=1, column=2, sticky="e", padx=(12,6), pady=6)
entry_dpi = ctk.CTkEntry(grp_opt, width=80); entry_dpi.insert(0, "300")
entry_dpi.grid(row=1, column=3, sticky="w", padx=6, pady=6)

lbl_fonts = ctk.CTkLabel(grp_opt, text=tr("fonts_dir"))
lbl_fonts.grid(row=2, column=0, sticky="e", padx=(12,6), pady=6)
entry_fonts = ctk.CTkEntry(grp_opt); entry_fonts.grid(row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=6)

# Advanced
adv = ctk.CTkFrame(grp_opt); adv.grid(row=3, column=0, columnspan=4, sticky="ew", padx=12, pady=(4, 6))
lbl_autocrop_group = ctk.CTkLabel(adv, text=tr("autocrop_group"), font=ctk.CTkFont(weight="bold"))
lbl_autocrop_group.grid(row=0, column=0, columnspan=4, sticky="w", pady=(4, 2))
for c in range(4): adv.grid_columnconfigure(c, weight=1)

lbl_start_pad = ctk.CTkLabel(adv, text=tr("start_pad"))
lbl_start_pad.grid(row=1, column=0, sticky="e", padx=(0,6), pady=4)
entry_pad = ctk.CTkEntry(adv, width=80); entry_pad.insert(0, "32")
entry_pad.grid(row=1, column=1, sticky="w", padx=6, pady=4)

lbl_safety = ctk.CTkLabel(adv, text=tr("safety"))
lbl_safety.grid(row=1, column=2, sticky="e", padx=(12,6), pady=4)
entry_safety = ctk.CTkEntry(adv, width=80); entry_safety.insert(0, "24")
entry_safety.grid(row=1, column=3, sticky="w", padx=6, pady=4)

var_autosafe = ctk.IntVar(value=1)
chk_autosafe = ctk.CTkCheckBox(adv, text=tr("auto_safe"), variable=var_autosafe)
chk_autosafe.grid(row=2, column=0, columnspan=4, sticky="w", pady=(0,4))

lbl_dilate = ctk.CTkLabel(adv, text=tr("dilation"))
lbl_dilate.grid(row=3, column=0, sticky="e", padx=(0,6), pady=4)
entry_dilate = ctk.CTkEntry(adv, width=80); entry_dilate.insert(0, "2")
entry_dilate.grid(row=3, column=1, sticky="w", padx=6, pady=4)

lbl_alpha = ctk.CTkLabel(adv, text=tr("alpha_thr"))
lbl_alpha.grid(row=3, column=2, sticky="e", padx=(12,6), pady=4)
entry_alpha = ctk.CTkEntry(adv, width=80); entry_alpha.insert(0, "2")
entry_alpha.grid(row=3, column=3, sticky="w", padx=6, pady=4)

lbl_dpidetect = ctk.CTkLabel(adv, text=tr("detect_dpi"))
lbl_dpidetect.grid(row=4, column=0, sticky="e", padx=(0,6), pady=4)
entry_dpidetect = ctk.CTkEntry(adv, width=80); entry_dpidetect.insert(0, "72")
entry_dpidetect.grid(row=4, column=1, sticky="w", padx=6, pady=4)

lbl_inset = ctk.CTkLabel(adv, text=tr("inset"))
lbl_inset.grid(row=4, column=2, sticky="e", padx=(12,6), pady=4)
entry_inset = ctk.CTkEntry(adv, width=80); entry_inset.insert(0, "0.5")
entry_inset.grid(row=4, column=3, sticky="w", padx=6, pady=4)

# Buttons (Start/Cancel)
row_btns = ctk.CTkFrame(grp_opt); row_btns.grid(row=4, column=0, columnspan=4, sticky="ew", padx=12, pady=(8, 12))
row_btns.grid_columnconfigure(0, weight=1); row_btns.grid_columnconfigure(3, weight=1)
btn_start  = ctk.CTkButton(row_btns, text=tr("convert"), command=start, width=160)
btn_cancel = ctk.CTkButton(row_btns, text=tr("cancel"), command=cancel, fg_color="#8a1d1d", hover_color="#6e1515")
btn_start.grid(row=0, column=0, sticky="w"); btn_cancel.grid(row=0, column=1, sticky="w", padx=8)

# Untere Zeile: Toggle links + System-Check rechts
bottom_bar = ctk.CTkFrame(app); bottom_bar.pack(fill="x", padx=16, pady=(0, 8))
bottom_bar.grid_columnconfigure(0, weight=1)
log_visible = ctk.IntVar(value=0)
toggle = ctk.CTkCheckBox(bottom_bar, text=tr("log_toggle"), variable=log_visible, command=toggle_log)
toggle.grid(row=0, column=0, sticky="w", padx=4, pady=6)
btn_check  = ctk.CTkButton(bottom_bar, text=tr("system_check"), command=system_check, width=140)
btn_check.grid(row=0, column=1, sticky="e", padx=4, pady=6)

# Optionales Log
log_frame = ctk.CTkFrame(app)
txt_log = ctk.CTkTextbox(log_frame, height=220, state="disabled"); txt_log.pack(padx=12, pady=12, fill="both", expand=True)

# Controls, Queue, Start pump
controls = [entry_files, cmb_size, entry_dpi, entry_fonts,
            entry_pad, entry_safety, entry_dilate, entry_alpha, entry_dpidetect, entry_inset,
            btn_start, btn_cancel, btn_check]
qmsg = queue.Queue()
worker_cancel = threading.Event()
app.after(50, pump_queue)

# Frühes Warm-up (reduziert Latenz der ersten Seite)
warmup_async(lambda: entry_fonts.get().strip() or None)

app.mainloop()
