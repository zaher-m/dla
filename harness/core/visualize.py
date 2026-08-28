"""Standardized, system-agnostic layout visualisation.

Every system is drawn with exactly the same canvas size, scaling, colour map,
stroke weight, fill opacity and label typography, so no implementation can be
made to look better by its own renderer.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont
from core.taxonomy import COLORS

PANEL_W = 1100           # canonical panel width in px for every rendering
STROKE = 4
FILL_ALPHA = 46
LABEL_H = 26

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_page(image_path, regions, out_path, title=None, show_conf=True,
                show_order=False, panel_w=PANEL_W, draw_polygons=True):
    base = Image.open(image_path).convert("RGB")
    scale = panel_w / base.width
    W, H = panel_w, int(round(base.height * scale))
    base = base.resize((W, H), Image.LANCZOS)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # translucent fills first so strokes stay crisp on top
    for r in regions:
        c = COLORS.get(r["class"], COLORS["other"])
        poly = r.get("polygon")
        if draw_polygons and poly and len(poly) >= 3:
            od.polygon([(x * scale, y * scale) for x, y in poly], fill=c + (FILL_ALPHA,))
        else:
            x1, y1, x2, y2 = [v * scale for v in r["bbox"]]
            od.rectangle([x1, y1, x2, y2], fill=c + (FILL_ALPHA,))
    img = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    d = ImageDraw.Draw(img)
    f_lab = _font(15)
    for r in regions:
        c = COLORS.get(r["class"], COLORS["other"])
        poly = r.get("polygon")
        x1, y1, x2, y2 = [v * scale for v in r["bbox"]]
        if draw_polygons and poly and len(poly) >= 3:
            pts = [(x * scale, y * scale) for x, y in poly]
            d.line(pts + [pts[0]], fill=c, width=STROKE)
        else:
            d.rectangle([x1, y1, x2, y2], outline=c, width=STROKE)
        lab = r["class"]
        if show_order and r.get("reading_order") is not None:
            lab = f"{r['reading_order']}·{lab}"
        if show_conf and r.get("confidence") is not None:
            lab += f" {r['confidence']:.2f}"
        tw = d.textlength(lab, font=f_lab)
        ly = max(0, y1 - LABEL_H + 4)
        d.rectangle([x1, ly, x1 + tw + 10, ly + LABEL_H - 5], fill=c)
        d.text((x1 + 5, ly + 2), lab, fill=(255, 255, 255), font=f_lab)

    if title:
        img = _add_titlebar(img, title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path


def _add_titlebar(img, title, bar_h=46):
    W, H = img.size
    out = Image.new("RGB", (W, H + bar_h), (17, 24, 39))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((12, 12), title, fill=(255, 255, 255), font=_font(22))
    return out


def render_original(image_path, out_path, title="Original", panel_w=PANEL_W):
    base = Image.open(image_path).convert("RGB")
    scale = panel_w / base.width
    base = base.resize((panel_w, int(round(base.height * scale))), Image.LANCZOS)
    base = _add_titlebar(base, title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base.save(out_path)
    return out_path


def legend_image(classes, out_path, width=1100, cols=5):
    f = _font(17)
    rows = (len(classes) + cols - 1) // cols
    cw, rh = width // cols, 34
    img = Image.new("RGB", (width, rows * rh + 12), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, c in enumerate(classes):
        x = (i % cols) * cw + 10
        y = (i // cols) * rh + 8
        d.rectangle([x, y + 4, x + 22, y + 22], fill=COLORS.get(c, COLORS["other"]),
                    outline=(0, 0, 0))
        d.text((x + 30, y + 4), c, fill=(0, 0, 0), font=f)
    img.save(out_path)
    return out_path


def grid(panel_paths, out_path, cols=4, pad=10, bg=(243, 244, 246)):
    """Compose equally-sized panels into a comparison sheet."""
    ims = [Image.open(p).convert("RGB") for p in panel_paths]
    cw = max(i.width for i in ims)
    ch = max(i.height for i in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw + pad * (cols + 1), rows * ch + pad * (rows + 1)), bg)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (cw + pad), pad + r * (ch + pad)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    return out_path
