"""Canonical layout taxonomy and per-system class mapping tables.

The canonical set is the one mandated by the benchmark specification.  Two
*documented extensions* are added because the corpus genuinely contains those
regions and folding them into a mandated class would be a silent merge of
unrelated semantics (spec section 7 forbids that):

  page_number  -- distinct from `footer`; several systems predict it separately
  separator    -- rules/lines; PRImA and Eynollah emit these as real regions

Mapping confidence:
  exact       -- the source class means the same thing as the canonical class
  approximate -- overlapping but not identical semantics (documented in notes)
  ambiguous   -- retained as `other`; source_class is always preserved
"""

CANONICAL = [
    "title", "heading", "text", "list", "table", "figure", "caption",
    "formula", "header", "footer", "footnote", "sidebar", "other",
]
EXTENSIONS = ["page_number", "separator"]
ALL_CLASSES = CANONICAL + EXTENSIONS

# Spec-mandated colours (RGB). Extensions get distinct, non-clashing hues.
COLORS = {
    "title":       (220,  38,  38),   # red
    "heading":     (249, 115,  22),   # orange
    "text":        ( 37,  99, 235),   # blue
    "table":       ( 22, 163,  74),   # green
    "figure":      (147,  51, 234),   # purple
    "caption":     (236,  72, 153),   # pink
    "formula":     (234, 179,   8),   # yellow
    "header":      (  6, 182, 212),   # cyan
    "footer":      (107, 114, 128),   # gray
    "list":        (146,  64,  14),   # brown
    "footnote":    (132, 133,  34),   # olive
    "sidebar":     (217,  70, 239),   # magenta
    "other":       ( 71,  85, 105),   # slate
    "page_number": (148, 163, 184),   # light slate
    "separator":   ( 13, 148, 136),   # teal
}

# --------------------------------------------------------------------------
# Per-system mapping tables.  Key = system id used in the registry.
# Value = {source_class: (canonical, confidence, note)}
# --------------------------------------------------------------------------

MAPPINGS = {
    # ---- PubLayNet 5-class taxonomy (Layout-Parser D2, LayoutLMv3-PubLayNet,
    #      Detectron2 baseline, EfficientDet) ------------------------------
    "publaynet": {
        "Text":   ("text",    "exact",       ""),
        "Title":  ("title",   "approximate", "PubLayNet 'Title' covers both document titles and section headings; "
                                             "cannot be split without a second signal."),
        "List":   ("list",    "exact",       ""),
        "Table":  ("table",   "exact",       ""),
        "Figure": ("figure",  "exact",       ""),
    },
    # ---- PRImA Layout (Layout-Parser PrimaLayout checkpoint) -------------
    "prima": {
        "TextRegion":      ("text",      "exact",       ""),
        "ImageRegion":     ("figure",    "exact",       ""),
        "TableRegion":     ("table",     "exact",       ""),
        "MathsRegion":     ("formula",   "exact",       ""),
        "SeparatorRegion": ("separator", "exact",       ""),
        "OtherRegion":     ("other",     "ambiguous",   "PRImA catch-all class."),
    },
    # ---- HJDataset (historical Japanese) --------------------------------
    "hjdataset": {
        "Page Frame": ("other",   "ambiguous", "Page-frame boundary, no canonical equivalent."),
        "Row":        ("other",   "ambiguous", "Structural row grouping."),
        "Title Region": ("title", "approximate", ""),
        "Text Region":  ("text",  "exact", ""),
        "Title":        ("title", "approximate", ""),
        "Subtitle":     ("heading", "approximate", ""),
        "Other":        ("other", "ambiguous", ""),
    },
    # ---- NewspaperNavigator ---------------------------------------------
    "newspaper": {
        "Photograph":  ("figure",  "exact", ""),
        "Illustration":("figure",  "exact", ""),
        "Map":         ("figure",  "approximate", "Map is a figure sub-type."),
        "Comics/Cartoon": ("figure", "exact", ""),
        "Editorial Cartoon": ("figure", "exact", ""),
        "Headline":    ("heading", "exact", ""),
        "Advertisement": ("other", "ambiguous", "No canonical equivalent."),
    },
    # ---- DocLayNet 11-class (YOLO-DocLayNet, doc-layout-net, Docling) ----
    "doclaynet": {
        "Caption":          ("caption",  "exact", ""),
        "Footnote":         ("footnote", "exact", ""),
        "Formula":          ("formula",  "exact", ""),
        "List-item":        ("list",     "exact", "DocLayNet labels each list item; canonical 'list' is per-item here."),
        "Page-footer":      ("footer",   "exact", ""),
        "Page-header":      ("header",   "exact", ""),
        "Picture":          ("figure",   "exact", ""),
        "Section-header":   ("heading",  "exact", ""),
        "Table":            ("table",    "exact", ""),
        "Text":             ("text",     "exact", ""),
        "Title":            ("title",    "exact", ""),
    },
    # ---- Docling layout model (heron / RT-DETR, DocLayNet-v2 label set) --
    # ---- D4LA (Diverse DocLayout Analysis, 27 classes) --------------------
    # DocLayout-YOLO's D4LA checkpoint and VGT's D4LA checkpoint.  D4LA is a
    # business/letter-heavy distribution, so several of its classes have no
    # canonical counterpart and are mapped as "approximate" or "ambiguous"
    # rather than silently dropped.
    #
    # Three of these names are actively misleading and are mapped from the
    # dataset paper's own definitions (arXiv:2308.14978 Appendix B), not from
    # what the English word suggests:
    #   Footer      "is the footnote of the document"      -> footnote, NOT footer
    #                (D4LA has a separate PageFooter for the real page footer)
    #   LetterHead  "the inside address ... name and address of the recipient"
    #                -> body text, NOT a running header or a letterhead
    #   Number      "the special number in IIT-CDIP that is not the content of
    #                the document and often vertical text" -> other, NOT a page
    #                number (D4LA has a separate PageNumber)
    # `Regionlist` / `Footnote` are the spellings used in VGT's demo script
    # (object_detection/inference.py); the dataset's own class list and
    # DocLayout-YOLO's d4la.yaml both say `RegionList` / `Footer`.  Both
    # spellings are accepted so no prediction can go unmapped.
    "d4la": {
        "DocTitle":    ("title",   "exact", ""),
        "ParaTitle":   ("heading", "exact", ""),
        "RegionTitle": ("heading", "approximate", "Title of a sub-region rather than a section."),
        "ParaText":    ("text",    "exact", ""),
        "ListText":    ("list",    "exact", ""),
        "OtherText":   ("text",    "approximate", ""),
        "Abstract":    ("text",    "approximate", ""),
        "Author":      ("text",    "approximate", ""),
        "Date":        ("text",    "approximate", "Letter date line."),
        "LetterHead":  ("text",    "approximate", "Inside address of a letter (recipient name/address), not a page header."),
        "LetterDear":  ("text",    "approximate", "Salutation line."),
        "LetterSign":  ("text",    "approximate", "Signature block."),
        "Question":    ("text",    "approximate", ""),
        "RegionKV":    ("other",   "ambiguous", "Key-value region (form field)."),
        "RegionList":  ("list",    "approximate", "Line-less list / wireless form region, whole block rather than per-item."),
        "Regionlist":  ("list",    "approximate", "Line-less list / wireless form region, whole block rather than per-item."),
        "TableName":   ("caption", "exact", ""),
        "Table":       ("table",   "exact", ""),
        "Figure":      ("figure",  "exact", ""),
        "FigureName":  ("caption", "exact", ""),
        "Equation":    ("formula", "exact", ""),
        "Reference":   ("text",    "approximate", "Bibliography entry."),
        "Footer":      ("footnote", "exact", "D4LA's `Footer` is the document footnote (paper, App. B), not a page footer."),
        "Footnote":    ("footnote", "exact", ""),
        "PageHeader":  ("header",  "exact", ""),
        "PageFooter":  ("footer",  "exact", ""),
        "PageNumber":  ("page_number", "exact", ""),
        "Number":      ("other",   "ambiguous", "IIT-CDIP stamp number, often vertical; not document content and not a page number."),
        "Catalog":     ("other",   "approximate", "Table-of-contents block."),
    },
    "docling": {
        "caption": ("caption", "exact", ""), "footnote": ("footnote", "exact", ""),
        "formula": ("formula", "exact", ""), "list_item": ("list", "exact", ""),
        "page_footer": ("footer", "exact", ""), "page_header": ("header", "exact", ""),
        "picture": ("figure", "exact", ""), "section_header": ("heading", "exact", ""),
        "table": ("table", "exact", ""), "text": ("text", "exact", ""),
        "title": ("title", "exact", ""), "document_index": ("other", "approximate", "Table-of-contents block."),
        "code": ("other", "approximate", "Code block; no canonical class."),
        "checkbox_selected": ("other", "ambiguous", ""), "checkbox_unselected": ("other", "ambiguous", ""),
        "form": ("other", "ambiguous", ""), "key_value_region": ("other", "ambiguous", ""),
        "picture_group": ("figure", "approximate", "Group container around pictures."),
        "list_group": ("list", "approximate", "Group container around list items."),
        "grading_scale": ("other", "ambiguous", ""), "handwritten_text": ("text", "approximate", ""),
        "empty_value": ("other", "ambiguous", ""), "reference": ("text", "approximate", "Bibliographic reference line."),
        "chart": ("figure", "approximate", "Chart is a figure sub-type."),
    },
    # ---- MinerU PP-DocLayoutV2 / MinerU2.5 VLM block types ---------------
    "mineru": {
        "text": ("text", "exact", ""), "title": ("title", "approximate", "MinerU 'title' is any heading level."),
        "doc_title": ("title", "exact", ""), "paragraph_title": ("heading", "exact", ""),
        "abstract": ("text", "approximate", "Abstract block."),
        "image": ("figure", "exact", ""), "image_body": ("figure", "exact", ""),
        "chart": ("figure", "approximate", "Chart is a figure sub-type."),
        "chart_body": ("figure", "approximate", ""),
        "table": ("table", "exact", ""), "table_body": ("table", "exact", ""),
        "caption": ("caption", "exact", ""), "image_caption": ("caption", "exact", ""),
        "table_caption": ("caption", "exact", ""), "chart_caption": ("caption", "exact", ""),
        "code_caption": ("caption", "exact", ""), "algorithm_caption": ("caption", "exact", ""),
        "footnote": ("footnote", "exact", ""), "image_footnote": ("footnote", "exact", ""),
        "table_footnote": ("footnote", "exact", ""), "chart_footnote": ("footnote", "exact", ""),
        "page_footnote": ("footnote", "exact", ""),
        "interline_equation": ("formula", "exact", ""), "equation": ("formula", "exact", ""),
        "formula_number": ("formula", "approximate", "Equation numbering label."),
        "list": ("list", "exact", ""), "index": ("other", "approximate", "Index/ToC block."),
        "header": ("header", "exact", ""), "footer": ("footer", "exact", ""),
        "header_image": ("header", "approximate", "Image inside the running header."),
        "footer_image": ("footer", "approximate", "Image inside the running footer."),
        "page_number": ("page_number", "exact", ""),
        "aside_text": ("sidebar", "exact", ""),
        "ref_text": ("text", "approximate", "Reference/bibliography text."),
        "discarded": ("other", "ambiguous", "MinerU internal discard bucket."),
        "code": ("other", "approximate", ""), "code_body": ("other", "approximate", ""),
        "algorithm": ("other", "approximate", ""), "phonetic": ("other", "ambiguous", ""),
        "vertical_text": ("text", "approximate", "Vertically-set text run."),
        # PP-DocLayoutV2 label set as re-implemented in MinerU
        "figure_title": ("caption", "exact", ""),
        "number": ("page_number", "exact", ""),
        "vision_footnote": ("footnote", "exact",
                            "Footnote attached to a figure/chart/table, not a page footnote."),
        "display_formula": ("formula", "exact", ""),
        "inline_formula": ("formula", "approximate", "Inline math span, not a display equation."),
        "reference": ("text", "approximate", "Bibliography container."),
        "reference_content": ("text", "approximate", "Bibliography entry."),
        "seal": ("other", "ambiguous", "Stamp/seal imprint."),
        "content": ("other", "approximate", "Table-of-contents block."),
    },
    # ---- Surya layout (datalab) -----------------------------------------
    "surya": {
        "Caption": ("caption", "exact", ""), "Footnote": ("footnote", "exact", ""),
        "Formula": ("formula", "exact", ""), "List-item": ("list", "exact", ""),
        "Page-footer": ("footer", "exact", ""), "Page-header": ("header", "exact", ""),
        "Picture": ("figure", "exact", ""), "Figure": ("figure", "exact", ""),
        "Section-header": ("heading", "exact", ""), "Table": ("table", "exact", ""),
        "Text": ("text", "exact", ""), "Title": ("title", "exact", ""),
        "Text-inline-math": ("text", "approximate", "Text containing inline math."),
        "Code": ("other", "approximate", ""), "TableOfContents": ("other", "approximate", ""),
        "Handwriting": ("text", "approximate", ""), "Form": ("other", "ambiguous", ""),
        "PageHeader": ("header", "exact", ""), "PageFooter": ("footer", "exact", ""),
        "SectionHeader": ("heading", "exact", ""), "ListItem": ("list", "exact", ""),
        "Equation": ("formula", "exact", ""), "TextInlineMath": ("text", "approximate", ""),
        # Surya 2 public label set (surya.layout.label.LAYOUT_PRED_RELABEL targets).
        "ListGroup": ("list", "exact", "Surya 2 groups list items into one block."),
        "Diagram": ("figure", "exact", ""),
        "ChemicalBlock": ("figure", "approximate", "Chemical structure drawing."),
        "Bibliography": ("text", "approximate", "Reference-list block."),
        "BlankPage": ("other", "ambiguous", "Whole-page 'nothing here' prediction."),
        # Raw pre-relabel forms, in case a config reports raw labels.
        "Page-Header": ("header", "exact", ""), "Page-Footer": ("footer", "exact", ""),
        "Section-Header": ("heading", "exact", ""), "List-Group": ("list", "exact", ""),
        "Equation-Block": ("formula", "exact", ""), "Code-Block": ("other", "approximate", ""),
        "Complex-Block": ("figure", "approximate", "Mixed figure+text composite region."),
        "Chemical-Block": ("figure", "approximate", ""),
        "Table-Of-Contents": ("other", "approximate", ""),
        "Image": ("figure", "exact", ""), "Blank-Page": ("other", "ambiguous", ""),
    },
    # ---- PaddleOCR PP-DocLayout / PP-StructureV3 -------------------------
    "paddle": {
        "text": ("text", "exact", ""), "paragraph_title": ("heading", "exact", ""),
        "doc_title": ("title", "exact", ""), "abstract": ("text", "approximate", ""),
        "content": ("other", "approximate", "Table-of-contents block."),
        "figure_title": ("caption", "exact", ""), "chart_title": ("caption", "exact", ""),
        "table_title": ("caption", "exact", ""), "figure": ("figure", "exact", ""),
        "image": ("figure", "exact", ""), "chart": ("figure", "approximate", ""),
        "table": ("table", "exact", ""), "formula": ("formula", "exact", ""),
        "formula_number": ("formula", "approximate", ""),
        "header": ("header", "exact", ""), "footer": ("footer", "exact", ""),
        "header_image": ("header", "approximate", ""), "footer_image": ("footer", "approximate", ""),
        "number": ("page_number", "exact", ""), "page_number": ("page_number", "exact", ""),
        "footnote": ("footnote", "exact", ""), "reference": ("text", "approximate", ""),
        "algorithm": ("other", "approximate", ""), "seal": ("other", "ambiguous", ""),
        "aside_text": ("sidebar", "exact", ""), "vision_footnote": ("footnote", "exact", ""),
        "reference_content": ("text", "approximate", ""),
        "display_formula": ("formula", "exact", "Block-level equation."),
        "inline_formula": ("formula", "approximate", "Equation inside a text line; "
                           "a sub-line region, not a block."),
        "text_without_layout": ("text", "approximate", ""),
        "vertical_text": ("text", "approximate", ""), "region": ("other", "ambiguous", ""),
    },
    # ---- Eynollah (PAGE-XML region types) --------------------------------
    "eynollah": {
        "TextRegion":      ("text",      "exact", ""),
        "paragraph":       ("text",      "exact", ""),
        "heading":         ("heading",   "exact", ""),
        "header":          ("heading",   "approximate", "PAGE-XML 'header' is a headline within the text body, "
                                                        "not necessarily the running page header."),
        "catch-word":      ("other",     "ambiguous", ""),
        "page-number":     ("page_number", "exact", ""),
        "drop-capital":    ("other",     "ambiguous", ""),
        "credit":          ("caption",   "approximate", ""),
        "marginalia":      ("sidebar",   "exact", ""),
        "footnote":        ("footnote",  "exact", ""),
        "footnote-continued": ("footnote", "exact", ""),
        "caption":         ("caption",   "exact", ""),
        "endnote":         ("footnote",  "approximate", ""),
        "signature-mark":  ("other",     "ambiguous", ""),
        "other":           ("other",     "ambiguous", ""),
        "ImageRegion":     ("figure",    "exact", ""),
        "GraphicRegion":   ("figure",    "exact", ""),
        "TableRegion":     ("table",     "exact", ""),
        "SeparatorRegion": ("separator", "exact", ""),
        "MathsRegion":     ("formula",   "exact", ""),
        "NoiseRegion":     ("other",     "ambiguous", ""),
        "AdvertRegion":    ("other",     "ambiguous", ""),
        "ChartRegion":     ("figure",    "approximate", ""),
        "LineDrawingRegion": ("figure",  "exact", ""),
    },
    # ---- Dolphin (layout-stage element labels) ---------------------------
    "dolphin": {
        "title": ("title", "approximate", "Dolphin 'title' spans document title and section headings."),
        "sec": ("heading", "exact", ""), "sub_sec": ("heading", "exact", ""),
        "para": ("text", "exact", ""), "text": ("text", "exact", ""),
        "list": ("list", "exact", ""), "tab": ("table", "exact", ""), "table": ("table", "exact", ""),
        "fig": ("figure", "exact", ""), "figure": ("figure", "exact", ""),
        "cap": ("caption", "exact", ""), "caption": ("caption", "exact", ""),
        "formula": ("formula", "exact", ""), "equation": ("formula", "exact", ""),
        "fnote": ("footnote", "exact", ""), "foot": ("footer", "exact", ""),
        "head": ("header", "exact", ""), "page_number": ("page_number", "exact", ""),
        "alg": ("other", "approximate", ""), "watermark": ("other", "ambiguous", ""),
        "reference": ("text", "approximate", ""), "abstract": ("text", "approximate", ""),
        "author": ("text", "approximate", ""), "affiliation": ("text", "approximate", ""),
    },
    # ---- PDF-Extract-Kit layout (DocLayout-YOLO DocStructBench) ----------
    "pek_yolo": {
        "title": ("title", "approximate", "DocStructBench 'title' = any heading."),
        "plain text": ("text", "exact", ""), "abandon": ("other", "approximate",
            "DocStructBench 'abandon' aggregates headers, footers, page numbers and page footnotes; "
            "cannot be split into canonical header/footer/page_number."),
        "figure": ("figure", "exact", ""), "figure_caption": ("caption", "exact", ""),
        "table": ("table", "exact", ""), "table_caption": ("caption", "exact", ""),
        "table_footnote": ("footnote", "exact", ""),
        "isolate_formula": ("formula", "exact", ""),
        "formula_caption": ("formula", "approximate", "Equation number/caption."),
    },
    # ---- YOLOX / unstructured layout (PubLayNet-derived) -----------------
    "yolox": {
        "Caption": ("caption", "exact", ""), "Footnote": ("footnote", "exact", ""),
        "Formula": ("formula", "exact", ""), "List-item": ("list", "exact", ""),
        "Page-footer": ("footer", "exact", ""), "Page-header": ("header", "exact", ""),
        "Picture": ("figure", "exact", ""), "Section-header": ("heading", "exact", ""),
        "Table": ("table", "exact", ""), "Text": ("text", "exact", ""), "Title": ("title", "exact", ""),
    },
    # ---- NDLOCR v2 layout (ndl_layout, National Diet Library of Japan) ----
    #      17 classes mixing line-level and block-level regions.  The `line_*`
    #      classes are individual text *lines*, which is a finer granularity
    #      than any other system here; that is recorded in the notes rather
    #      than hidden, and the registry also offers a blocks-only run.
    #      Japanese terms per the NDLOCR v2 annotation guideline.
    "ndl": {
        "line_main":     ("text",        "approximate", "A single line of body text, not a text region."),
        "line_inote":    ("footnote",    "approximate", "割注 — an interlinear/split gloss set inside the "
                                                        "text; annotation apparatus, but not at the page foot."),
        "line_hnote":    ("sidebar",     "approximate", "頭注 — an annotation printed in the top margin "
                                                        "alongside the body text."),
        "line_caption":  ("caption",     "exact",       ""),
        "line_ad":       ("other",       "approximate", "A line of advertisement text; no canonical equivalent."),
        "block_fig":     ("figure",      "exact",       ""),
        "block_table":   ("table",       "exact",       ""),
        "block_pillar":  ("header",      "approximate", "柱 — the running head carrying the title/section "
                                                        "at the page edge."),
        "block_folio":   ("page_number", "exact",       "ノンブル."),
        "block_rubi":    ("other",       "approximate", "ルビ — phonetic gloss printed alongside a character run."),
        "block_chart":   ("figure",      "approximate", "Chart/graph is a figure sub-type."),
        "block_eqn":     ("formula",     "exact",       ""),
        "block_cfm":     ("formula",     "approximate", "化学式 — a chemical formula."),
        "block_eng":     ("text",        "approximate", "欧文 — a block of Western-script text."),
        "block_ad":      ("other",       "approximate", "Advertisement block; no canonical equivalent."),
        "text_block":    ("text",        "exact",       ""),
        "text_block_ad": ("other",       "approximate", "Advertisement text block."),
    },
    # ---- DocSAM with the canonical class set as its prompt ---------------
    #      DocSAM embeds class *names* as semantic queries, so the taxonomy is
    #      chosen at inference time.  These runs are prompted with this
    #      benchmark's own canonical class names, which makes the mapping an
    #      identity and removes the usual translation step entirely.  The one
    #      rewrite is "page number" -> page_number: the prompt is a natural
    #      language phrase, the canonical id is a slug.
    "docsam_canonical": {
        "title":       ("title",       "exact", ""),
        "heading":     ("heading",     "exact", ""),
        "text":        ("text",        "exact", ""),
        "list":        ("list",        "exact", ""),
        "table":       ("table",       "exact", ""),
        "figure":      ("figure",      "exact", ""),
        "caption":     ("caption",     "exact", ""),
        "formula":     ("formula",     "exact", ""),
        "header":      ("header",      "exact", ""),
        "footer":      ("footer",      "exact", ""),
        "footnote":    ("footnote",    "exact", ""),
        "sidebar":     ("sidebar",     "exact", ""),
        "page number": ("page_number", "exact", ""),
        "separator":   ("separator",   "exact", ""),
    },
    # ---- CDLA 10-class (RapidLayout's PP PicoDet CDLA checkpoint, and the
    #      360LayoutAnalysis "paper" YOLOv8n, which was trained on the same
    #      cdla_label.yaml).  Keys are lowercase; map_class also matches the
    #      capitalised/space-separated spellings the YOLO metadata uses.
    "cdla": {
        "text":          ("text",    "exact", ""),
        "title":         ("title",   "approximate", "CDLA 'title' covers document titles and section "
                                                    "headings alike; cannot be split without a second signal."),
        "figure":        ("figure",  "exact", ""),
        "figure_caption":("caption", "exact", ""),
        "table":         ("table",   "exact", ""),
        "table_caption": ("caption", "exact", ""),
        "header":        ("header",  "exact", ""),
        "footer":        ("footer",  "exact", ""),
        "reference":     ("text",    "approximate", "Bibliography block; body text by geometry."),
        "equation":      ("formula", "exact", ""),
    },
    # ---- 360LayoutAnalysis "general6" ------------------------------------
    "layout360_general6": {
        "Text":     ("text",    "exact", ""),
        "Title":    ("title",   "approximate", "Covers document titles and section headings alike."),
        "Figure":   ("figure",  "exact", ""),
        "Table":    ("table",   "exact", ""),
        "Caption":  ("caption", "exact", ""),
        "Equation": ("formula", "exact", ""),
    },
    # ---- 360LayoutAnalysis "report" (Chinese research reports) -----------
    #      Nine classes; kept separate from cdla because the label sets differ.
    "layout360_report": {
        "Text":          ("text",    "exact", ""),
        "Title":         ("title",   "approximate", "Covers document titles and section headings alike."),
        "Header":        ("header",  "exact", ""),
        "Footer":        ("footer",  "exact", ""),
        "Figure":        ("figure",  "exact", ""),
        "Table":         ("table",   "exact", ""),
        "Toc":           ("list",    "approximate", "Table of contents: a list of entries with leaders; "
                                                    "no canonical 'toc' class exists."),
        "Figure caption":("caption", "exact", ""),
        "Table caption": ("caption", "exact", ""),
    },
    # ---- kraken blla ------------------------------------------------------
    #      The bundled blla model has one baseline class ("default") and one
    #      region class ("text").  Line output is *line-level*: each region is
    #      a single text line, not a block, which is recorded in the note
    #      rather than hidden behind an "exact" mapping.
    "kraken": {
        "line:default": ("text", "approximate", "One text line (baseline environment polygon), "
                                                "not a text region."),
        "line:text":    ("text", "approximate", "One text line (baseline environment polygon), "
                                                "not a text region."),
        "region:text":  ("text", "approximate", "kraken's region head; on this corpus it returns "
                                                "connected components rather than layout blocks."),
    },
}


def map_class(system_taxonomy, source_class):
    """Return (canonical, confidence, note) for a source class."""
    table = MAPPINGS.get(system_taxonomy, {})
    if source_class in table:
        return table[source_class]
    lowered = {k.lower().replace("-", "_").replace(" ", "_"): v for k, v in table.items()}
    key = str(source_class).lower().replace("-", "_").replace(" ", "_")
    if key in lowered:
        return lowered[key]
    return ("other", "unmapped", f"Source class '{source_class}' has no entry in the '{system_taxonomy}' table.")


def mapping_rows():
    """Flat rows for the report's mapping table."""
    rows = []
    for tax, table in sorted(MAPPINGS.items()):
        for src, (canon, conf, note) in sorted(table.items()):
            rows.append({"taxonomy": tax, "source_class": src, "canonical_class": canon,
                         "confidence": conf, "notes": note})
    return rows
