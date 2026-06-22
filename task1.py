import re
import pymupdf
import pdfplumber
import json
import os
import hashlib
import pytesseract
import io
import sys


# SCAN OR DIGITAL

def if_digital(doc, threshold=0.7):
    total_pages = len(doc)

    if total_pages == 0:
        return False

    empty_pages = 0

    for i in range(total_pages):
        page = doc[i]

        try:
            if page.get_text().strip() == "":
                empty_pages += 1
        except Exception:
            empty_pages += 1

    return (empty_pages / total_pages) < threshold


# OCR

def ocr_extract_text(doc, dpi=300):
    page_texts = []
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    for i in range(len(doc)):
        page = doc[i]
        try:
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            page_texts.append(text)
        except Exception:
            page_texts.append("")
    return page_texts


# TEXT NORMALISATION

_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def normalize_chars(text):
    for bad, good in _LIGATURE_MAP.items():
        text = text.replace(bad, good)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_for_comparison(text):
    t = text.lower().strip()
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\s+", " ", t)
    return t


# HEADER FOOTER STRIP

def detect_repeated_lines(pages, edge_count=2, min_page_fraction=0.5, min_pages=3):
    total_pages = len(pages)
    if total_pages < min_pages:
        return set()

    counts = {}
    for lines in pages:
        if not lines:
            continue
        edge_lines = lines[:edge_count] + lines[-edge_count:]
        seen_this_page = set()
        for line in edge_lines:
            text = line["text"]
            if len(text.split()) > 12:
                continue
            norm = normalize_for_comparison(text)
            if not norm or norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            counts[norm] = counts.get(norm, 0) + 1

    threshold = max(min_pages, int(total_pages * min_page_fraction))
    return {norm for norm, c in counts.items() if c >= threshold}


def strip_repeated_headers_footers(pages, edge_count=2):
    repeated = detect_repeated_lines(pages, edge_count=edge_count)
    if not repeated:
        return pages

    cleaned_pages = []
    for lines in pages:
        n = len(lines)
        keep = []
        for i, line in enumerate(lines):
            near_edge = i < edge_count or i >= n - edge_count
            if near_edge and normalize_for_comparison(line["text"]) in repeated:
                continue
            keep.append(line)
        cleaned_pages.append(keep)
    return cleaned_pages


# HYPHEN JOINING

_HYPHEN_BREAK_RE = re.compile(r"(\w)-$")


def join_hyphenated_breaks(flat_lines):
    merged = []
    i = 0
    n = len(flat_lines)
    while i < n:
        current = dict(flat_lines[i])
        text = current["text"]

        while i + 1 < n:
            m = _HYPHEN_BREAK_RE.search(text)
            if not m:
                break
            nxt = flat_lines[i + 1]["text"]
            if not nxt or not nxt[0].isalpha() or not nxt[0].islower():
                break
            text = text[: m.start(1) + 1] + nxt
            i += 1

        current["text"] = text
        merged.append(current)
        i += 1

    return merged


# OCR CLEANING

def clean_ocr_pages(page_texts):
    pages = []
    for page_number, text in enumerate(page_texts, start=1):
        lines = [
            {"text": normalize_chars(raw), "page": page_number, "is_heading": False}
            for raw in text.splitlines()
            if raw.strip()
        ]
        pages.append(lines)

    pages = strip_repeated_headers_footers(pages)
    flat_lines = [line for page in pages for line in page]
    flat_lines = join_hyphenated_breaks(flat_lines)

    grouped = {}
    for line in flat_lines:
        grouped.setdefault(line["page"], []).append(line["text"])

    cleaned_page_texts = []
    for page_number in range(1, len(page_texts) + 1):
        cleaned_page_texts.append(" ".join(grouped.get(page_number, [])))

    return cleaned_page_texts


# CHUNK ID

def make_chunk_id(pdf_path, chunk_text, page_start, page_end):
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    raw = (
        f"{page_start}_"
        f"{page_end}_"
        f"{chunk_text[:200]}"
    )

    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    return f"{base}_{digest}"


# PLAIN TEXT CHUNKING

def chunk_plain_text(page_texts, pdf_path, chunk_size=400, overlap=70):
    tagged_words = []
    for page_number, text in enumerate(page_texts, start=1):
        for word in text.split():
            tagged_words.append((word, page_number, None))

    full_text = " ".join(w for w, _, _ in tagged_words)
    step = chunk_size - overlap
    chunks = []
    i = 0

    while i < len(tagged_words):
        window = tagged_words[i: i + chunk_size]
        if not window:
            break
        words = [w for w, _, _ in window]
        pages = [p for _, p, _ in window]

        chunk_text = " ".join(words)

        chunks.append({
            "chunk_id": make_chunk_id(
                pdf_path,
                chunk_text,
                min(pages),
                max(pages)
            ),
            "text": chunk_text,
            "word_count": len(words),
            "page_start": min(pages),
            "page_end": max(pages),
            "section_heading": None,
            "type": "text",
        })
        i += step

    return full_text, chunks





# TITLE

def extract_title(doc):
    try:
        title = doc.metadata.get("title", "").strip()
        if title:
            return normalize_chars(title)
    except Exception:
        pass

    try:
        if len(doc) == 0:
            return None

        page = doc[0]
        title_words = []
        max_size = None

        for block in page.get_text("dict").get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line.get("spans", [])
                if not spans:
                    continue

                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue

                sizes = [s.get("size") for s in spans if s.get("size") is not None]
                if not sizes:
                    continue
                size = max(sizes)

                if max_size is None:
                    max_size = size

                if max_size - size > 2 or text.endswith("."):
                    return " ".join(title_words) if title_words else None

                title_words.append(normalize_chars(text))

        return " ".join(title_words) if title_words else None

    except Exception:
        return None


# AUTHORS

def extract_authors(doc):
    try:
        raw = doc.metadata.get("author", "").strip()
        if raw:
            return [normalize_chars(a.strip()) for a in raw.split(",") if a.strip()]
    except Exception:
        pass

    try:
        if len(doc) == 0:
            return []

        lines = []
        for block in doc[0].get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                sizes = [span.get("size") for span in spans if span.get("size") is not None]
                if not sizes:
                    continue
                lines.append((text, max(sizes)))

        if not lines:
            return []

        title_size = max(size for _, size in lines)
        smaller_sizes = [size for _, size in lines if size < title_size]
        if not smaller_sizes:
            return []
        author_size = smaller_sizes[0]

        authors = []
        for text, size in lines:
            if size == author_size and not any(
                word in text.lower()
                for word in ["department", "university", "email", "@"]
            ):
                authors.append(normalize_chars(text))

        return authors

    except Exception:
        return []


# LINE EXTRACTION WITH HEADING DETECTION

def extract_lines_with_metadata(doc, table_bboxes=None, body_size_threshold_delta=1.5):
    if table_bboxes is None:
        table_bboxes = {}

    pages = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        page_table_bboxes = table_bboxes.get(page_number, [])
        blocks = page.get_text("dict")["blocks"]
        sizes = []

        for block in blocks:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))

        body_size = max(set(sizes), key=sizes.count) if sizes else None

        page_lines = []
        for block in blocks:
            if "lines" not in block:
                continue

            bx0, by0, bx1, by1 = block["bbox"]
            in_table = any(
                tx0 <= bx0 and ty0 <= by0 and bx1 <= tx1 and by1 <= ty1
                for (tx0, ty0, tx1, ty1) in page_table_bboxes
            )
            if in_table:
                continue

            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text:
                    continue
                line_size = max(s["size"] for s in line["spans"])

                is_heading = (
                    body_size is not None
                    and line_size - body_size >= body_size_threshold_delta
                    and len(text.split()) <= 15
                )

                page_lines.append({
                    "text": normalize_chars(text),
                    "page": page_number,
                    "is_heading": is_heading,
                })

        pages.append(page_lines)

    return pages


# WORD EXTRACTION WITH METADATA

def extract_words_with_metadata(doc, table_bboxes=None, body_size_threshold_delta=1.5):
    pages = extract_lines_with_metadata(doc, table_bboxes, body_size_threshold_delta)
    pages = strip_repeated_headers_footers(pages)
    flat_lines = [line for page in pages for line in page]
    flat_lines = join_hyphenated_breaks(flat_lines)

    tagged_words = []
    current_heading = None
    for line in flat_lines:
        if line["is_heading"]:
            current_heading = line["text"]
        for word in line["text"].split():
            tagged_words.append((word, line["page"], current_heading))

    return tagged_words


# DIGITAL TEXT CHUNKING

def extract_text(doc, pdf_path, table_bboxes=None, chunk_size=400, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    tagged_words = extract_words_with_metadata(doc, table_bboxes)
    full_text = " ".join(w for w, _, _ in tagged_words)

    step = chunk_size - overlap
    chunks = []
    i = 0

    while i < len(tagged_words):
        window = tagged_words[i: i + chunk_size]
        if not window:
            break

        words = [w for w, _, _ in window]
        pages = [p for _, p, _ in window]
        headings = [h for _, _, h in window if h]

        chunk_text = " ".join(words)

        chunk = {
            "chunk_id": make_chunk_id(
                pdf_path,
                chunk_text,
                min(pages),
                max(pages)
            ),
            "text": chunk_text,
            "word_count": len(words),
            "page_start": min(pages),
            "page_end": max(pages),
            "section_heading": max(set(headings), key=headings.count) if headings else None,
            "type": "text",
        }

        chunks.append(chunk)
        i += step

    return full_text, chunks


# WORD COUNT

def word_count(text):
    return len(text.split())


# TABLE DETECTION 

TABLE_SETTINGS_LINES = {"vertical_strategy": "lines", "horizontal_strategy": "lines"} 

TABLE_SETTINGS_TEXT  = {"vertical_strategy": "text",  "horizontal_strategy": "text", 

                        "snap_tolerance": 4, "join_tolerance": 4} 

  

def _find_page_tables(page): 

    # ruled tables first; if none, fall back to text-aligned (borderless) 

    tables = page.find_tables(TABLE_SETTINGS_LINES) 

    if not tables: 

        tables = page.find_tables(TABLE_SETTINGS_TEXT) 

    return tables 

  

def _looks_like_table(rows): 

    # guard against the text-strategy turning prose / multi-column text into a "table" 

    if not rows or len(rows) < 2: 

        return False 

    ncols = max(len(r) for r in rows) 

    if ncols < 2: 

        return False 

    filled = sum(1 for r in rows for c in r if c and c.strip()) 

    total  = sum(len(r) for r in rows) or 1 

    return filled / total >= 0.5 

  

def extract_tables_and_bboxes(pdf_path): 


    table_chunks, bboxes = [], {} 

    try: 

        with pdfplumber.open(pdf_path) as pdf: 

            for page_number, page in enumerate(pdf.pages, start=1): 

                page_bboxes = [] 

                for t in _find_page_tables(page): 

                    rows = t.extract() 

                    if not _looks_like_table(rows): 

                        continue 

                    page_bboxes.append(t.bbox)          # same table feeds the mask 

                    text = "\n".join( 

                        " | ".join((c or "").strip() for c in row) for row in rows 

                    ) 

                    table_chunks.append({ 

                        "chunk_id": make_chunk_id(pdf_path, text, page_number, page_number), 

                        "text": text, 

                        "word_count": len(text.split()), 

                        "page_start": page_number, 

                        "page_end": page_number, 

                        "section_heading": None, 

                        "type": "table", 

                    }) 

                if page_bboxes: 

                    bboxes[page_number] = page_bboxes 

    except Exception: 

        pass 

    return table_chunks, bboxes 

# FIGURE COUNT

def count_figures(doc):
    count = 0
    for page in doc:
        count += len(page.get_images())
    return count


# MAIN

def main(pdf_path: str | None = None) -> dict | None:

    if pdf_path is None:                   
        if len(sys.argv) < 2:
            print("usage: python script.py <pdf_path>")
            return None
        pdf_path = sys.argv[1]              

    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return None

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Failed to open pdf: {e}")
        return None

    if doc.needs_pass:
        print("pdf has password")
        doc.close()
        return None

    page_count = len(doc)
    if page_count == 0:
        print("pdf has no pages")
        doc.close()
        return None
    
    is_digital   = if_digital(doc)
    title        = extract_title(doc)
    authors      = extract_authors(doc)
    tables, table_bboxes = extract_tables_and_bboxes(pdf_path)

    ocr_used = False
    if is_digital:
        text, chunks = extract_text(doc, pdf_path, table_bboxes, chunk_size=400, overlap=50)
    else:
        print("PDF appears to be scanned — running OCR...")
        page_texts = ocr_extract_text(doc, dpi=300)
        page_texts = clean_ocr_pages(page_texts)
        text, chunks = chunk_plain_text(page_texts, pdf_path, chunk_size=400, overlap=70)
        ocr_used = True

    figure_count = count_figures(doc)

   

    doc.close()

    all_chunks = chunks + tables                
    print("Is Digital  :", is_digital)
    print("OCR Used    :", ocr_used)
    print("Page Count  :", page_count)
    print("Title       :", title)
    print("Authors     :", authors)
    print("Word Count  :", word_count(text))
    print("Chunk Count :", len(all_chunks))
    print("Figure Count:", figure_count)

    pdf_data = {
        "is_digital":   is_digital,
        "ocr_used":     ocr_used,
        "page_count":   page_count,
        "title":        title,
        "authors":      authors if isinstance(authors, list) else [authors],
        "word_count":   word_count(text),
        "chunks":       all_chunks,            
        "table_count":  len(tables),            
        "figure_count": figure_count,
    }

    with open("pdf_output.json", "w", encoding="utf-8") as f:
        json.dump(pdf_data, f, indent=4, ensure_ascii=False)

    print("pdf_output.json saved")
    return pdf_data


if __name__ == "__main__":
    main()


    
