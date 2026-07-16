import re
import pymupdf
import pdfplumber
import json
import hashlib
import pytesseract
import io
import sys
from collections import defaultdict
from typing import Optional
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
_groq_client = None
SHORT_DOC_LIMIT = 6000



def _get_groq_client():
    global _groq_client

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set."
            )

        _groq_client = Groq(api_key=api_key)

    return _groq_client


def llm(prompt: str, model: str = None) -> str:
    try:
        client = _get_groq_client()

        resp = client.chat.completions.create(
            model=model or GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq error: {e}")
        return ""


def _parse_summary_json(raw: str) -> dict:
    cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return {
            "overview": data.get("overview", ""),
            "key_points": data.get("key_points", []),
            "topics": data.get("topics", []),
        }
    except Exception as e:
        print(f"Summary JSON parse failed: {e}")
        return {"overview": raw.strip(), "key_points": [], "topics": []}
    


def _summarize_single(text: str, title: str = "") -> dict:
    """Single-pass summary for short docs."""
    prompt = f"""You are a research assistant. Analyze the following document.
Respond ONLY with valid JSON (no markdown, no preamble, no explanation) in this EXACT shape:
{{
  "overview": "2-3 sentence summary covering objective, methods, findings, and conclusions, under 200 words",
  "key_points": ["short factual bullet 1", "short factual bullet 2", "..."],
  "topics": ["topic1", "topic2", "..."]
}}

Title: {title}

Document:
{text}

JSON:"""
    raw = llm(prompt)
    return _parse_summary_json(raw)


def _summarize_chunk(chunk_text: str, index: int, total: int) -> str:
    """Summarize one chunk during map phase."""
    prompt = f"""Summarize this section (part {index}/{total}) of a research document in 3-4 sentences.
Focus on key facts, methods, or findings mentioned.

Section:
{chunk_text}

Summary:"""
    return llm(prompt)


def _summarize_reduce(partial_summaries: list[str], title: str = "") -> dict:
    BATCH_SIZE = 5  

    while len(partial_summaries) > BATCH_SIZE:
        batched = []
        for i in range(0, len(partial_summaries), BATCH_SIZE):
            batch = partial_summaries[i:i+BATCH_SIZE]
            combined = "\n\n".join(
                f"[Part {j+1}]: {s}" for j, s in enumerate(batch) if s
            )
            prompt = f"""Combine these section summaries into one paragraph (max 100 words):

{combined}

Combined Summary:"""
            print(f"  Batch reducing {i//BATCH_SIZE + 1}...")
            result = llm(prompt)
            if result:
                batched.append(result)
        partial_summaries = batched

    combined = "\n\n".join(
        f"[Part {i+1}]: {s}" for i, s in enumerate(partial_summaries) if s
    )
    prompt = f"""You are a research assistant. Below are section-wise summaries of a document.
Respond ONLY with valid JSON (no markdown, no preamble, no explanation) in this EXACT shape:
{{
  "overview": "single coherent paragraph covering objective, methods, findings, and conclusions, under 250 words",
  "key_points": ["short factual bullet 1", "short factual bullet 2", "..."],
  "topics": ["topic1", "topic2", "..."]
}}

Title: {title}

Section Summaries:
{combined}

JSON:"""
    raw = llm(prompt)
    return _parse_summary_json(raw)

    

def generate_summary(full_text: str, chunks: list, title: str = "") -> dict:
    wc = len(full_text.split())
    print(f"Generating summary (strategy: {'single-pass' if wc <= SHORT_DOC_LIMIT else 'map-reduce'}, {wc} words)...")

    if wc <= SHORT_DOC_LIMIT:
        return _summarize_single(full_text, title)

    text_chunks = [c for c in chunks if c.get("type") == "text"]
    total = len(text_chunks)
    partial = []
    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Summarizing chunk {i}/{total}...")
        s = _summarize_chunk(chunk["text"], i, total)
        if s:
            partial.append(s)

    print("  Reducing to final summary...")
    return _summarize_reduce(partial, title)

# VECTOR DB
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")  
    return _embedder


def get_chroma_collection(db_path="./chroma_db", collection_name="rag_docs"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    assert collection.metadata.get("hnsw:space") == "cosine", \
        f"Expected cosine space, got: {collection.metadata}"
    return collection


def upsert_chunks_to_chroma(chunks, collection):
    embedder = get_embedder()

    ids       = [c["chunk_id"] for c in chunks]
    texts     = [c["text"] for c in chunks]
    metadatas = [
    {
        "doc_id":          c.get("doc_id", ""),
        "type":            c.get("type", "text"),
        "page_start":      c.get("page_start", 0),
        "page_end":        c.get("page_end", 0),
        "section_heading": c.get("section_heading") or "",
        "word_count":      c.get("word_count", 0),
    }
  
        for c in chunks
    ]

    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    batch = 400
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start:start+batch],
            documents=texts[start:start+batch],
            embeddings=embeddings[start:start+batch],
            metadatas=metadatas[start:start+batch],
        )

    print(f"inserted {len(ids)} chunks → ChromaDB")


def query_chroma(query_text, collection, n_results=5, chunk_type=None, doc_id=None):
    embedder = get_embedder()
    query_vec = embedder.encode([query_text], normalize_embeddings=True).tolist()

    conditions = []
    if chunk_type:
        conditions.append({"type": chunk_type})
    if doc_id:
        conditions.append({"doc_id": doc_id})

    if len(conditions) > 1:
        where = {"$and": conditions}
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = None

    results = collection.query(
        query_embeddings=query_vec,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "meta": meta, "score": 1 - dist})  # cosine → similarity

    return hits

# RELATIONAL DB
def init_documents_table():
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    doc_name    TEXT,
    title       TEXT,
    authors     TEXT,
    page_count  INTEGER,
    word_count  INTEGER,
    chunk_count INTEGER,
    table_count INTEGER,
    figure_count INTEGER,
    summary     TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
)
""")
    conn.commit()
    conn.close()


def store_document_record(doc_id, doc_name, title, authors, page_count,
                          word_count, chunk_count, table_count, figure_count, summary):
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("""
INSERT INTO documents (
    doc_id, doc_name, title, authors, page_count,
    word_count, chunk_count, table_count, figure_count, summary
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(doc_id) DO UPDATE SET
    doc_name = excluded.doc_name,
    title = excluded.title,
    authors = excluded.authors,
    page_count = excluded.page_count,
    word_count = excluded.word_count,
    chunk_count = excluded.chunk_count,
    table_count = excluded.table_count,
    figure_count = excluded.figure_count,
    summary = excluded.summary;
""", (
    doc_id,
    doc_name,
    title,
    json.dumps(authors),
    page_count,
    word_count,
    chunk_count,
    table_count,
    figure_count,
    json.dumps(summary),
))
    conn.commit()
    conn.close()


def store_chunks_db(chunks):
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()

    # table with doc_id
    cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks(
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    doc_name TEXT, 
    text TEXT,
    page_start INTEGER,
    page_end INTEGER,
    word_count INTEGER,
    section_heading TEXT,
    type TEXT
)
""")

    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(chunks)")]
    if "doc_id" not in existing_cols:
        cursor.execute("ALTER TABLE chunks ADD COLUMN doc_id TEXT")

    if "doc_name" not in existing_cols:
        cursor.execute("ALTER TABLE chunks ADD COLUMN doc_name TEXT")

    for chunk in chunks:
        cursor.execute("""
INSERT OR REPLACE INTO chunks (
    chunk_id,
    doc_id,
    doc_name,
    text,
    page_start,
    page_end,
    word_count,
    section_heading,
    type
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
            chunk["chunk_id"],
            chunk.get("doc_id", ""),  
            chunk.get("doc_name", ""),
            chunk["text"],
            chunk["page_start"],
            chunk["page_end"],
            chunk["word_count"],
            chunk["section_heading"],
            chunk["type"],
        ))

    conn.commit()
    conn.close()
    
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


# TITLE CASE DETECTION
def _is_title_case(text):
    words = text.split()
    if not words:
        return False
    for w in words:
        core = re.sub(r"[^A-Za-z]", "", w)
        if not core:
            continue
        if core.isupper():         
            continue
        if not core[0].isupper():
            return False
    return True

_CAPTION_RE = re.compile(r"^(TABLE|FIG(?:URE)?\.?|ALGORITHM|EQ(?:UATION)?\.?)\s*[\dIVXLC]+\b", re.IGNORECASE)

def _is_caption(text):
  
    return bool(_CAPTION_RE.match(text.strip()))


def _classify_heading_level(text):
    if re.match(r"^[IVX]+\.\s+[A-Za-z]", text):
        return "major"
    if re.match(r"^[A-Z]\.\s+[A-Za-z]", text):
        return "minor"
    if re.match(r"^\d+\.\d", text):
        return "minor"
    if re.match(r"^\d+\.?\s+[A-Za-z]", text):
        return "major"
    if text.isupper() and len(text.split()) <= 8:
        return "major"
    return "minor"


# SECTION HEADING DETECTION
def is_likely_heading(text, line_size, body_size, is_bold):
    if not text or len(text.split()) > 8:
        return False
    
    if _is_caption(text):         
        return False

    size_signal      = body_size is not None and (line_size - body_size) >= 1.5
    numbered_signal   = bool(re.match(r"^\d+(\.\d+){0,2}\.?\s+[A-Za-z]", text))
    roman_signal      = bool(re.match(r"^[IVX]+\.\s+[A-Za-z]", text))
    letter_signal     = bool(re.match(r"^[A-Z]\.\s+[A-Za-z]", text))
    all_caps_signal   = text.isupper() and len(text.split()) <= 8
    title_case_signal = _is_title_case(text) and len(text.split()) <= 8

    score = 0
    if size_signal:      score += 1
    if is_bold:           score += 1
    if numbered_signal:   score += 2
    if roman_signal:      score += 2
    if letter_signal:     score += 2
    if all_caps_signal:   score += 2
    if title_case_signal: score += 1

    return score >= 3


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
    raw = f"{page_start}_{page_end}_{chunk_text[:200]}"
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

def extract_lines_with_metadata(doc, table_bboxes=None, title_text=None):
    if table_bboxes is None:
        table_bboxes = {}

    all_sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        all_sizes.append(round(span["size"], 1))
    body_size = max(set(all_sizes), key=all_sizes.count) if all_sizes else None

    title_norm = normalize_for_comparison(title_text) if title_text else None

    gate_patterns = ("abstract", "introduction", "background", "motivation")
    content_started = False  

    pages = []
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        page_table_bboxes = table_bboxes.get(page_number, [])
        blocks = page.get_text("dict")["blocks"]

       
        mid_x = page.rect.width / 2

        def _col_sort_key(block):
            bx0, by0, bx1, by1 = block["bbox"]
            center = (bx0 + bx1) / 2
            col = 0 if center < mid_x else 1   
            return (col, by0)                   

        blocks = sorted(blocks, key=_col_sort_key)

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

                spans = line["spans"]
                line_size = max(s["size"] for s in spans)
                is_bold = any(
                    "bold" in s.get("font", "").lower()
                    for s in spans
                )

                norm = normalize_for_comparison(text)

                
                is_title_line = (
                    page_number == 1
                    and title_norm
                    and norm in title_norm
                )

                if not content_started:
                    if any(p in norm for p in gate_patterns):
                        content_started = True
                    elif page_number >= 3:
                      
                        content_started = True

                is_heading = (
                    not is_title_line
                    and content_started
                    and is_likely_heading(text, line_size, body_size, is_bold)
                )

                heading_level = _classify_heading_level(text) if is_heading else None

                page_lines.append({
                    "text": normalize_chars(text),
                    "page": page_number,
                    "is_heading": is_heading,
                    "heading_level": heading_level,
                })

        pages.append(page_lines)

    return pages

# WORD EXTRACTION WITH METADATA

def extract_words_with_metadata(doc, table_bboxes=None, title_text=None):
    pages = extract_lines_with_metadata(doc, table_bboxes, title_text=title_text)
    pages = strip_repeated_headers_footers(pages)
    flat_lines = [line for page in pages for line in page]
    flat_lines = join_hyphenated_breaks(flat_lines)

    tagged_words = []
    current_major = None
    current_minor = None

    for line in flat_lines:
        text = line["text"]

        # synthetic sections 
        if re.match(r"^abstract\s*[-—]", text, re.IGNORECASE):
            current_major, current_minor = "Abstract", None
        elif re.match(r"^index terms\s*[-—]", text, re.IGNORECASE):
            current_major, current_minor = "Index Terms", None
        elif line["is_heading"]:
            if line["heading_level"] == "major":
                current_major = text
                current_minor = None         
            else:
                current_minor = text

        combined = current_major
        if current_minor:
            combined = f"{current_major} > {current_minor}" if current_major else current_minor

        for word in text.split():
            tagged_words.append((word, line["page"], combined))

    return tagged_words

# DIGITAL TEXT CHUNKING

def extract_text(doc, pdf_path, table_bboxes=None, chunk_size=400, overlap=50, title_text=None):
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    tagged_words = extract_words_with_metadata(doc, table_bboxes, title_text=title_text)
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
TABLE_SETTINGS_TEXT  = {"vertical_strategy": "text", "horizontal_strategy": "text",
                        "snap_tolerance": 4, "join_tolerance": 4}
TABLE_SETTINGS_HLINES = {"vertical_strategy": "text", "horizontal_strategy": "lines",
                         "snap_tolerance": 4, "join_tolerance": 4}

def _looks_like_table(rows, min_cols=2, min_rows=2, max_rows=15, min_fill=0.4, debug=False):
    if not rows or len(rows) < min_rows:
        return False
    if len(rows) > max_rows:
        if debug: print(f"    REJECTED: too many rows ({len(rows)})")
        return False
    ncols = max(len(r) for r in rows)
    if ncols < min_cols:
        return False
   
    filled = sum(1 for r in rows for c in r if c and c.strip())
    total  = sum(len(r) for r in rows) or 1
    if filled / total < min_fill:
        if debug: print(f"    REJECTED: low fill rate")
        return False
    long_cells = sum(1 for r in rows for c in r if c and len(c.split()) > 6)
    if long_cells / total > 0.15:
        if debug: print(f"    REJECTED: long cells ({long_cells}/{total})")
        return False
    if ncols <= 4:
        avg_words = sum(len(c.split()) for r in rows for c in r if c and c.strip()) / max(filled, 1)
        if avg_words > 5:
            if debug: print(f"    REJECTED: avg words too high ({avg_words:.1f})")
            return False
    sparse_rows = sum(1 for r in rows if sum(1 for c in r if c and c.strip()) <= 1)
    if sparse_rows / len(rows) > 0.25:
        if debug: print(f"    REJECTED: sparse rows ({sparse_rows}/{len(rows)})")
        return False
    if debug: print(f"    ACCEPTED: {len(rows)} rows x {ncols} cols")
    return True

def _cluster_values(values, gap):
    if not values:
        return []
    values = sorted(set(round(v, 1) for v in values))
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _words_to_grid(words, col_gap=15, row_gap=4):
    if not words:
        return []
    ys = _cluster_values([w["top"] for w in words], gap=row_gap)
    xs = _cluster_values([w["x0"] for w in words], gap=col_gap)
    if len(xs) < 2:
        return []

    def nearest(val, centers):
        return min(range(len(centers)), key=lambda i: abs(centers[i] - val))

    grid = defaultdict(lambda: defaultdict(list))
    for w in words:
        r = nearest(w["top"], ys)
        c = nearest(w["x0"], xs)
        grid[r][c].append(w["text"])

    return [
        [" ".join(grid[r].get(c, [])) for c in range(len(xs))]
        for r in sorted(grid)
    ]


def _bboxes_overlap(b1, b2, tolerance=20):
  
    if abs(b1[0] - b2[0]) < tolerance and abs(b1[1] - b2[1]) < tolerance:
        return True
    # Check if vertical ranges overlap 
    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    y_range = max(b1[3] - b1[1], b2[3] - b2[1], 1)
    if y_overlap / y_range > 0.5:
        # Also check horizontal overlap
        x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        x_range = max(b1[2] - b1[0], b2[2] - b2[0], 1)
        if x_overlap / x_range > 0.3:
            return True
    return False


def _find_line_bounded_tables(page, min_lines=2):
    
    mid_x = page.width / 2
    h_lines = [
        l for l in page.lines
        if abs(l["x1"] - l["x0"]) > 20
        and abs(l["top"] - l["bottom"]) < 2
    ]
    if len(h_lines) < min_lines:
        return []

    # Classify lines by column side
    def side_of(line):
        if line["x1"] <= mid_x + 10:
            return "left"
        elif line["x0"] >= mid_x - 10:
            return "right"
        return "full"

    # Group lines by side
    groups = {"left": [], "right": [], "full": []}
    for l in h_lines:
        s = side_of(l)
        groups[s].append(l)

    results = []
    for side, lines in groups.items():
        if len(lines) < min_lines:
            continue
        lines = sorted(lines, key=lambda l: l["top"])
        # Cluster lines 
        clusters = []
        current = [lines[0]]
        for l in lines[1:]:
            if l["top"] - current[-1]["top"] < 80:
                current.append(l)
            else:
                clusters.append(current)
                current = [l]
        clusters.append(current)

        for cluster in clusters:
            if len(cluster) < min_lines:
                continue
            
            x0 = min(l["x0"] for l in cluster) - 2
            x1 = max(l["x1"] for l in cluster) + 2
            y0 = min(l["top"] for l in cluster) - 4
            y1 = max(l["top"] for l in cluster) + 25 

            try:
                crop = page.crop((x0, y0, x1, y1))
                text = crop.extract_text()
            except Exception:
                continue
            if not text or not text.strip():
                continue

            # Parse into rows
            raw_lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
            if len(raw_lines) < 2:
                continue

            
            rows = [re.split(r"  +", ln) for ln in raw_lines]
            col_counts = [len(r) for r in rows]
            # If 2+ space split doesn't work, try single space
            if max(col_counts) < 2:
                rows = [ln.split() for ln in raw_lines]

            if max(len(r) for r in rows) < 2:
                continue

           
            while len(rows) > 2:
                last_row = rows[-1]
                text_in_row = " ".join(last_row)
           
                words = text_in_row.split()
                num_count = sum(1 for w in words if re.match(r"^[\d.+\-∆]+$", w))
                if len(words) > 5 and num_count == 0:
                    rows.pop()
                else:
                    break

            bbox = (x0, y0, x1, y1)
            results.append((bbox, rows))

    return results


def _rows_to_text(rows):
    return "\n".join(" | ".join((c or "").strip() for c in row) for row in rows)


def _expand_multiline_cells(rows):
   
    expanded = []
    for row in rows:
        cells = [(c or "").strip() for c in row]
        # Check if any cell has newlines
        max_lines = max((c.count("\n") + 1) for c in cells) if cells else 1
        if max_lines <= 1:
            expanded.append(cells)
        else:
            # Split each cell by newlines and zip them into rows
            split_cells = [c.split("\n") for c in cells]
            for i in range(max_lines):
                new_row = []
                for sc in split_cells:
                    new_row.append(sc[i].strip() if i < len(sc) else "")
                expanded.append(new_row)
    return expanded


def _try_split_single_col_table(rows):
    
    if not rows:
        return rows
    ncols = max(len(r) for r in rows)
    if ncols != 1:
        return rows
    
    for sep_pattern in [r"  +", r" +"]:
        split_rows = []
        for row in rows:
            cell = row[0] if row else ""
            parts = re.split(sep_pattern, cell)
            split_rows.append(parts)
        
        col_counts = [len(r) for r in split_rows if any(c.strip() for c in r)]
        if not col_counts:
            continue
        
        if min(col_counts) >= 2:
            return split_rows
    return rows


def _make_table_chunk(pdf_path, text, page_number):
    return {
        "chunk_id":        make_chunk_id(pdf_path, text, page_number, page_number),
        "text":            text,
        "word_count":      len(text.split()),
        "page_start":      page_number,
        "page_end":        page_number,
        "section_heading": None,
        "type":            "table",
    }


def extract_tables_and_bboxes(pdf_path):
    table_chunks, bboxes = [], {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_bboxes = []
                seen_bboxes = []

                def _register(bbox, rows, _seen=seen_bboxes, _page_bboxes=page_bboxes):
                    if any(_bboxes_overlap(bbox, s) for s in _seen):
                        return
                    _seen.append(bbox)
                    _page_bboxes.append(bbox)
                    table_chunks.append(
                        _make_table_chunk(pdf_path, _rows_to_text(rows), page_number)
                    )

                    # Strategy 1 — ruled (vertical + horizontal lines)
                for t in page.find_tables(TABLE_SETTINGS_LINES):
                    rows = t.extract()
                    if _looks_like_table(rows):
                        _register(t.bbox, rows)

                # Strategy 2 — horizontal lines + text columns 
                for t in page.find_tables(TABLE_SETTINGS_HLINES):
                    rows = _expand_multiline_cells(t.extract())
                    if _looks_like_table(rows):
                        _register(t.bbox, rows)

                # Strategy 2b — crop into left/right halves for two-column layouts
                mid_x = page.width / 2
                for crop_bbox in [(0, 0, mid_x + 5, page.height), (mid_x - 5, 0, page.width, page.height)]:
                    cropped = page.crop(crop_bbox)
                    for t in cropped.find_tables(TABLE_SETTINGS_HLINES):
                        rows = _expand_multiline_cells(t.extract())
                        rows = _try_split_single_col_table(rows)
                        if _looks_like_table(rows):
                            _register(t.bbox, rows)

                # Strategy 3 — text aligned, skip full-page detections
                for t in page.find_tables(TABLE_SETTINGS_TEXT):
                    rows = t.extract()
                    _, y0, _, y1 = t.bbox
                    if (y1 - y0) / page.height > 0.6:
                        continue
                    if _looks_like_table(rows):
                        _register(t.bbox, rows)

                # Strategy 4 — line-guided extraction for horizontal-rule tables
                for bbox, rows in _find_line_bounded_tables(page):
                    if _looks_like_table(rows):
                        _register(bbox, rows)

                if page_bboxes:
                    bboxes[page_number] = page_bboxes

    except Exception as e:
        print(f"Table extraction error: {e}")

    return table_chunks, bboxes

         
# FIGURE COUNT

def count_figures(doc):
    count = 0
    for page in doc:
        count += len(page.get_images())
    return count


# MAIN

def main(pdf_path: Optional[str] = None) -> Optional[dict]:

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

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
        doc_id = hashlib.md5(file_bytes).hexdigest()[:12]

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
        text, chunks = extract_text(doc, pdf_path, table_bboxes, chunk_size=400, overlap=50, title_text=title)
    else:
        print("PDF appears to be scanned — running OCR...")
        page_texts = ocr_extract_text(doc, dpi=300)
        page_texts = clean_ocr_pages(page_texts)
        text, chunks = chunk_plain_text(page_texts, pdf_path, chunk_size=400, overlap=70)
        ocr_used = True

    figure_count = count_figures(doc)

    doc.close()

  
    doc_name = re.sub(r"\s*\(\d+\)$", "", os.path.splitext(os.path.basename(pdf_path))[0]).strip()

    all_chunks = chunks + tables
    for chunk in all_chunks:
        chunk["doc_id"]   = doc_id
        chunk["doc_name"] = doc_name

   # generate summary
    summary = generate_summary(text, all_chunks, title or doc_name)
    print("\n--- DOCUMENT SUMMARY ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))  
    print("------------------------\n")

    # store document record 
    init_documents_table()
    store_document_record(
        doc_id, doc_name, title,
        authors if isinstance(authors, list) else [authors],
        page_count, word_count(text),
        len(all_chunks), len(tables), figure_count,
        summary
    )
    print("document record saved")

    print("Doc ID      :", doc_id)
    print("Doc Name    :", doc_name)
    print("Is Digital  :", is_digital)
    print("OCR Used    :", ocr_used)
    print("Page Count  :", page_count)
    print("Title       :", title)
    print("Authors     :", authors)
    print("Word Count  :", word_count(text))
    print("Chunk Count :", len(all_chunks))
    print("Figure Count:", figure_count)
    print("Table Count :", len(tables))
    print("Table BBoxes:", {k: len(v) for k, v in table_bboxes.items()})

    pdf_data = {
        "doc_id":       doc_id,
        "doc_name":     doc_name,
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

    store_chunks_db(all_chunks)
    print("saved to database")

    collection = get_chroma_collection()
    upsert_chunks_to_chroma(all_chunks, collection)
    print("chunks inserted to ChromaDB")
    return pdf_data

if __name__ == "__main__":
    main()
