import pymupdf
import pdfplumber
import json
import re

# SCAN OR DIGITAL
def if_digital(pdf_path):
    document = pymupdf.open(pdf_path)
    for page in document:
        if page.get_text().strip() == "":
            return False
    return True

# TITLE
def extract_title(pdf_path):
    document = pymupdf.open(pdf_path)
    title = document.metadata.get("title")
    if title:
        return title
    page= document[0]
    title=[]
    max_size= None
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            size = max(s["size"] for s in line["spans"])

            if not text:
                continue

            if max_size is None:
                max_size = size

            if max_size - size > 2 or text.endswith("."):
                return " ".join(title)

            title.append(text)

    return " ".join(title)


def extract_authors(pdf_path):
    doc = pymupdf.open(pdf_path)

    if doc.metadata.get("author"):
        return doc.metadata["author"].split(", ")

    lines = []

    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):

            text = "".join(span["text"] for span in line["spans"]).strip()

            if text:
                size = max(span["size"] for span in line["spans"])
                lines.append((text, size))

    title_size = max(size for _, size in lines)

    author_size = next(
        size for _, size in lines
        if size < title_size
    )

    authors = []

    for text, size in lines:
        if size == author_size and not any(
            word in text.lower()
            for word in ["department", "university", "email", "@"]
        ):
            authors.append(text)

    return authors

# TEXT
def extract_text(pdf_path, chunk_size=500):
    document = pymupdf.open(pdf_path)

    text = ""
    for page in document:
        text += page.get_text() + " "

    words = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return text, chunks

# WORD COUNT
def word_count(text):
    return len(text.split())

# TABLES 
def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)
    return tables

# TABLE COUNT
def count_tables(pdf_path):
    tables = extract_tables(pdf_path)
    return len(tables)

# FIGURE COUNT 
def count_figures(pdf_path):
    document = pymupdf.open(pdf_path)
    count = 0
    for page in document:
        count = count + len(page.get_images())
    return count


pdf_path = "multivariate_paper (7).pdf"

document    = pymupdf.open(pdf_path)
page_count  = len(document)
is_digital  = if_digital(pdf_path)
title       = extract_title(pdf_path)
authors     = extract_authors(pdf_path)

text, chunks = extract_text(pdf_path)

tables      = extract_tables(pdf_path)
table_count = count_tables(pdf_path)
figure_count = count_figures(pdf_path)

print("Is Digital :", is_digital)
print("Page Count :", page_count)
print("Title      :", title)
print("Authors    :", authors)
print("Word Count :", word_count(text))
print("Table Count:", len(tables))
print("Figure Count:", figure_count)


pdf_data = {
    "is_digital"   : is_digital,
    "page_count"   : page_count,
    "title"        : title,
    "authors"      : authors if isinstance(authors, list) else [authors],
    "word_count"   : word_count(text),
    "chunks"       : chunks,
    "table_count"  : len(tables),
    "figure_count" : figure_count,
    "text"         : text,
    "tables"       : tables,
}

with open("pdf_output.json", "w", encoding="utf-8") as f:
    json.dump(pdf_data, f, indent=4, ensure_ascii=False)

print(" save " + "pdf_output.json")
