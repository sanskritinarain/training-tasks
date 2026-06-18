import pymupdf
import pdfplumber
import json

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
    return document[0].get_text().split("\n")[0]

# AUTHORS
def extract_authors(pdf_path):
    document = pymupdf.open(pdf_path)
    authors = document.metadata.get("author")
    if authors:
        return authors.split(", ")
    return document[0].get_text().split("\n")[1]

# TEXT
def extract_text(pdf_path):
    document = pymupdf.open(pdf_path)
    text = ""
    for page in document:
        text = text + page.get_text()
    return text

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
text        = extract_text(pdf_path)
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
    "table_count"  : len(tables),
    "figure_count" : figure_count,
    "text"         : text,
    "tables"       : tables,
}

with open("pdf_output.json", "w", encoding="utf-8") as f:
    json.dump(pdf_data, f, indent=4, ensure_ascii=False)

print(" save " + "pdf_output.json")