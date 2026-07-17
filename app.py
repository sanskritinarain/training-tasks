from fastapi import FastAPI, UploadFile, File
import os
from task_1 import main as process_document
import sqlite3

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize db
conn = sqlite3.connect("chunks.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        doc_name TEXT
    )
""")
conn.commit()
conn.close()

@app.get("/")
def health_check():
    return {"status": "API is running"}

# FIRST API (NEW DOC UPLOAD
@app.post("/uploadDocument")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    result = process_document(file_path)

    if result is None:
        return {"error": "Failed to process document"}

    return {
        "doc_id": result["doc_id"],
        "filename": file.filename,
        "message": "Document uploaded and processed successfully"
    }


# SECOND API (FETCH ALL UPLOADED DOCS INFO)
@app.get("/getAllUploadedDocuments")
def get_all_uploaded_documents():
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, doc_name FROM documents")
    rows = cursor.fetchall()
    conn.close()

    documents = [{"doc_id": row[0], "doc_name": row[1]} for row in rows]

    return {"documents": documents}
