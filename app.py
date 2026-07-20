from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from task_1 import main as process_document
from conversation_store import create_conversation
from token_utils import verify_chat_token, create_chat_token
from query import handle_query
import jwt
import sqlite3
import os
from token_utils import verify_chat_token, create_chat_token, decode_expired_token
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


app = FastAPI()


def init_sessions_table():
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    doc_id TEXT,
    last_active_at TEXT
)
""")
    conn.commit()
    conn.close()


init_sessions_table()

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


# 1st API (NEW DOC UPLOAD)
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


# 2nd API (FETCH UPLOADED DOCS ID)
@app.get("/getAllUploadedDocuments")
def get_all_uploaded_documents():
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, doc_name FROM documents")
    rows = cursor.fetchall()
    conn.close()

    documents = [{"doc_id": row[0], "doc_name": row[1]} for row in rows]

    return {"documents": documents}


# 3RD API (GET DOC BY ID + CHAT TOKEN)
def get_document_record(doc_id: str):
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, doc_name FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {"doc_id": row[0], "doc_name": row[1]}


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    document = get_document_record(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    token = create_chat_token(doc_id)

    return {
        "doc_id": document["doc_id"],
        "doc_name": document["doc_name"],
        "chat_token": token,
        "expires_in": 3600
    }


SESSION_TIMEOUT_MINUTES = 30


# 4TH API (START OR CONTINUE CHAT SESSION)
security = HTTPBearer()

@app.post("/initiateChat")
def initiate_chat(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials  

    try:
        payload = verify_chat_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    doc_id = payload["doc_id"]

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT session_id, last_active_at FROM sessions WHERE doc_id = ? ORDER BY last_active_at DESC LIMIT 1",
        (doc_id,)
    )
    row = cursor.fetchone()

    now = datetime.utcnow()

    if row is not None:
        existing_session_id, last_active_str = row
        last_active = datetime.fromisoformat(last_active_str)

        if now - last_active < timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            cursor.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                (now.isoformat(), existing_session_id)
            )
            conn.commit()
            conn.close()
            return {"session_id": existing_session_id, "status": "continued"}

    conn.close()

    new_session_id = create_conversation(doc_id, db_path="chunks.db")

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (session_id, doc_id, last_active_at) VALUES (?, ?, ?)",
        (new_session_id, doc_id, now.isoformat())
    )
    conn.commit()
    conn.close()

    return {"session_id": new_session_id, "status": "created"}


# 5TH API (SEND CHAT MESSAGE)
class ChatRequest(BaseModel):
    session_id: str
    query: str


@app.post("/sendChat")
def send_chat(request: ChatRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials  

    try:
        payload = verify_chat_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    doc_id = payload["doc_id"]

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM sessions WHERE session_id = ?", (request.session_id,))
    session_row = cursor.fetchone()
    conn.close()

    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_row[0] != doc_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this document")

    result = handle_query(
        question=request.query,
        doc_id=doc_id,
        conversation_id=request.session_id,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Document or conversation not found")

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
        (datetime.utcnow().isoformat(), request.session_id)
    )
    conn.commit()
    conn.close()

    return {
        "session_id": result["conversation_id"],
        "answer": result["answer"],
        "sources": result["sources"],
        "source_type": result["source_type"],
    }


# 6TH API (REFRESH EXPIRED TOKEN)
@app.post("/refreshToken")
def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    old_token = credentials.credentials

    try:
        payload = decode_expired_token(old_token)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    doc_id = payload["doc_id"]

    document = get_document_record(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    new_token = create_chat_token(doc_id)

    return {
        "doc_id": doc_id,
        "chat_token": new_token,
        "expires_in": 3600
    }
