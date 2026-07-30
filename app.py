from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from task_1 import main as process_document
from conversation_store import create_conversation
from query import handle_query
import jwt
import sqlite3
import os
from token_utils import verify_chat_token, create_chat_token, decode_expired_token, TOKEN_EXPIRE_MINUTES, MAX_REFRESH_AGE_HOURS
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  
from fastapi.middleware.cors import CORSMiddleware
from task_1 import init_documents_table
from fastapi.concurrency import run_in_threadpool
import secrets
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

cors_origins = os.getenv("CORS_ORIGINS", "").split(",")

app.add_middleware(                                       
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_API_KEY = os.getenv("DOCUMENTS_API_KEY")

if not DOCUMENTS_API_KEY:
    raise RuntimeError("DOCUMENTS_API_KEY environment variable is required")


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or not secrets.compare_digest(x_api_key, DOCUMENTS_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

init_documents_table()

def ensure_status_column():
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(documents)")
    columns = [row[1] for row in cursor.fetchall()]
    if "status" not in columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'ready'")
        conn.commit()
    conn.close()

init_documents_table()
ensure_status_column()

def init_sessions_table():
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    doc_id TEXT
)
""")
    conn.commit()
    conn.close()



UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
init_sessions_table()


@app.get("/")
def health_check():
    return {"status": "API is running"}


# 1st API (NEW DOC UPLOAD)

import hashlib
from fastapi import BackgroundTasks


def run_extraction_and_mark_ready(file_path: str, doc_id: str):
    try:
        process_document(file_path)
    except Exception:
        conn = sqlite3.connect("chunks.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET status = 'failed' WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()
        return

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE documents SET status = 'ready' WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()


@app.post("/uploadDocument")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
):
    filename = os.path.basename(file.filename)

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    doc_id = hashlib.md5(contents).hexdigest()[:12]
    doc_name = os.path.splitext(filename)[0]

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO documents (doc_id, doc_name, status) VALUES (?, ?, 'processing')",
        (doc_id, doc_name),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(run_extraction_and_mark_ready, file_path, doc_id)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "status": "processing",
        "message": "Document uploaded — processing in background",
    }

# 2nd API (FETCH UPLOADED DOCS ID)

@app.get("/getAllUploadedDocuments")
def get_all_uploaded_documents(
    _: None = Depends(require_api_key),
):
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, doc_name FROM documents")
    rows = cursor.fetchall()
    conn.close()

    documents = [{"doc_id": row[0], "doc_name": row[1]} for row in rows]

    return {"documents": documents}


# 3RD API (GET DOC BY ID + CHAT TOKEN

@app.get("/document/{doc_id}")
def get_document(
    doc_id: str,
    _: None = Depends(require_api_key),
):
    document = get_document_record(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    token = create_chat_token(doc_id)
    payload = verify_chat_token(token)

    def to_gmt_string(unix_ts: int) -> str:
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    return {
        "doc_id": document["doc_id"],
        "doc_name": document["doc_name"],
        "chat_token": token,
        "created_at": to_gmt_string(payload["iat"]),
        "valid_till": to_gmt_string(payload["exp"]),
    }


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
   

    new_session_id = create_conversation(doc_id, db_path="chunks.db")

    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sessions (session_id, doc_id)
        VALUES (?, ?)
        """,
        (new_session_id, doc_id),
    )
    conn.commit()
    conn.close()

    return {"session_id": new_session_id, "status": "created"}

# 5TH API (SEND CHAT MESSAGE)
class ChatRequest(BaseModel):
    session_id: str
    query: str

def get_document_record(doc_id: str):
    conn = sqlite3.connect("chunks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, doc_name, status FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {"doc_id": row[0], "doc_name": row[1], "status": row[2]}

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

    # ---- NEW: status check ----
    document = get_document_record(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if document["status"] == "processing":
        return {"status": "processing", "message": "Document is still being processed. Please try again shortly."}

    if document["status"] == "failed":
        raise HTTPException(status_code=500, detail="Document processing failed. Please re-upload.")
    # ---- NEW block ends ----

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

    if not result["ok"]:
        error = result["error"]

        if error["code"] == "document_not_found":
            raise HTTPException(status_code=404, detail=error["message"])

        raise HTTPException(status_code=500, detail=error["message"])

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

    refresh_expires_at = payload.get("refresh_expires_at")

    if (
        not isinstance(refresh_expires_at, int)
        or datetime.now(timezone.utc).timestamp() >= refresh_expires_at
    ):
        raise HTTPException(
            status_code=401,
            detail="Token refresh period has expired. Request a new chat token.",
        )

    doc_id = payload["doc_id"]

    if get_document_record(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    new_token = create_chat_token(
        doc_id,
        refresh_expires_at=refresh_expires_at,
    )

    return {
        "doc_id": doc_id,
        "chat_token": new_token,
        "expires_in": TOKEN_EXPIRE_MINUTES * 60,
    }
