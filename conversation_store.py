import os
import sqlite3
import json
import hashlib
import time
import tempfile

def _connect(db_path="chunks.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_conversation_tables(db_path="chunks.db"):
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            doc_id TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            rolling_summary TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            source_type TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(conversation_id)
                REFERENCES conversations(conversation_id)
                ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(doc_id, db_path="chunks.db"):
    init_conversation_tables(db_path)
    conn = _connect(db_path)
    cursor = conn.cursor()
    conversation_id = None
    for _ in range(5):
        raw = f"{doc_id}_{time.time_ns()}"
        candidate_id = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        cursor.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (candidate_id,)
        )
        if cursor.fetchone() is None:
            conversation_id = candidate_id
            break

    if conversation_id is None:
        conn.close()
        raise RuntimeError("Could not create a unique conversation id")

    cursor.execute(
        "INSERT INTO conversations (conversation_id, doc_id, rolling_summary) VALUES (?, ?, ?)",
        (conversation_id, doc_id, None)
    )
    conn.commit()
    conn.close()
    return conversation_id


def add_message(conversation_id, role, content, sources=None, source_type=None, db_path="chunks.db"):
    conn = _connect(db_path)
    cursor = conn.cursor()
    sources_json = json.dumps(sources) if sources is not None else json.dumps([])
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, source_type) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, sources_json, source_type)
    )
    conn.commit()
    conn.close()


def get_conversation(conversation_id, db_path="chunks.db"):
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,))
    convo_row = cursor.fetchone()
    if convo_row is None:
        conn.close()
        return None

    convo = dict(convo_row)

    cursor.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,)
    )
    message_rows = cursor.fetchall()
    conn.close()

    messages = []
    for row in message_rows:
        m = dict(row)
        try:
            m["sources"] = json.loads(m["sources"]) if m["sources"] else []
        except (TypeError, json.JSONDecodeError):
            m["sources"] = []
        messages.append(m)

    convo["messages"] = messages
    return convo


def get_latest_conversation_id(doc_id, db_path="chunks.db"):
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT conversation_id FROM conversations WHERE doc_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (doc_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["conversation_id"] if row else None


def summarize_turn(ollama_fn, previous_summary, question, answer, doc_title=""):
    prev = previous_summary or "(no prior discussion yet)"
    prompt = f"""You are maintaining a short rolling summary of an ongoing Q&A conversation
about the document "{doc_title}".

Previous summary:
{prev}

Latest turn:
User asked: {question}
Agent answered: {answer}

Update the summary to reflect the conversation so far: what was asked, what was
answered, and any open threads. Keep it short (3-5 sentences), factual, and
cumulative -- don't drop earlier context unless it's fully resolved.
Return only the updated summary text, nothing else."""

    updated = ollama_fn(prompt)
    return updated.strip() if isinstance(updated, str) else str(updated)


def update_rolling_summary(conversation_id, ollama_fn, doc_title="", db_path="chunks.db"):
    convo = get_conversation(conversation_id, db_path=db_path)
    if convo is None:
        return None

    messages = convo["messages"]
    if len(messages) < 2:
        return convo.get("rolling_summary")

   
    last_user = messages[-2]
    last_agent = messages[-1]
    if not (last_user["role"] == "user" and last_agent["role"] == "agent"):
        # Fallback: irregular ordering, scan backwards for the most recent pair.
        last_user, last_agent = None, None
        for m in reversed(messages):
            if last_agent is None and m["role"] == "agent":
                last_agent = m
                continue
            if last_agent is not None and last_user is None and m["role"] == "user":
                last_user = m
                break
        if last_user is None or last_agent is None:
            return convo.get("rolling_summary")

    previous_summary = convo.get("rolling_summary")
    updated_summary = summarize_turn(
        ollama_fn,
        previous_summary,
        last_user["content"],
        last_agent["content"],
        doc_title=doc_title,
    )

    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE conversations SET rolling_summary = ? WHERE conversation_id = ?",
        (updated_summary, conversation_id)
    )
    conn.commit()
    conn.close()

    return updated_summary


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = f.name

    try:
        cid = create_conversation("f62eee6c88a0", db_path=test_db)
        add_message(cid, "user", "What is the RMSE for Kerala?", db_path=test_db)
        add_message(
            cid,
            "agent",
            "Kerala's RMSE is 4.2",
            sources=[{"type": "document", "page_start": 5}],
            source_type="document",
            db_path=test_db,
        )

        print(get_conversation(cid, db_path=test_db))
    finally:
        os.remove(test_db)
