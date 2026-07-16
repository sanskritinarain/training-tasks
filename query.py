import argparse
import json
import sqlite3
from rag_agent import answer_question as generate_answer
from task_1 import llm
from conversation_store import (
    create_conversation,
    get_conversation,
    add_message,
    update_rolling_summary,
)


def get_document_record(doc_id, db_path="chunks.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    record = dict(row)
    try:
        record["authors"] = json.loads(record["authors"]) if record["authors"] else []
    except (TypeError, json.JSONDecodeError):
        record["authors"] = []

    try:
        record["summary"] = json.loads(record["summary"]) if record["summary"] else {}
    except (TypeError, json.JSONDecodeError):
        record["summary"] = {}

    return record


def _resolve_conversation(conversation_id, doc_id, db_path="chunks.db"):
 
    if conversation_id:
        existing = get_conversation(conversation_id, db_path=db_path)
        if existing is not None and existing["doc_id"] == doc_id:
            return conversation_id
        if existing is not None and existing["doc_id"] != doc_id:
            print(f"Note: conversation {conversation_id} belongs to a different document. Starting a new conversation.")
    return create_conversation(doc_id, db_path=db_path)


def handle_query(question, doc_id, n_results=3, conversation_id=None, db_path="chunks.db"):

    record = get_document_record(doc_id, db_path=db_path)
    if record is None:
        print(f"No document found for doc_id={doc_id}.")
        return None

    conversation_id = _resolve_conversation(conversation_id, doc_id, db_path=db_path)

    convo_state = get_conversation(conversation_id, db_path=db_path)
    rolling_summary = convo_state.get("rolling_summary") if convo_state else None
    recent_messages = convo_state.get("messages", []) if convo_state else []

    question_lower = question.lower()

    is_shortcut = False
    answer_text = None
    sources = []
    source_type = "document"

    if any(word in question_lower for word in ["author", "wrote", "written", "writer"]):
        is_shortcut = True
        authors = record.get("authors") or []
        answer_text = ", ".join(authors) if authors else "Unknown"
        print("Authors:", answer_text)

    elif any(word in question_lower for word in ["title", "name"]):
        is_shortcut = True
        answer_text = record.get("title") or "Unknown"
        print("Title:", answer_text)

    else:
        try:
            result = generate_answer(
                question, doc_id, k=n_results,
                rolling_summary=rolling_summary,
                recent_messages=recent_messages,
            )
            answer_text = result["answer"]
            sources = result["sources"]
            source_type = result["source_type"]

            print("Answer:", result["answer"])
            print("Grounded:", result["grounded"])
            if result["confidence"] is not None:
                print("Confidence:", round(result["confidence"], 3))
            else:
                print("Confidence: N/A")

            if result["sources"]:
                print("Sources:")
                for s in result["sources"]:
                    if s.get("type") == "web":
                        print(f"  - {s.get('title') or 'Untitled'} ({s.get('url')})")
                    else:
                        loc = f"page {s['page_start']}" + (
                            f"-{s['page_end']}" if s['page_end'] != s['page_start'] else ""
                        )
                        sect = f", {s['section']}" if s.get("section") else ""
                        score = s.get("score")
                        score_str = f", score={score:.2f}" if score is not None else ""
                        print(f"  - {loc}{sect} ({s['type']}{score_str})")
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

    try:
        add_message(conversation_id, "user", question, sources=None, source_type="user", db_path=db_path)
        add_message(conversation_id, "agent", answer_text, sources=sources, source_type=source_type, db_path=db_path)
    except Exception as e:
        print(f"Warning: failed to save conversation: {e}")

    try:
        update_rolling_summary(conversation_id, llm, doc_title=record.get("title", ""), db_path=db_path)
    except Exception as e:
        print(f"Warning: failed to update rolling summary: {e}")

    return {
        "conversation_id": conversation_id,
        "answer": answer_text,
        "sources": sources,
        "source_type": source_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="query a single ingested document by doc_id"
    )
    parser.add_argument("doc_id", help="doc_id of the document to query (from the documents table)")
    parser.add_argument("ques", help="ques to ask about the document")
    parser.add_argument("--n-results", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--conversation-id", default=None, help="Continue an existing conversation, or omit to start fresh")
    args = parser.parse_args()

    outcome = handle_query(args.ques, args.doc_id, args.n_results, conversation_id=args.conversation_id)

    if outcome:
        print(f"(conversation_id: {outcome['conversation_id']})")


if __name__ == "__main__":
    main()
