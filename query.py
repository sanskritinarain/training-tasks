import argparse
import json
import logging
import sqlite3

from rag_agent import answer_question as generate_answer
from task_1 import llm
from conversation_store import (
    create_conversation,
    get_conversation,
    add_message,
    update_rolling_summary,
)

logger = logging.getLogger(__name__)


def _error(code: str, message: str):
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


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
        record["authors"] = (
            json.loads(record["authors"])
            if record["authors"]
            else []
        )
    except (TypeError, json.JSONDecodeError):
        record["authors"] = []

    try:
        record["summary"] = (
            json.loads(record["summary"])
            if record["summary"]
            else {}
        )
    except (TypeError, json.JSONDecodeError):
        record["summary"] = {}

    return record


def _resolve_conversation(conversation_id, doc_id, db_path="chunks.db"):
    if conversation_id:
        existing = get_conversation(conversation_id, db_path=db_path)

        if existing is not None and existing["doc_id"] == doc_id:
            return conversation_id

        if existing is not None and existing["doc_id"] != doc_id:
            logger.warning(
                "Conversation belongs to a different document; "
                "creating a new conversation",
                extra={
                    "conversation_id": conversation_id,
                    "doc_id": doc_id,
                },
            )

    return create_conversation(doc_id, db_path=db_path)


def handle_query(
    question,
    doc_id,
    n_results=3,
    conversation_id=None,
    db_path="chunks.db",
):
    try:
        record = get_document_record(doc_id, db_path=db_path)
    except Exception:
        logger.exception(
            "Failed to retrieve document",
            extra={"doc_id": doc_id},
        )
        return _error(
            "document_lookup_failed",
            "Unable to retrieve the document",
        )

    if record is None:
        return _error(
            "document_not_found",
            "Document not found",
        )

    try:
        conversation_id = _resolve_conversation(
            conversation_id,
            doc_id,
            db_path=db_path,
        )

        convo_state = get_conversation(
            conversation_id,
            db_path=db_path,
        )

        rolling_summary = (
            convo_state.get("rolling_summary")
            if convo_state
            else None
        )
        recent_messages = (
            convo_state.get("messages", [])
            if convo_state
            else []
        )
    except Exception:
        logger.exception(
            "Failed to load conversation",
            extra={
                "doc_id": doc_id,
                "conversation_id": conversation_id,
            },
        )
        return _error(
            "conversation_load_failed",
            "Unable to load the chat session",
        )

    question_lower = question.lower()
    answer_text = None
    sources = []
    source_type = "document"

    if any(
        word in question_lower
        for word in ["author", "wrote", "written", "writer"]
    ):
        authors = record.get("authors") or []
        answer_text = ", ".join(authors) if authors else "Unknown"

    elif any(word in question_lower for word in ["title", "name"]):
        answer_text = record.get("title") or "Unknown"

    else:
        try:
            result = generate_answer(
                question,
                doc_id,
                k=n_results,
                rolling_summary=rolling_summary,
                recent_messages=recent_messages,
            )

            answer_text = result["answer"]
            sources = result["sources"]
            source_type = result["source_type"]

        except Exception:
            logger.exception(
                "Failed to generate answer",
                extra={
                    "doc_id": doc_id,
                    "conversation_id": conversation_id,
                },
            )
            return _error(
                "answer_generation_failed",
                "Unable to process the chat request",
            )

    try:
        add_message(
            conversation_id,
            "user",
            question,
            sources=None,
            source_type="user",
            db_path=db_path,
        )
        add_message(
            conversation_id,
            "agent",
            answer_text,
            sources=sources,
            source_type=source_type,
            db_path=db_path,
        )
    except Exception:
        logger.exception(
            "Failed to save conversation messages",
            extra={
                "doc_id": doc_id,
                "conversation_id": conversation_id,
            },
        )
        return _error(
            "conversation_save_failed",
            "Unable to save the chat message",
        )

    try:
        update_rolling_summary(
            conversation_id,
            llm,
            doc_title=record.get("title", ""),
            db_path=db_path,
        )
    except Exception:
        logger.exception(
            "Failed to update rolling summary",
            extra={
                "doc_id": doc_id,
                "conversation_id": conversation_id,
            },
        )

    return {
        "ok": True,
        "conversation_id": conversation_id,
        "answer": answer_text,
        "sources": sources,
        "source_type": source_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Query a single ingested document by doc_id"
    )
    parser.add_argument(
        "doc_id",
        help="doc_id of the document to query",
    )
    parser.add_argument(
        "ques",
        help="Question to ask about the document",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Continue an existing conversation, or omit to start fresh",
    )
    args = parser.parse_args()

    outcome = handle_query(
        args.ques,
        args.doc_id,
        args.n_results,
        conversation_id=args.conversation_id,
    )

    if outcome["ok"]:
        print(f"(conversation_id: {outcome['conversation_id']})")
    else:
        print(f"Error: {outcome['error']['message']}")


if __name__ == "__main__":
    main()
