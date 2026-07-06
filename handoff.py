import argparse
import textwrap
from query import get_document_record
from conversation_store import get_conversation, get_latest_conversation_id

SEPARATOR = "-" * 60
ROLE_LABELS = {"user": "User", "agent": "Assistant"}


def _format_sources(sources, source_type=None):
    if not sources:
        return "  (no sources)"
    tag = f" ({source_type})" if source_type else ""
    lines = []
    for s in sources:
        if s.get("type") == "web":
            lines.append(f"  - {s.get('title') or 'Untitled'} ({s.get('url')}){tag}")
        else:
            loc = f"page {s.get('page_start')}"
            if s.get("page_end") and s.get("page_end") != s.get("page_start"):
                loc += f"-{s['page_end']}"
            sect = f", {s['section']}" if s.get("section") else ""
            lines.append(f"  - {loc}{sect}{tag}")
    return "\n".join(lines)


def build_handoff(doc_id=None, conversation_id=None, db_path="chunks.db"):
    if conversation_id:
        convo = get_conversation(conversation_id, db_path=db_path)
        if convo is None:
            print(f"No conversation found for conversation_id={conversation_id}.")
            return None
        doc_id = convo["doc_id"]
    else:
        if doc_id is None:
            print("Must provide either doc_id or conversation_id.")
            return None
        conversation_id = get_latest_conversation_id(doc_id, db_path=db_path)
        convo = get_conversation(conversation_id, db_path=db_path) if conversation_id else None

    record = get_document_record(doc_id, db_path=db_path)

    print("=" * 60)
    print("DOCUMENT")
    print("=" * 60)
    if record is None:
       
        print("Unavailable (no document record found for this doc_id).")
        if convo is None:
            print(f"No document found for doc_id={doc_id}, and no conversation to fall back on.")
            return None
    else:
        summary = record.get("summary") or {}
        print("Title:", record.get("title") or "Unknown")
        authors = record.get("authors") or []
        print("Authors:", ", ".join(authors) if authors else "Unknown")
        print("Overview:", summary.get("overview", "N/A"))
        key_points = summary.get("key_points") or []
        if key_points:
            print("Key points:")
            for kp in key_points:
                print(f"  - {kp}")
        topics = summary.get("topics") or []
        if topics:
            print("Topics:", ", ".join(topics))

    print()
    print("=" * 60)
    print("CONVERSATION SO FAR")
    print("=" * 60)

    if convo is None:
        print("No conversation history yet for this document.")
        return {"doc_id": doc_id, "conversation_id": None, "record": record, "convo": None}

    print("Conversation ID:", convo["conversation_id"])
    print("Rolling summary:", convo.get("rolling_summary") or "(none yet)")

    print()
    print("=" * 60)
    print("MESSAGE HISTORY")
    print("=" * 60)
    for i, m in enumerate(convo["messages"], start=1):
        role_label = ROLE_LABELS.get(m["role"], m["role"])
        print(f"[{i}] {role_label} ({m['timestamp']}):")
        wrapped = textwrap.fill(m["content"], width=90)
        print(wrapped)
        if m["role"] == "agent" and m.get("sources"):
            print(_format_sources(m["sources"], m.get("source_type")))
        if i < len(convo["messages"]):
            print(SEPARATOR)

    return {"doc_id": doc_id, "conversation_id": convo["conversation_id"], "record": record, "convo": convo}


def main():
    parser = argparse.ArgumentParser(description="Load a full handoff view for a document or conversation")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--db-path", default="chunks.db")
    args = parser.parse_args()
    build_handoff(doc_id=args.doc_id, conversation_id=args.conversation_id, db_path=args.db_path)


if __name__ == "__main__":
    main()