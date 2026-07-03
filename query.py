import argparse
import json
import sqlite3
from rag_agent import answer_question as generate_answer  


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


def handle_query(question, doc_id, n_results=3):       

    record = get_document_record(doc_id)
    if record is None:
        print(f"No document found for doc_id={doc_id}.")
        return

    q = question.lower()

    if any(word in q for word in ["author", "wrote", "written", "writer"]):
        authors = record.get("authors") or []
        print("Authors:", ", ".join(authors) if authors else "Unknown")
        return

    if any(word in q for word in ["title", "name"]):
        print("Title:", record.get("title") or "Unknown")
        return

    
    result = generate_answer(question, doc_id, k=n_results)

    print("Answer:", result["answer"])
    print("Grounded:", result["grounded"])
    if result["confidence"] is not None:
        print("Confidence:", round(result["confidence"], 3))
    else:
        print("Confidence: N/A (web result)")

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

def main():
    parser = argparse.ArgumentParser(
        description="query a single ingested document by doc_id"
    )
    parser.add_argument("doc_id", help="doc_id of the document to query (from the documents table)")
    parser.add_argument("ques", help="ques to ask about the document")
    parser.add_argument("--n-results", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    handle_query(args.ques, args.doc_id, args.n_results)  


if __name__ == "__main__":
    main()
