import argparse

import rag_agent

rag_agent.MIN_CONFIDENCE = 0.05

from query import handle_query
from conversation_store import get_conversation


TURNS = [
    "What is the objective of this paper?",
    "What model did they use?",
    "What about Kerala?",
    "What was its MAE?",
]


def run_conversation(doc_id):
    conversation_id = None

    print("=" * 70)
    print("Starting multi-turn conversation")
    print("=" * 70)

    for i, question in enumerate(TURNS, 1):
        print(f"\n--- Turn {i}: {question} ---")

        outcome = handle_query(
            question,
            doc_id,
            conversation_id=conversation_id,
        )

        if not outcome["ok"]:
            error = outcome["error"]
            print(
                f"Turn {i} FAILED: "
                f"{error['code']} - {error['message']}"
            )
            return None

        conversation_id = outcome["conversation_id"]

    return conversation_id


def print_handoff(conversation_id):
    print("\n" + "=" * 70)
    print("Handoff (fresh session)")
    print("=" * 70)

    convo = get_conversation(conversation_id)

    if convo is None:
        print("No conversation found - something is wrong.")
        return

    print(f"\nconversation_id: {convo['conversation_id']}")
    print(f"doc_id: {convo['doc_id']}")
    print(f"created_at: {convo['created_at']}")

    print("\n--- Rolling Summary ---")
    print(convo["rolling_summary"] or "(none)")

    print(f"\n--- Message History ({len(convo['messages'])} messages) ---")
    for message in convo["messages"]:
        role = message["role"].upper()
        content = message["content"]
        print(f"[{role}] {content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a multi-turn document-chat test"
    )
    parser.add_argument(
        "doc_id",
        help="Document ID to use for the test",
    )
    args = parser.parse_args()

    cid = run_conversation(args.doc_id)

    if cid:
        print_handoff(cid)
        print(f"\n\nconversation_id for manual re-check: {cid}")
