import rag_agent
rag_agent.MIN_CONFIDENCE = 0.05  

from query import handle_query
from conversation_store import get_conversation

DOC_ID = "f62eee6c88a0"

TURNS = [
    "What is the objective of this paper?",
    "What model did they use?",
    "What about Kerala?",
    "What was its MAE?",
]


def run_conversation():
    conversation_id = None
    print("=" * 70)
    print("starting multi turn conversation")
    print("=" * 70)

    for i, question in enumerate(TURNS, 1):
        print(f"\n--- Turn {i}: {question} ---")
        outcome = handle_query(question, DOC_ID, conversation_id=conversation_id)
        if outcome is None:
            print(f"Turn {i} FAILED — stopping test.")
            return None
        conversation_id = outcome["conversation_id"]

    return conversation_id


def print_handoff(conversation_id):
    print("\n" + "=" * 70)
    print("handoff (fresh session)")
    print("=" * 70)

    convo = get_conversation(conversation_id)
    if convo is None:
        print("No conversation found — something's wrong.")
        return

    print(f"\nconversation_id: {convo['conversation_id']}")
    print(f"doc_id: {convo['doc_id']}")
    print(f"created_at: {convo['created_at']}")

    print(f"\n--- Rolling Summary ---")
    print(convo["rolling_summary"] or "(none)")

    print(f"\n--- Message History ({len(convo['messages'])} messages) ---")
    for m in convo["messages"]:
        role = m["role"].upper()
        content = m["content"]
        print(f"[{role}] {content}")


if __name__ == "__main__":
    cid = run_conversation()
    if cid:
        print_handoff(cid)
        print(f"\n\nconversation_id for manual re-check: {cid}")