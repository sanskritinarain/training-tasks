"""Guardrail: verify all project .py files are syntactically valid before committing."""
import ast
import sys

FILES_TO_CHECK = [
    "rag_agent.py",
    "query.py",
    "task_1.py",
    "web_search.py",
    "conversation_store.py",
    "handoff.py",
]

def main():
    failed = []
    for fname in FILES_TO_CHECK:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=fname)
            print(f"OK    {fname}")
        except FileNotFoundError:
            print(f"SKIP  {fname} (not found)")
        except SyntaxError as e:
            print(f"FAIL  {fname}: {e}")
            failed.append(fname)

    if failed:
        print(f"\n{len(failed)} file(s) failed to parse: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll files parse cleanly.")
    sys.exit(0)

if __name__ == "__main__":
    main()