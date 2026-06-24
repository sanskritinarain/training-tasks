from tsk1 import get_chroma_collection, query_chroma
import json

with open("pdf_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

col = get_chroma_collection()

question = "who wrote this paper?"

if any(word in question.lower() for word in ["author", "wrote", "written", "writer"]):
    print("Authors:", ", ".join(data["authors"]))

elif any(word in question.lower() for word in ["title",  "name"]):
    print("Title:", data["title"])

else:
    hits = query_chroma(question, col, n_results=3)
    for h in hits:
        print(h["text"][:300])
        print("---")
