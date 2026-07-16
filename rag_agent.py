import re
import task_1
from web_search import web_search

# CONFIG
K_DEFAULT = 5
WEB_K_DEFAULT = 3
MIN_CONFIDENCE = 0.15
WEB_MIN_OVERLAP = 0.15
WEB_NOT_FOUND_MSG = "I couldn't find reliable information on the web."

PROMPT_TEMPLATE = """You are a research assistant. Answer the question using ONLY the sources below.

Rules:
- Answer concisely in 1-3 sentences using only the specific facts. Do not add background or restate the question.
- Give a direct, specific answer. Do NOT say "there is no specific answer" if relevant information exists in the sources — extract it. Do NOT summarize the passage.
- Give a longer answer only if the user explicitly asks for more detail.
- Cite the source number like [Source 2].
- If truly no relevant information exists, reply exactly: "NOT_IN_CONTEXT".
- Do not use outside knowledge.

Question: {question}

Sources:
{context}

Direct Answer:"""

WEB_PROMPT_TEMPLATE = """You are a research assistant. Answer the question using ONLY the web search results below.

Rules:
- Answer concisely in 1-3 sentences using only the specific facts. Do not add background or restate the question.
- Give a direct, specific answer. Do NOT say "there is no specific answer" if relevant information exists — extract it. Do NOT summarize the passage.
- Give a longer answer only if the user explicitly asks for more detail.
- Cite the source number like [Web 2].
- If the results disagree, note the disagreement briefly instead of picking one side silently.
- Never combine specific facts, numbers, steps, or details from different sources into a single unified answer unless one source actually states that exact combination. Each specific claim (a number, a date, a step, a measurement, a name) must come from one identifiable source. If different sources give different specifics for the same question, attribute each specific to its own source rather than blending them into one version.
- Answer ONLY using the information present in the web search snippets below. Do not use your own knowledge.
- If truly no relevant information exists in these results, reply exactly: "NOT_IN_CONTEXT".

Question: {question}

Web Results:
{context}

Direct Answer:"""

def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        loc = f"page {m['page_start']}" + (
            f"-{m['page_end']}" if m['page_end'] != m['page_start'] else ""
        )
        sect = f", {m['section_heading']}" if m.get("section_heading") else ""
        blocks.append(f"[Source {i} | {loc}{sect}]\n{h['text']}")
    return "\n\n".join(blocks)


def build_web_context(results):
   
    blocks = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Untitled"
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        blocks.append(f"[Web {i} | {url}]\n\nTitle:\n{title}\n\nSnippet:\n{snippet}")
    return "\n\n".join(blocks)


def _extract_cited_sources(answer_text, hits):
    cited_nums = sorted(set(int(n) for n in re.findall(r"\[Source (\d+)\]", answer_text)))
    sources = []
    for n in cited_nums:
        if 1 <= n <= len(hits):
            m = hits[n - 1]["meta"]
            sources.append({
                "page_start": m["page_start"],
                "page_end": m["page_end"],
                "section": m.get("section_heading"),
                "type": m.get("type"),
                "score": hits[n - 1]["score"],
            })
    return sources


def _extract_cited_web_sources(answer_text, results):
   
    cited_nums = sorted(set(int(n) for n in re.findall(r"\[Web (\d+)\]", answer_text)))
    sources = []
    for n in cited_nums:
        if 1 <= n <= len(results):
            r = results[n - 1]
            sources.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "type": "web",
            })
    return sources


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "on", "at", "of", "to", "for", "and", "or",
    "what", "when", "where", "who", "how",
    "which", "why",
    "does", "do", "did",
    "this", "that", "these", "those",
    "with", "as", "by", "it",
}


def _keyword_overlap(question, text):
   
    q_words = {w.strip(".,?!'\"") for w in question.lower().split()} - _STOPWORDS
    if not q_words:
        return 1.0  
    t_words = {w.strip(".,?!'\"") for w in text.lower().split()}
    hits = sum(1 for w in q_words if w in t_words)
    return hits / len(q_words)


def _filter_relevant_web_results(question, results, min_overlap=WEB_MIN_OVERLAP):

    kept = []
    for r in results:
        combined = f"{r.get('title', '')} {r.get('snippet', '')}"
        if _keyword_overlap(question, combined) >= min_overlap:
            kept.append(r)
    return kept

REWRITE_PROMPT_TEMPLATE = """Given the conversation summary, the most recent conversation turns,
and a follow-up question, rewrite the follow-up as a fully standalone question.

Rules:
- Resolve pronouns like "it", "they", "that model".
- Resolve references like "Kerala", "that paper", "the second method", using the recent turns.
- Preserve the user's original intent -- do not add new facts or assumptions.
- Do NOT answer the question.
- Return ONLY the rewritten standalone question, nothing else -- no preamble, no quotes.

Conversation summary:
{summary}

Recent conversation:
{recent_turns}

Follow-up question: {question}

Standalone question:"""


def _format_recent_turns(messages, max_turns=2):
    """Format the last max_turns (user, agent) pairs as plain text."""
    if not messages:
        return "(none)"
    recent = messages[-(max_turns * 2):]
    lines = []
    for m in recent:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines) if lines else "(none)"


def rewrite_standalone_question(question, rolling_summary, recent_messages=None):
    if not rolling_summary:
        return question

    recent_turns = _format_recent_turns(recent_messages or [])
    prompt = REWRITE_PROMPT_TEMPLATE.format(
        summary=rolling_summary,
        recent_turns=recent_turns,
        question=question,
    )
    try:
        rewritten = task_1.llm(prompt).strip()
    except Exception:
        return question

    rewritten = rewritten.strip('"').strip("'").strip()

    bad_prefixes = (
        "answer:",
        "standalone question:",
        "rewritten question:",
        "here is",
        "here's",
    )

    if (
        not rewritten
        or len(rewritten) < 3
        or rewritten.lower().startswith(bad_prefixes)
    ):
        return question

    return rewritten

def generate_web_answer(question, web_results):
 
    if not web_results:
        return {
            "answer": WEB_NOT_FOUND_MSG,
            "sources": [],
            "confidence": None,
            "grounded": False,
            "source_type": "none",
        }

    web_results = _filter_relevant_web_results(
        question,
        web_results,
        min_overlap=WEB_MIN_OVERLAP,
    )
    if not web_results:
        return {
            "answer": WEB_NOT_FOUND_MSG,
            "sources": [],
            "confidence": None,
            "grounded": False,
            "source_type": "none",
        }

    context = build_web_context(web_results)
    prompt = WEB_PROMPT_TEMPLATE.format(question=question, context=context)
    raw = task_1.llm(prompt).strip()
    if not raw or "NOT_IN_CONTEXT" in raw:
        return {
            "answer": WEB_NOT_FOUND_MSG,
            "sources": [],
            "confidence": None,
            "grounded": False,
            "source_type": "none",
        }

    sources = _extract_cited_web_sources(raw, web_results)
    if not sources:
        r = web_results[0]
        sources = [{"title": r.get("title"), "url": r.get("url"), "type": "web"}]

    return {
        "answer": f"[Web Search] {raw}",
        "sources": sources,
        "confidence": None,  
        "grounded": True,
        "source_type": "web",
    }


def _web_fallback(question):
 
    results = web_search(question, k=WEB_K_DEFAULT)
    return generate_web_answer(question, results)

def answer_question(question, doc_id, k=K_DEFAULT, rolling_summary=None, recent_messages=None):
    if rolling_summary:
        search_question = rewrite_standalone_question(question, rolling_summary, recent_messages)
    else:
        search_question = question

    print(f"Original : {question}")
    print(f"Searching: {search_question}")

    collection = task_1.get_chroma_collection()
    hits = task_1.query_chroma(search_question, collection, n_results=k, doc_id=doc_id)


    if search_question != question and (not hits or hits[0]["score"] < MIN_CONFIDENCE):
        retry_hits = task_1.query_chroma(question, collection, n_results=k, doc_id=doc_id)
        if retry_hits and (not hits or retry_hits[0]["score"] > hits[0]["score"]):
            hits = retry_hits
            search_question = question
            print(f"Fallback to original question (better retrieval): {search_question}")

    if not hits:
        return _web_fallback(search_question)

    confidence = hits[0]["score"]

    if hits[0]["score"] < MIN_CONFIDENCE:
        return _web_fallback(search_question)

    context = build_context(hits)
    prompt = PROMPT_TEMPLATE.format(question=search_question, context=context)
    raw = task_1.llm(prompt).strip()

    if not raw or "NOT_IN_CONTEXT" in raw:
        return _web_fallback(search_question)
    sources = _extract_cited_sources(raw, hits)

    if not sources:
        m = hits[0]["meta"]
        sources = [{
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "section": m.get("section_heading"),
            "type": m.get("type"),
            "score": hits[0]["score"],
        }]

    return {
        "answer": raw,
        "sources": sources,
        "confidence": confidence,
        "grounded": True,
        "source_type": "document",
    }
