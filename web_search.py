import os
import logging

logger = logging.getLogger(__name__)

SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "tavily").lower()
DEFAULT_K = 3


def _search_tavily(query: str, k: int) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — skipping web search.")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not installed (pip install tavily-python) — skipping web search.")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=k, search_depth="basic")
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []

    results = []
    for r in resp.get("results", [])[:k]:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("content") or "").strip()
        if not url or not snippet:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})

    return results


def web_search(query: str, k: int = DEFAULT_K) -> list[dict]:
    if not query or not query.strip():
        return []

    if SEARCH_PROVIDER == "tavily":
        return _search_tavily(query, k)

    logger.warning(f"Unknown SEARCH_PROVIDER '{SEARCH_PROVIDER}' — skipping web search.")
    return []
