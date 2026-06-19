import logging

from anthropic.types import ToolParam
from ddgs import DDGS

logger = logging.getLogger(__name__)

SEARCH_TOOL_DEFINITION: ToolParam = {
    "name": "search",
    "description": (
        "Search the web for up-to-date information using DuckDuckGo. "
        "Use this tool for any factual claim that requires current or external data. "
        "Returns a list of results with title, url, and snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return normalised results.

    Each result dict has keys: title, url, snippet.
    Returns an empty list if no results are found.
    """
    logger.info("ddgs: querying %r", query)
    raw = DDGS().text(query, max_results=max_results)
    results = [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in (raw or [])]
    logger.info("ddgs: got %d results for %r", len(results), query)
    return results
