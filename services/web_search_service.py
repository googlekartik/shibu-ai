"""Web search: a thin wrapper over Chrome-based search.

The real work lives in services/chrome_search_service.py:
headless Chromium (Playwright, else Selenium) loading a search
results page, falling back to the LangSearch Web Search API when
LANGSEARCH_API_KEY is set, then to DuckDuckGo (no keys, no
browser) so search still returns results on hosts with neither.
This wrapper preserves the WebSearchService.search() /
max_results API and the {title, url, snippet, source} shape,
with source "Chrome", "LangSearch" or "DuckDuckGo".

search_provider_label() derives the user-visible label from the
results themselves ("Chrome search" / "LangSearch search" /
"DuckDuckGo search"), so the label can never name the wrong
provider. With no results it returns "Chrome search" (the
configured search path).

The LangSearch API key is read from the environment only, sent
only to LangSearch's API, and never logged, stored, or returned.
"""
from services.chrome_search_service import ChromeSearchService


def search_provider_label(results) -> str:
    """User-visible tool label for a result set, derived from the
    results themselves so the label can never name the wrong
    provider."""
    if results and isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("source") == "LangSearch":
                return "LangSearch search"
        for item in results:
            if isinstance(item, dict) and item.get("source") == "DuckDuckGo":
                return "DuckDuckGo search"
    return "Chrome search"


class WebSearchService:
    def __init__(self, max_results=5):
        self.max_results = max_results
        self._chrome = None

    @property
    def chrome_service(self):
        # Built lazily so env vars set after construction (tests,
        # per-request config) are honoured by the underlying service.
        if self._chrome is None or self._chrome.max_results != self.max_results:
            self._chrome = ChromeSearchService(max_results=self.max_results)
        return self._chrome

    def search(self, query: str):
        query = (query or "").strip()
        if not query:
            return []
        try:
            return self.chrome_service.search(query) or []
        except Exception:
            # A search failure must never fail the question.
            return []
