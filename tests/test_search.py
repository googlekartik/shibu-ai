import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch
from services.web_search_service import WebSearchService


def test_search_processing_with_mock():
    svc = WebSearchService()
    fake = [{"title": "AI News", "url": "https://example.com", "snippet": "AI...", "source": "DuckDuckGo"}]
    with patch.object(WebSearchService, "search", return_value=fake):
        results = svc.search("latest AI news")
        assert results[0]["title"] == "AI News"
        assert results[0]["url"].startswith("https://")
