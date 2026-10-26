import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.router import _is_search_request, _phrase_search_results


def test_search_detection_broadened():
    assert _is_search_request("search latest ai news")
    assert _is_search_request("browse latest technology news")
    assert _is_search_request("what happened today in technology")
    assert _is_search_request("latest ai developments")
    assert not _is_search_request("what is python")


def test_search_phrasing_includes_summary_and_sources():
    results = [
        {"title": "AI News Today", "url": "https://example.com/ai", "snippet": "AI models are getting faster", "source": "DuckDuckGo"},
        {"title": "Tech Update", "url": "https://example.com/tech", "snippet": "New tools launched this week", "source": "DuckDuckGo"},
    ]
    text = _phrase_search_results("latest AI news", results)
    assert "Here's what I found on the web" in text
    assert "AI models are getting faster" in text
    assert "https://example.com/ai" in text
    assert "not from my rule database" in text
