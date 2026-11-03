"""LangSearch web search (2026-11-03): LangSearch replaces the
Google Programmable Search JSON API as the non-browser fallback.
Headless Chromium (Playwright/Selenium, optional) loads a search
results page when installed; when no automation library is
installed, the LangSearch Web Search API is used when
LANGSEARCH_API_KEY is set, else DuckDuckGo, else [] — never any
other search engine. All HTTP is mocked and both automation
libraries are absent in this environment, so the browser path is
exercised through the pure HTML parser and the
disabled/no-automation branches. No live calls, no keys leave
the process.
"""
import json
import urllib.error

import pytest

from services.chrome_search_service import (
    ChromeSearchService, parse_google_results_page, _is_blocked_page)
from services.web_search_service import WebSearchService, search_provider_label

FAKE_KEY = "SECRETKEY-12345"


def _langsearch_payload(count=12):
    return {"data": {"webPages": {"value": [
        {"name": f"Result {i}", "url": f"https://example.com/{i}",
         "snippet": f"snippet {i}"} for i in range(count)]}}}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _langsearch_keys(monkeypatch):
    monkeypatch.setenv("LANGSEARCH_API_KEY", FAKE_KEY)
    monkeypatch.delenv("LANGSEARCH_API_BASE", raising=False)
    monkeypatch.setenv("BROWSER_SEARCH_ENABLED", "true")


def _no_langsearch_key(monkeypatch):
    monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSEARCH_API_BASE", raising=False)
    monkeypatch.setenv("BROWSER_SEARCH_ENABLED", "true")


def _no_browser_libs(monkeypatch):
    """Force the no-automation branch: neither Playwright nor
    Selenium is importable in this test environment anyway; the
    LangSearch API path is what remains."""
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)


# ------------------------------------------------------------- LangSearch API
def test_langsearch_success_parsing_and_shape(monkeypatch):
    _langsearch_keys(monkeypatch)
    _no_browser_libs(monkeypatch)
    seen = {}

    def fake_urlopen(req, timeout=10):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["auth"] = req.get_header("Authorization")
        assert req.full_url == "https://api.langsearch.com/v1/web-search"
        return _FakeResponse(_langsearch_payload())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    results = WebSearchService(max_results=5).search("tea")
    assert len(results) == 5  # capped at max_results, not 12
    assert results[0] == {"title": "Result 0", "url": "https://example.com/0",
                          "snippet": "snippet 0", "source": "LangSearch"}
    assert search_provider_label(results) == "LangSearch search"
    assert seen["body"]["query"] == "tea"
    assert seen["body"]["count"] == 5
    assert seen["body"]["freshness"] == "noLimit"
    assert seen["auth"] == "Bearer " + FAKE_KEY
    assert FAKE_KEY not in json.dumps(results)


def test_langsearch_failure_returns_empty_never_another_engine(monkeypatch):
    _langsearch_keys(monkeypatch)

    def fake_urlopen(req, timeout=10):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("tea") == []
    assert search_provider_label([]) == "Chrome search"


def test_langsearch_empty_value_returns_empty(monkeypatch):
    _langsearch_keys(monkeypatch)

    def fake_urlopen(req, timeout=10):
        return _FakeResponse({"data": {"webPages": {"value": []}}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("tea") == []


def test_langsearch_malformed_response_returns_empty(monkeypatch):
    _langsearch_keys(monkeypatch)

    def fake_urlopen(req, timeout=10):
        return _FakeResponse({"unexpected": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    svc = ChromeSearchService()
    assert svc.search("tea") == []
    # LangSearch finds nothing, so the DuckDuckGo fallback runs
    # against the same mocked page and sets the final reason.
    assert svc.last_reason in ("langsearch_no_results",
                               "duckduckgo_no_results")


def test_langsearch_text_field_and_bad_urls(monkeypatch):
    _langsearch_keys(monkeypatch)

    def fake_urlopen(req, timeout=10):
        return _FakeResponse({"data": {"webPages": {"value": [
            {"name": "Full text", "url": "https://example.com/full",
             "snippet": "", "text": "full text here"},
            {"name": "No URL", "url": "", "snippet": "skip me"},
            {"name": "Relative", "url": "/relative/path",
             "snippet": "skip me too"},
        ]}}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    results = WebSearchService().search("tea")
    assert results == [{"title": "Full text", "url": "https://example.com/full",
                        "snippet": "full text here", "source": "LangSearch"}]


def test_langsearch_base_override_is_used(monkeypatch):
    _langsearch_keys(monkeypatch)
    monkeypatch.setenv("LANGSEARCH_API_BASE", "https://example.test/api/")
    seen = {}

    def fake_urlopen(req, timeout=10):
        seen["url"] = req.full_url
        return _FakeResponse(_langsearch_payload(1))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("tea")
    assert seen["url"] == "https://example.test/api/web-search"


def test_keys_absent_returns_empty_without_langsearch_call(monkeypatch):
    _no_langsearch_key(monkeypatch)

    def fake_urlopen(req, timeout=10):
        raise AssertionError("LangSearch must not be called without a key: "
                             + req.full_url)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("tea") == []


def test_google_vars_alone_do_not_enable_search(monkeypatch):
    """The removed Google Programmable Search vars must not trigger
    any keyed search: with no LANGSEARCH_API_KEY, only the
    no-keys DuckDuckGo path may run."""
    _no_langsearch_key(monkeypatch)
    monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "GOOGLEKEY-999")
    monkeypatch.setenv("GOOGLE_SEARCH_CX", "GOOGLECX-999")

    def fake_urlopen(req, timeout=10):
        assert "langsearch" not in req.full_url
        assert "googleapis" not in req.full_url
        raise urllib.error.URLError("no network in tests")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("tea") == []


def test_langsearch_exception_never_fails_the_question(monkeypatch):
    _langsearch_keys(monkeypatch)

    def fake_urlopen(req, timeout=10):
        raise RuntimeError("network down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("tea") == []


def test_empty_query_returns_empty_without_http(monkeypatch):
    _langsearch_keys(monkeypatch)

    def fake_urlopen(req, timeout=10):
        raise AssertionError("no HTTP for an empty query")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert WebSearchService().search("") == []
    assert WebSearchService().search("   ") == []


def test_duckduckgo_fallback_when_no_automation_and_no_keys(monkeypatch):
    """No browser automation, no LangSearch key: the DuckDuckGo
    fallback (mocked here) still returns results, labelled
    DuckDuckGo — search must not return nothing."""
    _no_langsearch_key(monkeypatch)
    fixture = [{"title": "DDG tea", "url": "https://example.com/ddg",
                "snippet": "ddg snippet", "source": "DuckDuckGo"}]
    monkeypatch.setattr(ChromeSearchService, "_duckduckgo_search",
                        lambda self, q: list(fixture))
    svc = ChromeSearchService()
    results = svc.search("tea")
    assert results == fixture
    assert search_provider_label(results) == "DuckDuckGo search"
    assert svc.last_reason == ""


def test_duckduckgo_fallback_empty_reports_reason(monkeypatch):
    _no_langsearch_key(monkeypatch)
    monkeypatch.setattr(ChromeSearchService, "_duckduckgo_search",
                        lambda self, q: [])
    svc = ChromeSearchService()
    assert svc.search("tea") == []
    # The mocked fallback sets no reason of its own, so the last
    # real layer's reason (LangSearch unconfigured) stands.
    assert svc.last_reason in ("langsearch_unconfigured",
                               "duckduckgo_no_results")


def test_duckduckgo_empty_page_sets_reason(monkeypatch):
    import sys
    _no_langsearch_key(monkeypatch)
    monkeypatch.setitem(sys.modules, "ddgs", None)  # force HTML path

    def fake_urlopen(req, timeout=10):
        return _FakeResponse({})  # no result anchors in the page

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    svc = ChromeSearchService()
    assert svc._duckduckgo_search("tea") == []
    assert svc.last_reason == "duckduckgo_no_results"


def test_langsearch_results_win_over_duckduckgo(monkeypatch):
    _langsearch_keys(monkeypatch)
    called = []

    def fake_urlopen(req, timeout=10):
        return _FakeResponse(_langsearch_payload(2))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(ChromeSearchService, "_duckduckgo_search",
                        lambda self, q: called.append(q) or [])
    results = ChromeSearchService().search("tea")
    assert results and results[0]["source"] == "LangSearch"
    assert called == []  # DuckDuckGo never runs when LangSearch answers


# ------------------------------------------------------- Chrome configuration
def test_browser_disabled_returns_empty_without_http(monkeypatch):
    _langsearch_keys(monkeypatch)
    monkeypatch.setenv("BROWSER_SEARCH_ENABLED", "false")

    def fake_urlopen(req, timeout=10):
        raise AssertionError("no HTTP when browser search is disabled")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    svc = ChromeSearchService()
    assert svc.search("tea") == []
    assert svc.last_reason == "browser_search_disabled"


def test_no_automation_and_no_keys_reports_reason(monkeypatch):
    _no_langsearch_key(monkeypatch)
    monkeypatch.setattr(ChromeSearchService, "_duckduckgo_search",
                        lambda self, q: [])
    svc = ChromeSearchService()
    assert svc.search("tea") == []
    assert svc.last_reason in ("langsearch_unconfigured",
                               "duckduckgo_no_results")


def test_search_timeout_and_max_results_are_bounded(monkeypatch):
    monkeypatch.setenv("SEARCH_TIMEOUT_SECONDS", "9999")
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "999")
    svc = ChromeSearchService()
    assert svc.timeout == 60
    assert svc.max_results == 10
    monkeypatch.setenv("SEARCH_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("MAX_SEARCH_RESULTS", "0")
    svc = ChromeSearchService()
    assert svc.timeout == 5
    assert svc.max_results == 1


# ------------------------------------------------------- HTML parser (no browser)
_GOOGLE_PAGE = """
<html><body>
<a href="/url?q=https://example.com/tea&amp;sa=U&amp;ved=1"><h3>How to make tea</h3></a>
<div><span>Boil water, add tea leaves, steep for three minutes and serve hot today.</span></div>
<a href="/url?q=https://example.com/coffee&amp;sa=U"><h3>Coffee guide</h3></a>
<div><span>Coffee beans are roasted and ground before brewing with hot water.</span></div>
</body></html>
"""


def test_parse_google_results_page_extracts_title_url_snippet():
    results = parse_google_results_page(_GOOGLE_PAGE, 5)
    assert len(results) == 2
    assert results[0]["title"] == "How to make tea"
    assert results[0]["url"] == "https://example.com/tea"
    assert "Boil water" in results[0]["snippet"]
    assert results[0]["source"] == "Chrome"
    assert search_provider_label(results) == "Chrome search"


def test_parse_google_results_page_respects_max_results():
    assert len(parse_google_results_page(_GOOGLE_PAGE, 1)) == 1


def test_parse_google_results_page_blocked_page_returns_empty():
    assert parse_google_results_page(
        "Our systems have detected unusual traffic. Please complete the CAPTCHA.", 5) == []
    assert _is_blocked_page("Please verify you are human, captcha required")


def test_parse_google_results_page_empty_returns_empty():
    assert parse_google_results_page("", 5) == []
    assert parse_google_results_page("<html></html>", 5) == []
