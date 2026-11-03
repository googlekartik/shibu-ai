"""Chrome-based web search for Shibu AI (headless Chromium).

Configuration (environment only, never secrets in code):
  BROWSER_SEARCH_ENABLED=true   (default "true"; "false" disables the
                                browser path entirely — no browser
                                is launched)
  CHROME_EXECUTABLE_PATH=       (optional path to a Chromium/Chrome
                                binary; when unset, the automation
                                library's bundled Chromium is used)
  SEARCH_TIMEOUT_SECONDS=20     (bounded 5-60)
  MAX_SEARCH_RESULTS=5          (bounded 1-10)

Order for every query:
  (a) disabled -> no browser, no results from this layer;
  (b) headless Chromium via Playwright if installed, else Selenium
      if installed, loading a Google search results page and parsing
      title / URL / snippet only;
  (c) Google Programmable Search JSON API, only when both
      GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX are set;
  (d) DuckDuckGo (ddgs library, else its HTML endpoint) — the
      no-keys fallback, so search still returns results on hosts
      with no browser automation and no Google API keys;
  (e) [] with a clear reason (see `last_reason`).

Safety rules, enforced here and nowhere else needed:
- Headless only, with --no-sandbox; no cookies, passwords, history
  or any other browsing data is ever stored (a fresh context per
  search, closed afterwards; the context/browser is always closed
  in `finally`).
- No page JavaScript is executed by this code beyond loading the
  results page; no login, no form filling, no clicks.
- CAPTCHA / block / consent-wall text in the page means the search
  is blocked: return [] (never bypass a CAPTCHA, authentication,
  paywall, or website security control).
- Playwright and Selenium are OPTIONAL imports. When neither is
  installed this module still imports and works (Google API / []
  path), so the Render free build — which does not install them —
  is unaffected. Opt-in install: requirements-chrome.txt plus
  `playwright install chromium` in the build step (see
  docs/GEMINI_CHROME_UPGRADE.md). Without that install, headless
  Chromium does NOT run on Render; the Google API / empty path
  applies instead. Never claim otherwise.
- Only title, URL and snippet are kept. No page body is fetched.
- The DuckDuckGo fallback is a plain HTTP fetch (ddgs library /
  HTML endpoint), no browser, no stored data of any kind.
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import urllib.parse
import urllib.request

_BLOCK_MARKERS = (
    "unusual traffic", "captcha", "recaptcha", "i'm not a robot",
    "sorry, google", "our systems have detected", "verify you are human",
    "consent", "before you continue",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _bounded_int(name: str, default: int, lower: int, upper: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(value, upper))


def browser_search_enabled() -> bool:
    return _env_bool("BROWSER_SEARCH_ENABLED", True)


def search_timeout_seconds() -> int:
    return _bounded_int("SEARCH_TIMEOUT_SECONDS", 20, 5, 60)


def max_search_results() -> int:
    return _bounded_int("MAX_SEARCH_RESULTS", 5, 1, 10)


def _clean(text: str) -> str:
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _is_blocked_page(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _BLOCK_MARKERS)


def parse_google_results_page(page_html: str, max_results: int):
    """Parse a Google results page (HTML) into {title, url, snippet}.

    Pure function, used by both automation backends and unit-tested
    without a browser. Parses only result anchors (h3 inside an
    <a href="/url?q=...">) and the nearest following snippet text;
    anything that looks like a block/consent page yields [].
    """
    if not page_html or _is_blocked_page(page_html):
        return []
    results = []
    seen = set()
    for match in re.finditer(
            r'<a[^>]+href="/url\?q=([^"&]+)[^"]*"[^>]*>\s*<h3[^>]*>(.*?)</h3>',
            page_html, flags=re.S | re.I):
        raw_url = urllib.parse.unquote(match.group(1))
        if not raw_url.startswith("http") or raw_url in seen:
            continue
        seen.add(raw_url)
        title = _clean(match.group(2))
        window = page_html[match.end():match.end() + 1200]
        snippet = ""
        for span in re.finditer(r"<span[^>]*>(.*?)</span>", window, flags=re.S):
            candidate = _clean(span.group(1))
            if len(candidate) >= 20:
                snippet = candidate
                break
        if title:
            results.append({"title": title, "url": raw_url,
                            "snippet": snippet, "source": "Chrome"})
        if len(results) >= max_results:
            break
    return results


class ChromeSearchService:
    """Headless-Chromium Google search, with the Google JSON API
    and then DuckDuckGo as the non-browser fallbacks. Instances are
    stateless apart from `last_reason`, which explains the most
    recent [] result (disabled / blocked / no-automation /
    api-unconfigured / duckduckgo_error / no_results).
    """

    def __init__(self, max_results=None, timeout=None, executable_path=None):
        self.max_results = (max_results if max_results is not None
                            else max_search_results())
        self.timeout = (timeout if timeout is not None
                        else search_timeout_seconds())
        self.executable_path = (executable_path if executable_path is not None
                                else (os.environ.get("CHROME_EXECUTABLE_PATH", "") or "").strip() or None)
        self.last_reason = ""

    @property
    def enabled(self) -> bool:
        return browser_search_enabled()

    def search(self, query: str):
        """Return [{title, url, snippet, source}] or [] — never raise."""
        query = (query or "").strip()
        if not query:
            self.last_reason = "empty_query"
            return []
        if not self.enabled:
            self.last_reason = "browser_search_disabled"
            return []
        results = self._browser_search(query)
        if results:
            self.last_reason = ""
            return results
        results = self._google_api_search(query)
        if results:
            self.last_reason = ""
            return results
        results = self._duckduckgo_search(query)
        if results:
            self.last_reason = ""
            return results
        if not self.last_reason:
            self.last_reason = "no_results"
        return []

    # -- (b) headless Chromium -----------------------------------------
    def _browser_search(self, query: str):
        try:
            return self._playwright_search(query)
        except ImportError:
            pass
        except Exception:
            self.last_reason = "browser_error"
            return []
        try:
            return self._selenium_search(query)
        except ImportError:
            self.last_reason = "no_browser_automation_installed"
            return []
        except Exception:
            self.last_reason = "browser_error"
            return []

    def _playwright_search(self, query: str):
        from playwright.sync_api import sync_playwright  # optional dependency

        url = ("https://www.google.com/search?q=" + urllib.parse.quote(query)
               + "&num=" + str(self.max_results))
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        if self.executable_path:
            kwargs["executable_path"] = self.executable_path
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(**kwargs)
            try:
                context = browser.new_context()
                try:
                    page = context.new_page()
                    page.goto(url, timeout=self.timeout * 1000,
                              wait_until="domcontentloaded")
                    page_html = page.content()
                finally:
                    context.close()
            finally:
                browser.close()
        if _is_blocked_page(page_html):
            self.last_reason = "browser_blocked"
            return []
        results = parse_google_results_page(page_html, self.max_results)
        if not results:
            self.last_reason = "browser_no_results"
        return results

    def _selenium_search(self, query: str):
        from selenium import webdriver  # optional dependency
        from selenium.webdriver.chrome.options import Options

        url = ("https://www.google.com/search?q=" + urllib.parse.quote(query)
               + "&num=" + str(self.max_results))
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        if self.executable_path:
            options.binary_location = self.executable_path
        driver = webdriver.Chrome(options=options)
        try:
            driver.set_page_load_timeout(self.timeout)
            driver.get(url)
            page_html = driver.page_source
        finally:
            driver.quit()
        if _is_blocked_page(page_html):
            self.last_reason = "browser_blocked"
            return []
        results = parse_google_results_page(page_html, self.max_results)
        if not results:
            self.last_reason = "browser_no_results"
        return results

    # -- (c) Google Programmable Search JSON API -----------------------
    def _google_configured(self) -> bool:
        return bool((os.environ.get("GOOGLE_SEARCH_API_KEY", "") or "").strip()
                    and (os.environ.get("GOOGLE_SEARCH_CX", "") or "").strip())

    def _google_api_search(self, query: str):
        """Google JSON API results, or [] when unconfigured or on
        any failure. The key is sent only to Google's API endpoint,
        as the API requires; it is never logged or stored."""
        if not self._google_configured():
            self.last_reason = "google_api_unconfigured"
            return []
        try:
            base = ((os.environ.get("GOOGLE_SEARCH_BASE", "") or "").strip()
                    or "https://www.googleapis.com/customsearch/v1")
            params = {
                "q": query,
                "key": (os.environ.get("GOOGLE_SEARCH_API_KEY", "") or "").strip(),
                "cx": (os.environ.get("GOOGLE_SEARCH_CX", "") or "").strip(),
                "num": str(max(1, min(int(self.max_results), 10))),
                "safe": "active",
            }
            url = base.rstrip("/") + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    self.last_reason = "google_api_error"
                    return []
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                self.last_reason = "google_api_no_results"
                return []
            results = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                results.append({
                    "title": str(item.get("title", "") or "").strip(),
                    "url": str(item.get("link", "") or "").strip(),
                    "snippet": str(item.get("snippet", "") or "").strip(),
                    "source": "Google",
                })
                if len(results) >= self.max_results:
                    break
            return results
        except Exception:
            self.last_reason = "google_api_error"
            return []

    # -- (d) DuckDuckGo fallback (no keys, no browser) ------------------
    def _duckduckgo_search(self, query: str):
        """No-keys web fallback: the ddgs library first, else the
        DuckDuckGo HTML endpoint parsed with regex only. Never
        raises; [] on any failure. This is what keeps search
        working on hosts with no browser automation and no Google
        API keys (e.g. Render free), where the layers above
        return nothing."""
        try:
            from ddgs import DDGS  # type: ignore
            with DDGS() as ddgs:
                out = []
                for r in ddgs.text(query, max_results=self.max_results) or []:
                    out.append({"title": r.get("title", ""),
                                "url": r.get("href", ""),
                                "snippet": r.get("body", ""),
                                "source": "DuckDuckGo"})
                if out:
                    return out
        except Exception:
            pass
        try:
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                page = resp.read().decode("utf-8", errors="ignore")
            results = []
            for block in re.findall(
                    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
                    r'.*?class="result__snippet"[^>]*>(.*?)</a>',
                    page, flags=re.S):
                link, title, snippet = block
                # DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<real url>
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                real = parsed.get("uddg", [link])[0]
                results.append({
                    "title": _clean(title),
                    "url": urllib.parse.unquote(real),
                    "snippet": _clean(snippet),
                    "source": "DuckDuckGo",
                })
                if len(results) >= self.max_results:
                    break
            if not results:
                self.last_reason = "duckduckgo_no_results"
            return results
        except Exception:
            self.last_reason = "duckduckgo_error"
            return []
