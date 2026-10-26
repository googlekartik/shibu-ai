"""Free web search via DuckDuckGo (html endpoint, no API key).

Returns a list of {title, url, snippet, source}. Gracefully returns
[] on any network / parsing failure — the caller shows a friendly message.
Results are always labelled as web-derived in the UI.
"""
import html as html_lib
import re
import urllib.parse
import urllib.request


class WebSearchService:
    def __init__(self, max_results=5):
        self.max_results = max_results

    def search(self, query: str):
        query = (query or "").strip()
        if not query:
            return []
        # Try the ddgs library first if the user has it installed.
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
        # Fallback: DuckDuckGo html endpoint, parsed with regex only.
        try:
            url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                page = resp.read().decode("utf-8", errors="ignore")
            results = []
            for block in re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
                                    page, flags=re.S):
                link, title, snippet = block
                # DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<real url>
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                real = parsed.get("uddg", [link])[0]
                results.append({
                    "title": html_lib.unescape(re.sub(r"<[^>]+>", "", title)).strip(),
                    "url": urllib.parse.unquote(real),
                    "snippet": html_lib.unescape(re.sub(r"<[^>]+>", "", snippet)).strip(),
                    "source": "DuckDuckGo",
                })
                if len(results) >= self.max_results:
                    break
            return results
        except Exception:
            return []
