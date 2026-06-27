"""Multi-source web crawl for Uchi knowledge gap filling.

Architecture:
  Each source is a WebSource subclass with a single fetch(query) -> str method.
  MultiSourceSearch queries all enabled sources in parallel and merges results.
  perform_web_search() is the unchanged public interface used by OmniRouter.

Sources (all CPU/network only, no GPU):
  DuckDuckGoSource  — general web snippets via DuckDuckGo HTML (original)
  WikipediaSource   — article summaries via Wikipedia REST API
  ArxivSource       — paper abstracts via arXiv API (scientific/technical)
  NewsRSSSource     — headlines from configurable RSS feeds

Adding a new source: subclass WebSource, implement fetch(), register in
SOURCES dict, add to DEFAULT_SOURCES.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT  = 8    # seconds per source
_DEFAULT_MAX_BYTES = 2000  # chars to keep per source result


# ── SVO triple extractor (shared) ────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                           capture_output=True)
            import spacy
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _extract_svo(text: str) -> str:
    """Augment text with SVO triples for trie ingestion."""
    try:
        nlp  = _get_nlp()
        doc  = nlp(text[:5000])
        triples = []
        for sent in doc.sents:
            subj = verb = obj = None
            for tok in sent:
                if "subj" in tok.dep_:
                    subj = tok.text.lower()
                elif "obj" in tok.dep_:
                    obj = tok.text.lower()
                elif tok.pos_ == "VERB":
                    verb = tok.lemma_.lower()
            if subj and verb and obj:
                triples.append(f"{subj} {verb} {obj}")
        if triples:
            return text + " " + " ".join(triples)
    except Exception:
        pass
    return text


# ── base class ────────────────────────────────────────────────────────────────

class WebSource(ABC):
    """A single crawlable knowledge source."""

    name: str = "base"
    _headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    @abstractmethod
    def fetch(self, query: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        """Return a plain-text string of knowledge for the query, or ''."""

    def _get(self, url: str, timeout: int, **kwargs) -> Optional[requests.Response]:
        try:
            r = requests.get(url, headers=self._headers, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            _log.debug("%s fetch error: %s", self.name, e)
            return None


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

class DuckDuckGoSource(WebSource):
    name = "duckduckgo"

    def fetch(self, query: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        r = self._get(url, timeout)
        if r is None:
            return ""
        soup    = BeautifulSoup(r.text, "html.parser")
        snippets = soup.find_all("a", class_="result__snippet")
        texts   = [s.get_text(separator=" ", strip=True) for s in snippets[:5]]
        return " ".join(texts).lower()[:_DEFAULT_MAX_BYTES]


# ── Wikipedia ────────────────────────────────────────────────────────────────

class WikipediaSource(WebSource):
    name = "wikipedia"
    _API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
    _SEARCH = "https://en.wikipedia.org/w/api.php"

    def fetch(self, query: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        # First try direct title lookup, then fall back to search.
        title = query.replace(" ", "_")
        r = self._get(self._API.format(urllib.parse.quote(title)), timeout)
        if r is not None:
            try:
                data = r.json()
                extract = data.get("extract", "")
                if extract and len(extract) > 80:
                    return extract[:_DEFAULT_MAX_BYTES].lower()
            except Exception:
                pass

        # Search API fallback
        params = {
            "action": "query", "list": "search",
            "srsearch": query, "format": "json", "srlimit": 1,
        }
        r2 = self._get(self._SEARCH, timeout, params=params)
        if r2 is None:
            return ""
        try:
            hits = r2.json()["query"]["search"]
            if not hits:
                return ""
            page_title = hits[0]["title"].replace(" ", "_")
            r3 = self._get(self._API.format(urllib.parse.quote(page_title)), timeout)
            if r3 is None:
                return ""
            data = r3.json()
            return data.get("extract", "")[:_DEFAULT_MAX_BYTES].lower()
        except Exception:
            return ""


# ── arXiv ─────────────────────────────────────────────────────────────────────

class ArxivSource(WebSource):
    name = "arxiv"
    _API = "http://export.arxiv.org/api/query"

    def fetch(self, query: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        params = {
            "search_query": f"all:{urllib.parse.quote(query)}",
            "max_results": 3,
            "sortBy": "relevance",
        }
        r = self._get(self._API, timeout, params=params)
        if r is None:
            return ""
        try:
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(r.text)
            parts = []
            for entry in root.findall("atom:entry", ns)[:3]:
                title   = (entry.findtext("atom:title", "", ns) or "").strip()
                summary = (entry.findtext("atom:summary", "", ns) or "").strip()
                if title or summary:
                    parts.append(f"{title}. {summary}")
            combined = " ".join(parts).lower()
            return re.sub(r"\s+", " ", combined)[:_DEFAULT_MAX_BYTES]
        except Exception:
            return ""


# ── News RSS ──────────────────────────────────────────────────────────────────

_DEFAULT_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]

class NewsRSSSource(WebSource):
    name = "news_rss"

    def __init__(self, feeds: List[str] = _DEFAULT_RSS_FEEDS):
        self._feeds = feeds

    def fetch(self, query: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        query_words = set(query.lower().split())
        matched: List[str] = []

        for feed_url in self._feeds:
            r = self._get(feed_url, timeout)
            if r is None:
                continue
            try:
                root  = ET.fromstring(r.content)
                items = root.findall(".//item")
                for item in items[:20]:
                    title = (item.findtext("title") or "").strip()
                    desc  = (item.findtext("description") or "").strip()
                    text  = f"{title} {desc}".lower()
                    # Only include if at least one query word appears
                    if any(w in text for w in query_words if len(w) > 3):
                        matched.append(re.sub(r"<[^>]+>", "", text))
                        if len(matched) >= 3:
                            break
            except Exception:
                continue

        return " ".join(matched)[:_DEFAULT_MAX_BYTES]


# ── multi-source orchestrator ─────────────────────────────────────────────────

SOURCES: dict[str, type[WebSource]] = {
    "duckduckgo": DuckDuckGoSource,
    "wikipedia":  WikipediaSource,
    "arxiv":      ArxivSource,
    "news_rss":   NewsRSSSource,
}

DEFAULT_SOURCES = ["duckduckgo", "wikipedia", "arxiv", "news_rss"]


class MultiSourceSearch:
    """Query multiple sources in parallel and merge results."""

    def __init__(self, source_names: List[str] = DEFAULT_SOURCES,
                 timeout: int = _DEFAULT_TIMEOUT):
        self._sources: List[WebSource] = []
        for name in source_names:
            cls = SOURCES.get(name)
            if cls:
                self._sources.append(cls())
            else:
                _log.warning("Unknown web source '%s' — skipping.", name)
        self._timeout = timeout

    def search(self, query: str) -> str:
        if not self._sources:
            return ""

        results: List[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._sources)) as pool:
            future_to_src = {
                pool.submit(src.fetch, query, self._timeout): src.name
                for src in self._sources
            }
            for fut in concurrent.futures.as_completed(
                future_to_src, timeout=self._timeout + 2
            ):
                name = future_to_src[fut]
                try:
                    text = fut.result()
                    if text and text.strip():
                        _log.debug("%s returned %d chars", name, len(text))
                        results.append(text.strip())
                except Exception as e:
                    _log.debug("%s failed: %s", name, e)

        return " ".join(results)


# ── public API (unchanged interface for OmniRouter) ───────────────────────────

_searcher: Optional[MultiSourceSearch] = None

def _get_searcher() -> MultiSourceSearch:
    global _searcher
    if _searcher is None:
        _searcher = MultiSourceSearch()
    return _searcher


def perform_web_search(query: str, max_results: int = 5,
                       sources: Optional[List[str]] = None) -> str:
    """
    Query all enabled web sources in parallel and return merged text.
    Drop-in replacement for the original single-source DuckDuckGo search.

    Args:
        query:       Search query string.
        max_results: Kept for interface compatibility; sources control depth.
        sources:     Optional list of source names to use. Defaults to all.
    """
    if sources:
        searcher = MultiSourceSearch(sources)
    else:
        searcher = _get_searcher()

    combined = searcher.search(query)
    if not combined:
        return ""
    return _extract_svo(combined.lower())


# ── skill_registry compatibility ──────────────────────────────────────────────

def search(query: str) -> str:
    """Alias used by skill_registry.py."""
    return perform_web_search(query)
