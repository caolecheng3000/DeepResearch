"""Fixed evidence tools for reproducible evaluations; never fall back to network."""
from __future__ import annotations
import json
import math
import re
from collections import Counter
from pathlib import Path
from src.utils.tracing import record_event
from .web_search import BaseWebSearchTool
from .browser import BaseBrowserTool


class SnapshotSearchTool(BaseWebSearchTool):
    description = "Search the fixed evaluation evidence corpus. Input: query, optional top_n."

    def get_openai_tool_schema(self):
        schema = super().get_openai_tool_schema()
        schema["function"]["parameters"]["properties"]["top_n"].update(minimum=0, maximum=20)
        return schema

    def __init__(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.documents = data["documents"]
        self.mode = data.get("search_mode", "stable")
        if self.mode not in ("stable", "bm25"):
            raise ValueError("Unknown snapshot search_mode")
        self.snippet_chars = data.get("snippet_chars", 600)
        self._terms = [Counter(self._tokenize(d["title"] + " " + d["text"])) for d in self.documents]
        self._df = Counter(t for terms in self._terms for t in terms)
        self._average = sum(map(lambda c: sum(c.values()), self._terms)) / max(1, len(self.documents))

    @staticmethod
    def _tokenize(text):
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        for block in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.extend(block[i:i+2] for i in range(len(block)-1))
        return tokens

    async def execute(self, query: str, top_n: int = 5) -> dict:
        if type(top_n) is not int or not 0 <= top_n <= 20:
            raise ValueError("top_n must be an integer in 0..20")
        ranked = list(range(len(self.documents)))
        if self.mode == "bm25":
            query_terms = set(self._tokenize(query))
            def score(i):
                terms = self._terms[i]; length = sum(terms.values()); n = len(self.documents)
                return sum(math.log(1 + (n-self._df[t]+0.5)/(self._df[t]+0.5)) *
                    terms[t]*2.2/(terms[t]+1.2*(0.25+0.75*length/max(1,self._average)))
                    for t in query_terms if terms[t])
            ranked.sort(key=lambda i: (-score(i), i))
            ranked = [i for i in ranked if score(i) > 0]
        docs = [self.documents[i] for i in ranked[:top_n]]
        rows = [{"title": d["title"], "url": d["url"],
                 "snippet": d["text"] if self.mode == "stable" else d["text"][:self.snippet_chars]} for d in docs]
        record_event("snapshot_search", {"query": query, "urls": [d["url"] for d in docs], "mode": self.mode})
        return {"query": query, "results": rows, "source": "fixed_snapshot"}


class SnapshotBrowserTool(BaseBrowserTool):
    description = "Read a URL from the fixed evaluation evidence corpus; unavailable URLs return an error."

    def __init__(self, path: str):
        self.documents = {d["url"]: d for d in json.loads(Path(path).read_text(encoding="utf-8"))["documents"]}

    async def execute(self, url: str, max_chars: int = 12000) -> str:
        doc = self.documents.get(url)
        if doc is None:
            return "Error: URL is not in the fixed evidence snapshot."
        record_event("snapshot_read", {"url": url, "chars": min(max_chars, len(doc["text"]))})
        return doc["text"][:max_chars]
