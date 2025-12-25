from __future__ import annotations

import re
from collections import defaultdict

from contracts import Retriever, Tokenizer


class BasicTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())


class BooleanRetriever(Retriever):
    def __init__(self, documents: dict[str, str], tokenizer: Tokenizer) -> None:
        self.documents = documents
        self.tokenizer = tokenizer
        self.index = self._build_index(documents)

    def _build_index(self, documents: dict[str, str]) -> dict[str, set[str]]:
        index: dict[str, set[str]] = defaultdict(set)
        for doc_id, text in documents.items():
            for token in self.tokenizer.tokenize(text):
                index[token].add(doc_id)
        return index

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_tokens = self.tokenizer.tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        for token in query_tokens:
            for doc_id in self.index.get(token, set()):
                scores[doc_id] += 1.0
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
        return ranked[:top_k]
