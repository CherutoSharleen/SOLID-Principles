from __future__ import annotations

import re
from collections import defaultdict

from contracts import Retriever, Tokenizer


class BasicTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())


class PrefixStemTokenizer:
    def tokenize(self, text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9_]+", text.lower())
        normalized: list[str] = []
        for token in raw:
            if token.endswith("ing") and len(token) > 5:
                normalized.append(token[:-3])
            elif token.endswith("ed") and len(token) > 4:
                normalized.append(token[:-2])
            else:
                normalized.append(token)
        return normalized


class StopwordTokenizer:
    STOPWORDS = {"the", "and", "or", "for", "to", "in", "a", "of"}

    def tokenize(self, text: str) -> list[str]:
        base = re.findall(r"[a-z0-9_]+", text.lower())
        return [token for token in base if token not in self.STOPWORDS]


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


def unique_query_tokens(query: str, tokenizer: Tokenizer) -> list[str]:
    return sorted(set(tokenizer.tokenize(query)))
