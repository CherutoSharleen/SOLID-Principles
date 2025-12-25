from __future__ import annotations

import math
from collections import Counter, defaultdict

from contracts import Retriever, Tokenizer


class BM25Retriever(Retriever):
    def __init__(self, documents: dict[str, str], tokenizer: Tokenizer, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.doc_tokens = {doc_id: tokenizer.tokenize(text) for doc_id, text in documents.items()}
        self.doc_lengths = {doc_id: len(tokens) for doc_id, tokens in self.doc_tokens.items()}
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, len(self.doc_lengths))
        self.df = self._document_frequencies()

    def _document_frequencies(self) -> dict[str, int]:
        df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens.values():
            for token in set(tokens):
                df[token] += 1
        return dict(df)

    def _idf(self, token: str) -> float:
        n = len(self.documents)
        df = self.df.get(token, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        query_tokens = self.tokenizer.tokenize(query)
        scores: dict[str, float] = {}
        for doc_id, tokens in self.doc_tokens.items():
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf == 0:
                    continue
                idf = self._idf(token)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[doc_id] / max(1e-9, self.avg_doc_len)))
                score += idf * ((tf * (self.k1 + 1)) / denom)
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
        return ranked[:top_k]
