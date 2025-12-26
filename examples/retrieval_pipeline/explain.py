from __future__ import annotations

from collections import Counter

from contracts import Tokenizer


def explain_overlap(query: str, document: str, tokenizer: Tokenizer) -> dict[str, int]:
    query_tokens = tokenizer.tokenize(query)
    doc_tokens = tokenizer.tokenize(document)
    doc_counts = Counter(doc_tokens)
    return {token: doc_counts[token] for token in query_tokens if token in doc_counts}
