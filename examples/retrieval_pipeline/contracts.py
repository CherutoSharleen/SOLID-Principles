from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]:
        ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        ...
