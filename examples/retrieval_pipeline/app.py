from __future__ import annotations

from bm25_retriever import BM25Retriever
from boolean_retriever import BasicTokenizer, BooleanRetriever
from corpus import DOCUMENTS
from explain import explain_overlap


def main() -> None:
    tokenizer = BasicTokenizer()
    mode = input("Retriever mode (boolean/bm25) [boolean]: ").strip().lower() or "boolean"
    retriever = BM25Retriever(DOCUMENTS, tokenizer) if mode == "bm25" else BooleanRetriever(DOCUMENTS, tokenizer)
    print("SOLID Retrieval Pipeline Demo")
    query = input("Enter query terms: ").strip()
    results = retriever.search(query, top_k=3)
    if not results:
        print("No matches found.")
        return
    for doc_id, score in results:
        print(f"- {doc_id}: score={score:.2f}")
        print(f"  {DOCUMENTS[doc_id]}")
        if mode == "boolean":
            terms = explain_overlap(query, DOCUMENTS[doc_id], tokenizer)
            if terms:
                detail = ", ".join(f"{term}:{count}" for term, count in sorted(terms.items()))
                print(f"  matched terms -> {detail}")


if __name__ == "__main__":
    main()
