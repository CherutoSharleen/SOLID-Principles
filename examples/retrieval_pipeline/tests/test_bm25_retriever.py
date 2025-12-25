import unittest

from bm25_retriever import BM25Retriever
from boolean_retriever import BasicTokenizer
from corpus import DOCUMENTS


class BM25RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_boolean_document_first(self) -> None:
        retriever = BM25Retriever(DOCUMENTS, BasicTokenizer())
        top_doc = retriever.search("boolean operators", top_k=1)[0][0]
        self.assertEqual(top_doc, "d2")

    def test_bm25_returns_empty_for_unknown_query(self) -> None:
        retriever = BM25Retriever(DOCUMENTS, BasicTokenizer())
        self.assertEqual(retriever.search("zzzz qqqq", top_k=3), [])


if __name__ == "__main__":
    unittest.main()
