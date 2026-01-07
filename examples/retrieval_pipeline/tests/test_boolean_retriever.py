import unittest

from boolean_retriever import BasicTokenizer, BooleanRetriever, PrefixStemTokenizer, StopwordTokenizer
from corpus import DOCUMENTS


class BooleanRetrieverTests(unittest.TestCase):
    def test_query_ranks_boolean_document(self) -> None:
        retriever = BooleanRetriever(DOCUMENTS, BasicTokenizer())
        top_doc = retriever.search("boolean operators", top_k=1)[0][0]
        self.assertEqual(top_doc, "d2")

    def test_empty_query_returns_no_scores(self) -> None:
        retriever = BooleanRetriever(DOCUMENTS, BasicTokenizer())
        self.assertEqual(retriever.search("", top_k=3), [])

    def test_stem_tokenizer_normalizes_suffixes(self) -> None:
        tokens = PrefixStemTokenizer().tokenize("ranking ranked")
        self.assertIn("rank", tokens)

    def test_stopword_tokenizer_removes_common_words(self) -> None:
        tokens = StopwordTokenizer().tokenize("the ranking of data and search")
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)


if __name__ == "__main__":
    unittest.main()
