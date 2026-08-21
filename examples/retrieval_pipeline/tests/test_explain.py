import unittest

from boolean_retriever import BasicTokenizer
from explain import explain_overlap, overlap_ratio, query_token_count


class ExplainTests(unittest.TestCase):
    def test_explain_overlap_counts_query_terms(self) -> None:
        tokenizer = BasicTokenizer()
        explanation = explain_overlap(
            "search ranking",
            "Search systems use ranking and search indexes.",
            tokenizer,
        )
        self.assertEqual(explanation["search"], 2)
        self.assertEqual(explanation["ranking"], 1)

    def test_overlap_ratio(self) -> None:
        tokenizer = BasicTokenizer()
        ratio = overlap_ratio("search ranking missing", "Search systems use ranking.", tokenizer)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_query_token_count(self) -> None:
        tokenizer = BasicTokenizer()
        self.assertEqual(query_token_count("search ranking model", tokenizer), 3)


if __name__ == "__main__":
    unittest.main()
