import unittest

from boolean_retriever import BasicTokenizer
from explain import explain_overlap


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


if __name__ == "__main__":
    unittest.main()
