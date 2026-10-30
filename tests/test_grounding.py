import unittest

from claims_sample_miner.grounding import grounded_passage


class GroundingTests(unittest.TestCase):
    def test_layout_normalization_returns_the_original_pdf_substring(self):
        text = "The ﬁnding\n  increased signi\u00adficantly."
        result = grounded_passage([{"span_id": "actual", "text": text}],
                                  "actual", "The finding increased significantly.")
        self.assertEqual(result, {"span_id": "actual", "quote": text})

    def test_wrong_span_id_needs_a_unique_exact_passage(self):
        spans = [{"span_id": "a", "text": "Observed result."},
                 {"span_id": "b", "text": "Other result."}]
        self.assertEqual(grounded_passage(spans, "missing", "Observed result.")["span_id"], "a")
        spans.append({"span_id": "c", "text": "Observed result."})
        self.assertIsNone(grounded_passage(spans, "missing", "Observed result."))
        self.assertEqual(grounded_passage(spans, "a", "Observed result.")["span_id"], "a")

    def test_invented_paraphrases_and_empty_quotes_are_not_evidence(self):
        spans = [{"span_id": "a", "text": "An association was observed."}]
        for quote in ["A causal effect was proven.", "", "  ", None]:
            self.assertIsNone(grounded_passage(spans, "a", quote))
