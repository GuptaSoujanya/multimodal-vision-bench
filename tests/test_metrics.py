import unittest
from vision_bench.metrics.ocr_levenshtein import calculate_cer, normalized_levenshtein_similarity
from vision_bench.metrics.box_iou import calculate_box_iou_2d
from vision_bench.metrics.relaxed_accuracy import relaxed_chart_match

class TestVisionMetrics(unittest.TestCase):
    def test_cer_and_levenshtein(self):
        sim = normalized_levenshtein_similarity("invoice", "invoice")
        self.assertEqual(sim, 1.0)
        cer = calculate_cer("abc", "abd")
        self.assertAlmostEqual(cer, 0.3333, places=3)

    def test_box_iou(self):
        b1 = [0.0, 0.0, 1.0, 1.0]
        b2 = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(calculate_box_iou_2d(b1, b2), 1.0)

    def test_relaxed_chart_accuracy(self):
        # 42.0 vs 42.5 is within 5% tolerance
        self.assertTrue(relaxed_chart_match("42.5%", "42.0%"))
        # 10 vs 20 is not
        self.assertFalse(relaxed_chart_match("10", "20"))

if __name__ == "__main__":
    unittest.main()
