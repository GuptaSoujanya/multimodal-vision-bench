import unittest
from vision_bench.metrics import calculate_cer, calculate_iou_2d, relaxed_chart_accuracy

class TestVision(unittest.TestCase):
    def test_cer(self):
        self.assertEqual(calculate_cer("hello", "hello"), 0.0)
    def test_iou(self):
        self.assertAlmostEqual(calculate_iou_2d([0,0,1,1], [0,0,1,1]), 1.0)
    def test_relaxed(self):
        self.assertTrue(relaxed_chart_accuracy("42.1%", "42.0%"))

if __name__ == "__main__":
    unittest.main()
