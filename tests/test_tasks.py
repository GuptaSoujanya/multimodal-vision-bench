import unittest
from vision_bench.models.mock_vlm import MockVLM
from vision_bench.tasks.chartqa import ChartQATask
from vision_bench.tasks.docvqa import DocVQATask
from vision_bench.tasks.spatial import SpatialGroundingTask

class TestVisionTasks(unittest.TestCase):
    def setUp(self):
        self.vlm = MockVLM("test-vlm")

    def test_chartqa(self):
        res = ChartQATask().run_evaluation(self.vlm)
        self.assertEqual(res["task"], "chartqa")
        self.assertGreaterEqual(res["accuracy"], 0.5)

    def test_docvqa(self):
        res = DocVQATask().run_evaluation(self.vlm)
        self.assertEqual(res["task"], "docvqa")
        self.assertIn("mean_cer", res)

    def test_spatial(self):
        res = SpatialGroundingTask().run_evaluation(self.vlm)
        self.assertEqual(res["task"], "spatial")
        self.assertIn("mean_iou", res)

if __name__ == "__main__":
    unittest.main()
