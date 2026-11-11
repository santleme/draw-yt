import unittest

import numpy as np
from PIL import Image, ImageDraw

from drawyt.layers import analyze_layers


class TestAnalyzeLayers(unittest.TestCase):
    def test_returns_independent_layer_results_and_unavailable_detectors(self):
        image = Image.new("RGB", (160, 120), "white")

        result = analyze_layers(image)

        self.assertEqual(
            set(result), {"quality", "paper", "hand", "hand_landmarks", "strokes", "ink", "scene"}
        )
        self.assertEqual(result["quality"]["status"], "heuristic")
        self.assertEqual(result["paper"]["status"], "heuristic")
        self.assertEqual(result["hand"]["status"], "unavailable")
        self.assertIsNone(result["hand"]["presence"])
        self.assertEqual(result["hand_landmarks"]["status"], "unavailable")
        self.assertIsNone(result["hand_landmarks"]["points"])
        self.assertEqual(result["strokes"]["status"], "heuristic")
        self.assertEqual(result["ink"]["status"], "heuristic")
        self.assertEqual(result["scene"]["status"], "unavailable")
        self.assertIsNone(result["scene"]["class"])
        self.assertIsNone(result["scene"]["confidence"])

    def test_detects_paper_and_marks_without_claiming_hand_or_scene(self):
        image = Image.new("RGB", (160, 120), (45, 45, 45))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 10, 140, 110), fill=(245, 245, 240))
        draw.line((38, 40, 120, 40), fill=(10, 10, 10), width=5)

        result = analyze_layers(image)

        self.assertGreater(result["paper"]["presence"], 0.1)
        self.assertGreater(result["ink"]["density"], 0)
        self.assertGreater(result["strokes"]["density"], 0)
        self.assertEqual(result["hand"]["presence"], None)
        self.assertIsNone(result["scene"]["class"])

    def test_blank_paper_has_no_ink_signal(self):
        result = analyze_layers(Image.new("RGB", (160, 120), "white"))

        self.assertGreater(result["paper"]["presence"], 0.9)
        self.assertEqual(result["ink"]["density"], 0)

    def test_layer_values_are_json_serializable_scalars(self):
        result = analyze_layers(Image.new("RGB", (24, 24), "white"))

        self.assertTrue(np.isscalar(result["paper"]["presence"]))
        self.assertTrue(np.isscalar(result["ink"]["density"]))


if __name__ == "__main__":
    unittest.main()
