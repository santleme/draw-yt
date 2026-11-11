import tempfile
import unittest
import random
from pathlib import Path

from PIL import Image, ImageDraw

from drawyt.hashing import hamming
from drawyt.triage import _HammingIndex, triage_frames


class TestTriage(unittest.TestCase):
    def test_hamming_index_matches_sequential_scan(self):
        rng = random.Random(1729)
        values = [rng.getrandbits(64) for _ in range(300)]
        values.extend([values[5], values[5] ^ 0b10101])

        for threshold in (0, 1, 4, 8, 32, 63, 64, 70):
            index = _HammingIndex(threshold)
            representatives = []
            for value in values:
                expected = any(hamming(value, old) <= threshold for old in representatives)
                self.assertEqual(index.has_nearby(value), expected, (threshold, value))
                if not expected:
                    representatives.append(value)
                    index.add(value)

    def test_negative_hash_threshold_never_marks_duplicates(self):
        index = _HammingIndex(-1)
        index.add(0)
        self.assertFalse(index.has_nearby(0))

    def test_detects_dark_marks_on_bright_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp)
            image = Image.new("RGB", (160, 120), (35, 35, 35))
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 10, 140, 110), fill=(245, 245, 240))
            draw.line((40, 35, 115, 35), fill=(15, 15, 15), width=4)
            draw.line((40, 60, 100, 80), fill=(15, 15, 15), width=4)
            image.save(frames / "paper.jpg")

            row = triage_frames(frames, video_id="paper-video")[0]

            self.assertGreater(row["paper_like_ratio"], 0.1)
            self.assertGreater(row["ink_like_ratio"], 0)
            self.assertIn("dark_marks_on_bright_region", row["review_reason"])

    def test_blank_paper_has_no_ink_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp)
            Image.new("RGB", (160, 120), (245, 245, 240)).save(frames / "blank.jpg")

            row = triage_frames(frames)[0]

            self.assertEqual(row["ink_like_ratio"], 0)

    def test_dark_texture_is_not_scored_as_paper_or_ink(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp)
            image = Image.new("RGB", (160, 120), (80, 75, 70))
            draw = ImageDraw.Draw(image)
            for x in range(0, 160, 8):
                draw.line((x, 0, x + 30, 120), fill=(30, 30, 30), width=3)
            image.save(frames / "texture.jpg")

            row = triage_frames(frames)[0]

            self.assertLess(row["paper_like_ratio"], 0.15)
            self.assertEqual(row["ink_like_ratio"], 0)

    def test_empty_directory_returns_no_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(triage_frames(Path(tmp)), [])


if __name__ == "__main__":
    unittest.main()
