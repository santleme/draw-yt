import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTriageCsvContract(unittest.TestCase):
    def _make_frames(self, frames: Path) -> None:
        frames.mkdir()
        Image.new("RGB", (160, 120), "white").save(frames / "blank.jpg")
        image = Image.new("RGB", (160, 120), (40, 40, 40))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 10, 140, 110), fill="white")
        draw.line((35, 35, 120, 80), fill="black", width=5)
        image.save(frames / "drawn.jpg")

    def _run_triage(self, frames: Path, output: Path, *extra: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        return subprocess.run(
            [sys.executable, "-m", "drawyt", "triage", str(frames), "--out", str(output), *extra],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_writes_stable_layered_csv_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            self._make_frames(frames)

            output_a = root / "review-a.csv"
            result = self._run_triage(frames, output_a)

            self.assertEqual(result.returncode, 0, result.stderr)
            with output_a.open(newline="", encoding="utf-8") as fh:
                rows_a = list(csv.DictReader(fh))
                fields = set(rows_a[0])

            required = {
                "frame_id", "video_id", "review_score", "review_reason", "review_decision",
                "quality_status", "sheet_status", "sheet_presence", "hand_status", "hand_presence",
                "hand_landmarks_status", "hand_landmarks_points", "strokes_status", "strokes_density",
                "ink_status", "ink_density", "scene_status", "scene_class", "scene_confidence",
            }
            self.assertTrue(required.issubset(fields), required - fields)
            self.assertEqual(len(rows_a), 2)
            self.assertTrue(all(float(row["review_score"]) >= 0.0 for row in rows_a))
            self.assertTrue(all(float(row["sheet_presence"]) >= 0.0 for row in rows_a))
            self.assertTrue(all(float(row["strokes_density"]) >= 0.0 for row in rows_a))
            self.assertTrue(all(row["hand_status"] == "unavailable" for row in rows_a))
            self.assertTrue(all(row["hand_presence"] == "" for row in rows_a))
            self.assertTrue(all(row["hand_landmarks_status"] == "unavailable" for row in rows_a))
            self.assertTrue(all(row["scene_status"] == "unavailable" for row in rows_a))
            self.assertTrue(all(row["scene_class"] == "" for row in rows_a))

            output_b = root / "review-b.csv"
            result = self._run_triage(frames, output_b)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output_b.open(newline="", encoding="utf-8") as fh:
                rows_b = list(csv.DictReader(fh))
            self.assertEqual([row["frame_id"] for row in rows_a], [row["frame_id"] for row in rows_b])

    def test_cli_limit_caps_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            self._make_frames(frames)
            output = root / "limited.csv"

            result = self._run_triage(frames, output, "--limit", "1")

            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="", encoding="utf-8") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 1)


if __name__ == "__main__":
    unittest.main()
