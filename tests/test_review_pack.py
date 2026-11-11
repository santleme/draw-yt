import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from drawyt.review_pack import build_review_pack, tile_regions

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTileGeometry(unittest.TestCase):
    def test_tiles_cover_image_and_map_to_source_coordinates(self):
        width, height = 1000, 700
        regions = tile_regions(width, height, tile_size=512, overlap=64)
        covered = bytearray(width * height)
        for region in regions:
            x, y = region["x"], region["y"]
            right, bottom = x + region["width"], y + region["height"]
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(right, width)
            self.assertLessEqual(bottom, height)
            for row in range(y, bottom):
                covered[row * width + x:row * width + right] = b"\1" * (right - x)

        self.assertEqual(len(regions), 6)
        self.assertTrue(all(covered))
        first_row = [region for region in regions if region["row"] == 0]
        self.assertEqual(first_row[1]["x"], 448)
        self.assertEqual(first_row[1]["x"] - (first_row[0]["x"] + first_row[0]["width"] - 64), 0)

    def test_small_frame_is_not_redundantly_tiled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (200, 120), "white").save(frames / "small.jpg")

            receipt = build_review_pack(frames, root / "pack", tile_size=512, overlap=64)
            record = json.loads((root / "pack" / "manifest.jsonl").read_text().splitlines()[0])

            self.assertEqual(receipt["tiles"], 0)
            self.assertEqual(record["tiles"], [])
            self.assertTrue((root / "pack" / record["full_image"]["path"]).is_file())


class TestReviewPack(unittest.TestCase):
    def _make_frames(self, frames: Path) -> bytes:
        frames.mkdir()
        image = Image.new("RGB", (640, 360), (45, 45, 45))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 35, 560, 330), fill=(245, 245, 240))
        draw.line((120, 80, 510, 250), fill=(10, 10, 10), width=7)
        path = frames / "frame_000005.jpg"
        image.save(path)
        return path.read_bytes()

    def test_package_manifest_hashes_and_outputs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            original_bytes = self._make_frames(frames)
            first = build_review_pack(frames, root / "pack-a", video_id="video", tile_size=256, overlap=32)
            second = build_review_pack(frames, root / "pack-b", video_id="video", tile_size=256, overlap=32)
            manifest_a = root / "pack-a" / "manifest.jsonl"
            manifest_b = root / "pack-b" / "manifest.jsonl"
            row_a = json.loads(manifest_a.read_text(encoding="utf-8").splitlines()[0])
            row_b = json.loads(manifest_b.read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(first["package_id"], second["package_id"])
            self.assertEqual(row_a["package_id"], row_b["package_id"])
            self.assertEqual(row_a["review_unit"], "one_source_frame")
            self.assertTrue(row_a["tiles_are_detail_views_not_votes"])
            self.assertEqual(row_a["source_sha256"], hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual((frames / "frame_000005.jpg").read_bytes(), original_bytes)
            self.assertEqual(len(row_a["tiles"]), 6)

            full = root / "pack-a" / row_a["full_image"]["path"]
            self.assertEqual(hashlib.sha256(full.read_bytes()).hexdigest(), row_a["full_image"]["sha256"])
            for tile in row_a["tiles"]:
                path = root / "pack-a" / tile["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), tile["sha256"])
                self.assertEqual(tile["frame_id"] if "frame_id" in tile else row_a["frame_id"], row_a["frame_id"])
                self.assertGreater(tile["width"], 0)
                self.assertGreater(tile["height"], 0)
                self.assertLessEqual(tile["x"] + tile["width"], row_a["source_width"])
                self.assertLessEqual(tile["y"] + tile["height"], row_a["source_height"])

    def test_tile_limit_fails_before_creating_partial_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            self._make_frames(frames)
            out = root / "pack"

            with self.assertRaisesRegex(ValueError, "needs 6 tiles"):
                build_review_pack(frames, out, tile_size=256, overlap=32, max_tiles_per_frame=5)

            self.assertFalse(out.exists())

    def test_total_tile_limit_fails_before_creating_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            for index in range(2):
                Image.new("RGB", (600, 300), (index * 20, 40, 80)).save(frames / f"frame_{index}.jpg")
            out = root / "pack"

            with self.assertRaisesRegex(ValueError, "need 12 detail tiles"):
                build_review_pack(
                    frames, out, tile_size=256, overlap=32,
                    max_tiles_per_frame=8, max_total_tiles=11,
                )

            self.assertFalse(out.exists())

    def test_cli_generates_local_review_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            self._make_frames(frames)
            out = root / "review-pack"
            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
            result = subprocess.run(
                [sys.executable, "-m", "drawyt", "review-pack", str(frames), "--out", str(out), "--tile-size", "256", "--overlap", "32"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 frames, 6 detail tiles", result.stdout)
            records = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["tiles_are_detail_views_not_votes"], True)


if __name__ == "__main__":
    unittest.main()
