"""Stress battery: 80 geometry cases + 20 maximum-package checks."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from drawyt.review_pack import build_review_pack, tile_regions

REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_geometry(test: unittest.TestCase, width: int, height: int, tile_size: int, overlap: int) -> None:
    regions = tile_regions(width, height, tile_size, overlap)
    test.assertTrue(regions)
    by_row: dict[int, list[dict]] = {}
    by_column: dict[int, list[dict]] = {}
    for region in regions:
        test.assertGreater(region["width"], 0)
        test.assertGreater(region["height"], 0)
        test.assertGreaterEqual(region["x"], 0)
        test.assertGreaterEqual(region["y"], 0)
        test.assertLessEqual(region["x"] + region["width"], width)
        test.assertLessEqual(region["y"] + region["height"], height)
        by_row.setdefault(region["row"], []).append(region)
        by_column.setdefault(region["column"], []).append(region)

    for row in by_row.values():
        row.sort(key=lambda item: item["x"])
        test.assertEqual(row[0]["x"], 0)
        test.assertEqual(row[-1]["x"] + row[-1]["width"], width)
        for left, right in zip(row, row[1:]):
            test.assertLessEqual(right["x"], left["x"] + left["width"])
    for column in by_column.values():
        column.sort(key=lambda item: item["y"])
        test.assertEqual(column[0]["y"], 0)
        test.assertEqual(column[-1]["y"] + column[-1]["height"], height)
        for top, bottom in zip(column, column[1:]):
            test.assertLessEqual(bottom["y"], top["y"] + top["height"])


class GeometryStressBattery(unittest.TestCase):
    pass


_rng = random.Random(20261110)
_tile_sizes = (64, 96, 128, 192, 256, 384, 512, 768)
for _case in range(80):
    _size = _tile_sizes[_rng.randrange(len(_tile_sizes))]
    _overlap = _rng.randrange(0, min(_size, 128))
    _width = _rng.randrange(1, 4097)
    _height = _rng.randrange(1, 3073)

    def _make_geometry_test(width=_width, height=_height, tile_size=_size, overlap=_overlap):
        def test_geometry(self):
            _assert_geometry(self, width, height, tile_size, overlap)
        return test_geometry

    setattr(GeometryStressBattery, f"test_geometry_case_{_case:02d}", _make_geometry_test())


class MaxPackageStressBattery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".review-pack-stress-", dir=REPO_ROOT)
        cls.root = Path(cls.temp.name)
        cls.frames = cls.root / "frames"
        cls.frames.mkdir()
        source = Image.new("RGB", (1792, 1792), (246, 246, 242))
        draw = ImageDraw.Draw(source)
        for offset in range(16, 1792, 47):
            draw.line((offset, 0, 1791 - offset // 2, 1791), fill=(15, 24, 35), width=3)
            draw.line((0, offset, 1791, 1791 - offset), fill=(30, 35, 40), width=2)
        cls.source_path = cls.frames / "frame_000000.jpg"
        source.save(cls.source_path, quality=92)
        cls.source_hash = hashlib.sha256(cls.source_path.read_bytes()).hexdigest()
        cls.pack = cls.root / "package"
        cls.receipt = build_review_pack(
            cls.frames,
            cls.pack,
            video_id="stress-video",
            limit=1,
            tile_size=256,
            overlap=32,
            max_tiles_per_frame=64,
            max_total_tiles=64,
        )
        cls.manifest_lines = (cls.pack / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        cls.record = json.loads(cls.manifest_lines[0])
        cls.tiles = cls.record["tiles"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_hits_configured_per_frame_tile_limit(self):
        self.assertEqual(self.receipt["tiles"], 64)
        self.assertEqual(len(self.tiles), 64)

    def test_02_hits_configured_total_tile_limit(self):
        self.assertEqual(self.record["tile_config"]["max_total_tiles"], 64)

    def test_03_emits_exactly_one_review_unit(self):
        self.assertEqual(self.receipt["frames"], 1)
        self.assertEqual(len(self.manifest_lines), 1)
        self.assertEqual(self.record["review_unit"], "one_source_frame")

    def test_04_marks_tiles_as_detail_views_not_votes(self):
        self.assertTrue(self.record["tiles_are_detail_views_not_votes"])

    def test_05_preserves_source_sha256(self):
        self.assertEqual(self.record["source_sha256"], self.source_hash)

    def test_06_does_not_modify_source_image(self):
        self.assertEqual(hashlib.sha256(self.source_path.read_bytes()).hexdigest(), self.source_hash)

    def test_07_copies_full_image_byte_for_byte(self):
        full = self.pack / self.record["full_image"]["path"]
        self.assertEqual(full.read_bytes(), self.source_path.read_bytes())

    def test_08_full_asset_hash_matches_manifest(self):
        full = self.pack / self.record["full_image"]["path"]
        self.assertEqual(hashlib.sha256(full.read_bytes()).hexdigest(), self.record["full_image"]["sha256"])

    def test_09_tile_ids_are_unique(self):
        ids = [tile["tile_id"] for tile in self.tiles]
        self.assertEqual(len(ids), len(set(ids)))

    def test_10_tile_coordinates_are_unique(self):
        coords = [(tile["row"], tile["column"]) for tile in self.tiles]
        self.assertEqual(len(coords), len(set(coords)))

    def test_11_each_tile_links_to_the_source_frame(self):
        self.assertTrue(all(tile["frame_id"] == self.record["frame_id"] for tile in self.tiles))

    def test_12_each_tile_has_a_valid_content_hash(self):
        for tile in self.tiles:
            path = self.pack / tile["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), tile["sha256"])

    def test_13_each_tile_respects_source_bounds(self):
        for tile in self.tiles:
            self.assertGreater(tile["width"], 0)
            self.assertGreater(tile["height"], 0)
            self.assertLessEqual(tile["x"] + tile["width"], self.record["source_width"])
            self.assertLessEqual(tile["y"] + tile["height"], self.record["source_height"])

    def test_14_grid_has_eight_rows(self):
        self.assertEqual({tile["row"] for tile in self.tiles}, set(range(8)))

    def test_15_grid_has_eight_columns(self):
        self.assertEqual({tile["column"] for tile in self.tiles}, set(range(8)))

    def test_16_no_tile_exceeds_requested_edge_size(self):
        self.assertTrue(all(max(tile["width"], tile["height"]) <= 256 for tile in self.tiles))

    def test_17_entire_1792_square_is_covered_without_holes(self):
        coverage = bytearray(1792 * 1792)
        for tile in self.tiles:
            start_x, start_y = tile["x"], tile["y"]
            for y in range(start_y, start_y + tile["height"]):
                coverage[y * 1792 + start_x:y * 1792 + start_x + tile["width"]] = b"\1" * tile["width"]
        self.assertTrue(all(coverage))

    def test_18_horizontal_neighbors_overlap_by_configured_amount(self):
        row_zero = sorted((tile for tile in self.tiles if tile["row"] == 0), key=lambda item: item["column"])
        for left, right in zip(row_zero, row_zero[1:]):
            overlap = left["x"] + left["width"] - right["x"]
            self.assertEqual(overlap, 32)

    def test_19_vertical_neighbors_overlap_by_configured_amount(self):
        column_zero = sorted((tile for tile in self.tiles if tile["column"] == 0), key=lambda item: item["row"])
        for top, bottom in zip(column_zero, column_zero[1:]):
            overlap = top["y"] + top["height"] - bottom["y"]
            self.assertEqual(overlap, 32)

    def test_20_manifest_is_versioned_and_reproducible(self):
        self.assertEqual(self.record["schema_version"], "drawyt-review-pack-v1")
        repeat = build_review_pack(
            self.frames,
            self.root / "repeat",
            video_id="stress-video",
            limit=1,
            tile_size=256,
            overlap=32,
            max_tiles_per_frame=64,
            max_total_tiles=64,
        )
        self.assertEqual(repeat["package_id"], self.receipt["package_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
