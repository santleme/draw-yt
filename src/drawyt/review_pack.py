"""Build a local, traceable review bundle with full frames and detail tiles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from .triage import triage_frames

SCHEMA_VERSION = "drawyt-review-pack-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    return list(range(0, length, stride))


def tile_regions(width: int, height: int, tile_size: int, overlap: int) -> list[dict]:
    """Return an exhaustive row-major grid; edge tiles may be smaller."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    x_starts = _tile_starts(width, tile_size, overlap)
    y_starts = _tile_starts(height, tile_size, overlap)
    regions = []
    for row, y in enumerate(y_starts):
        for column, x in enumerate(x_starts):
            regions.append({
                "row": row,
                "column": column,
                "x": x,
                "y": y,
                "width": min(tile_size, width - x),
                "height": min(tile_size, height - y),
            })
    return regions


def build_review_pack(
    frames_dir: Path,
    out_dir: Path,
    *,
    video_id: str | None = None,
    limit: int = 50,
    tile_size: int = 512,
    overlap: int = 64,
    max_tiles_per_frame: int = 64,
    max_total_tiles: int = 500,
) -> dict:
    """Package highest-ranked frames without changing their source files.

    The full image is the single review target. Detail tiles are linked views,
    not separate examples or votes. The function refuses to overwrite a
    non-empty output directory and refuses to silently omit tiles.
    """
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    if not frames_dir.is_dir():
        raise ValueError(f"frames directory not found: {frames_dir}")
    if limit < 0:
        raise ValueError("limit must be >= 0 (0 means all frames)")
    if max_tiles_per_frame <= 0:
        raise ValueError("max_tiles_per_frame must be positive")
    if max_total_tiles <= 0:
        raise ValueError("max_total_tiles must be positive")
    # Validate geometry options before creating output artifacts.
    tile_regions(1, 1, tile_size, overlap)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {out_dir}")

    candidates = triage_frames(frames_dir, video_id=video_id)
    if limit > 0:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError(f"no image frames found in {frames_dir}")

    dimensions = {}
    total_tiles = 0
    for row in candidates:
        with Image.open(row["path"]) as image:
            dimensions[row["frame_id"]] = image.size
        width, height = dimensions[row["frame_id"]]
        needed = len(tile_regions(width, height, tile_size, overlap)) if (width > tile_size or height > tile_size) else 0
        if needed > max_tiles_per_frame:
            raise ValueError(
                f"frame {row['frame_id']} needs {needed} tiles; "
                f"limit is {max_tiles_per_frame}. Increase tile size or limit."
            )
        total_tiles += needed
    if total_tiles > max_total_tiles:
        raise ValueError(
            f"selected frames need {total_tiles} detail tiles; package limit is "
            f"{max_total_tiles}. Reduce --limit, increase tile size, or raise the cap."
        )

    source_hashes = {row["frame_id"]: _sha256(Path(row["path"])) for row in candidates}
    package_descriptor = {
        "schema_version": SCHEMA_VERSION,
        "tile_size": tile_size,
        "overlap": overlap,
        "frame_ids_and_hashes": [
            [row["frame_id"], source_hashes[row["frame_id"]]] for row in candidates
        ],
    }
    package_id = hashlib.sha256(
        json.dumps(package_descriptor, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]

    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(candidates):
        source = Path(row["path"])
        source_hash = source_hashes[row["frame_id"]]
        asset_dir = Path("frames") / f"frame_{index:06d}_{source_hash[:10]}"
        full_rel = asset_dir / f"full{source.suffix.lower()}"
        full_path = out_dir / full_rel
        full_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, full_path)

        tiles = []
        with Image.open(source) as image:
            width, height = image.size
            regions = tile_regions(width, height, tile_size, overlap)
            if width > tile_size or height > tile_size:
                for region in regions:
                    tile_id = f"{row['frame_id']}__r{region['row']:03d}_c{region['column']:03d}"
                    tile_rel = asset_dir / "tiles" / f"tile_r{region['row']:03d}_c{region['column']:03d}.jpg"
                    tile_path = out_dir / tile_rel
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    box = (
                        region["x"], region["y"],
                        region["x"] + region["width"],
                        region["y"] + region["height"],
                    )
                    image.crop(box).convert("RGB").save(tile_path, format="JPEG", quality=95)
                    tiles.append({
                        "tile_id": tile_id,
                        "frame_id": row["frame_id"],
                        **region,
                        "path": tile_rel.as_posix(),
                        "sha256": _sha256(tile_path),
                    })

        triage_signals = {
            key: value for key, value in row.items()
            if key not in {"path", "review_decision", "review_notes"}
        }
        records.append({
            "schema_version": SCHEMA_VERSION,
            "package_id": package_id,
            "frame_id": row["frame_id"],
            "review_unit": "one_source_frame",
            "video_id": row["video_id"],
            "timestamp_seconds": None,
            "source_path": str(source),
            "source_sha256": source_hash,
            "source_width": width,
            "source_height": height,
            "full_image": {
                "path": full_rel.as_posix(),
                "sha256": _sha256(full_path),
            },
            "tile_config": {
                "tile_size": tile_size,
                "overlap": overlap,
                "max_tiles_per_frame": max_tiles_per_frame,
                "max_total_tiles": max_total_tiles,
            },
            "tiles_are_detail_views_not_votes": True,
            "tiles": tiles,
            "triage_signals": triage_signals,
        })

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "stage": "review_pack",
        "package_id": package_id,
        "frames": len(records),
        "tiles": sum(len(record["tiles"]) for record in records),
        "manifest": str(manifest_path),
        "tile_size": tile_size,
        "overlap": overlap,
    }
