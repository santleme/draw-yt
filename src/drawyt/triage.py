"""Rank extracted frames for human review without treating scores as labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .hashing import average_hash, hamming
from .layers import analyze_layers
from .prune import IMAGE_SUFFIXES


class _HammingIndex:
    """Find nearby 64-bit hashes without scanning every representative.

    For threshold t < 64, split each hash into t+1 disjoint blocks. Any pair
    at Hamming distance <= t must share at least one whole block, so this
    index can narrow candidates without changing exact duplicate decisions.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.block_count = min(max(threshold + 1, 1), 64)
        self.block_specs: list[tuple[int, int]] = []
        remaining = 64
        for index in range(self.block_count):
            width = remaining // (self.block_count - index)
            remaining -= width
            shift = remaining
            self.block_specs.append((shift, (1 << width) - 1))
        self.buckets: dict[tuple[int, int], list[int]] = {}
        self.values: list[int] = []

    def _candidates(self, value: int):
        if self.threshold < 0:
            return ()
        if self.threshold >= 64:
            return range(len(self.values))
        candidate_indexes: set[int] = set()
        for block_index, (shift, mask) in enumerate(self.block_specs):
            candidate_indexes.update(
                self.buckets.get((block_index, (value >> shift) & mask), ())
            )
        return candidate_indexes

    def has_nearby(self, value: int) -> bool:
        return any(
            hamming(value, self.values[index]) <= self.threshold
            for index in self._candidates(value)
        )

    def add(self, value: int) -> None:
        value_index = len(self.values)
        self.values.append(value)
        for block_index, (shift, mask) in enumerate(self.block_specs):
            key = (block_index, (value >> shift) & mask)
            self.buckets.setdefault(key, []).append(value_index)


def triage_frames(
    frames_dir: Path,
    *,
    video_id: str | None = None,
    hash_distance: int = 4,
) -> list[dict]:
    """Return ranked review candidates; scores are heuristic review signals.

    Near-identical frames are grouped against earlier files to keep a review
    queue from being dominated by the same moment. No frame is discarded.
    """
    paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows: list[dict] = []
    hash_index = _HammingIndex(hash_distance)
    sharpnesses: list[float] = []

    for index, path in enumerate(paths):
        with Image.open(path) as image:
            layers = analyze_layers(image)
            feats = layers["quality"]
            paper_ratio = layers["paper"]["presence"]
            ink_ratio = layers["ink"]["density"]
            ahash = average_hash(image)

        near_duplicate = hash_index.has_nearby(ahash)
        if not near_duplicate:
            hash_index.add(ahash)
        sharpnesses.append(feats["sharpness"])
        rows.append({
            "frame_id": f"{video_id or frames_dir.name}_{index:06d}",
            "video_id": video_id or frames_dir.name,
            "path": str(path),
            "sharpness": feats["sharpness"],
            "luma": feats["luma"],
            "paper_like_ratio": paper_ratio,
            "ink_like_ratio": ink_ratio,
            "ahash": f"{ahash:016x}",
            "near_duplicate": near_duplicate,
            "review_score": 0.0,
            "review_reason": "",
            "review_decision": "",
            "review_notes": "",
            # Flat, namespaced CSV fields keep layer evidence inspectable.
            "quality_status": layers["quality"]["status"],
            "quality_sharpness": layers["quality"]["sharpness"],
            "quality_luma": layers["quality"]["luma"],
            "sheet_status": layers["paper"]["status"],
            "sheet_presence": layers["paper"]["presence"],
            "hand_status": layers["hand"]["status"],
            "hand_presence": layers["hand"]["presence"],
            "hand_reason": layers["hand"]["reason"],
            "hand_landmarks_status": layers["hand_landmarks"]["status"],
            "hand_landmarks_points": layers["hand_landmarks"]["points"],
            "hand_landmarks_reason": layers["hand_landmarks"]["reason"],
            "strokes_status": layers["strokes"]["status"],
            "strokes_density": layers["strokes"]["density"],
            "ink_status": layers["ink"]["status"],
            "ink_density": layers["ink"]["density"],
            "scene_status": layers["scene"]["status"],
            "scene_class": layers["scene"]["class"],
            "scene_confidence": layers["scene"]["confidence"],
            "scene_reason": layers["scene"]["reason"],
        })

    if not rows:
        return rows

    sharp_values = np.asarray(sharpnesses, dtype=np.float64)
    low, high = np.quantile(sharp_values, [0.10, 0.90])
    sharp_span = max(float(high - low), 1e-12)
    for row in rows:
        sharp_quality = float(np.clip((row["sharpness"] - low) / sharp_span, 0.0, 1.0))
        paper_signal = min(row["paper_like_ratio"] / 0.35, 1.0)
        ink_signal = min(row["ink_like_ratio"] / 0.015, 1.0)
        # Prefer plausible paper/drawing, while retaining sharpness as a
        # quality signal. This is a queue ordering, not a classifier.
        score = 0.55 * paper_signal + 0.25 * ink_signal + 0.20 * sharp_quality
        if row["near_duplicate"]:
            score *= 0.25
        row["review_score"] = round(score, 4)
        reasons = []
        if row["paper_like_ratio"] >= 0.15:
            reasons.append("paper_like_pixels")
        if row["ink_like_ratio"] >= 0.003:
            reasons.append("dark_marks_on_bright_region")
        if sharp_quality >= 0.7:
            reasons.append("sharp_frame")
        if row["near_duplicate"]:
            reasons.append("near_duplicate_review_lower_priority")
        row["review_reason"] = ";".join(reasons) or "low_signal_sample"

    return sorted(rows, key=lambda row: (-row["review_score"], row["frame_id"]))
