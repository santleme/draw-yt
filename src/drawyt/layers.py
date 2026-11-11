"""Independent per-frame layer signals for human-review triage.

Only quality, paper-like pixels, and ink-like marks have baseline heuristics.
Hand landmarks and semantic scene labels remain explicitly unavailable until
validated detectors are integrated.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .features import frame_features


def analyze_layers(image: Image.Image) -> dict:
    """Return JSON-serializable, independent layer results for one frame."""
    rgb = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    low_chroma = (rgb.max(axis=2) - rgb.min(axis=2)) < 0.20
    paper = (gray > 0.72) & low_chroma
    dark = (gray < 0.35) & low_chroma
    paper_image = Image.fromarray(paper.astype(np.uint8) * 255)
    near_paper = np.asarray(paper_image.filter(ImageFilter.MaxFilter(7))) > 0
    quality = frame_features(image)
    ink_density = float((dark & near_paper).mean())
    return {
        "quality": {"status": "heuristic", **quality},
        "paper": {"status": "heuristic", "presence": float(paper.mean())},
        "hand": {"status": "unavailable", "presence": None, "reason": "no_hand_detector_integrated"},
        "hand_landmarks": {"status": "unavailable", "points": None, "reason": "no_hand_landmarker_integrated"},
        "strokes": {"status": "heuristic", "density": ink_density},
        "ink": {"status": "heuristic", "density": ink_density},
        "scene": {"status": "unavailable", "class": None, "confidence": None, "reason": "no_scene_classifier_integrated"},
    }
