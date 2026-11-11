# Layered triage: test and improvement report

**Branch:** `feature/layered-frame-analysis`  
**Date:** 2026-11-10

## Verification performed

- `PYTHONPATH=src python3 -m unittest discover -s tests -v` — **32 tests passed** after the multiscale package was added.
- Stress battery: `PYTHONPATH=src python3 tests/test_stress_review_pack.py` — **100 tests passed in 10.177 s**: 80 deterministic geometry combinations and 20 checks against a 1792×1792 single-frame package at the configured maximum of 64 tiles.
- Full discovery, including the stress battery: **132 tests passed in 11.182 s**.
- Automated CLI regression invokes `python -m drawyt triage` on synthetic frames, parses the generated CSV, checks required headers, row count, numeric fields, explicit unavailable statuses with blank values, deterministic ranking, and `--limit` output behavior.
- Synthetic image check confirmed a positive paper-like signal for a white sheet, a positive stroke proxy for dark marks on that sheet, and no ink signal on a blank sheet or dark textured background.
- Real clip check on frames at 3, 5, and 12 seconds from `sVTAtgzPGhA`: the visually inspected sample shows a hand drawing a tree on white paper. All three frames had positive sheet-like and stroke signals. Hand presence, hand landmarks, and scene classification correctly remained unavailable.
- `git diff --check` passed.
- Exact Hamming-index benchmark on 3,000 seeded random 64-bit hashes: **2.4850 s** for the prior sequential scan versus **0.0222 s** for the indexed scan (**112.1× faster** in this synthetic workload); every duplicate decision matched.
- `drawyt review-pack` on three frames from the real 15-second sample created three original assets and 18 overlapping tiles with source-coordinate/hash links. Three blind native vision reviewers then inspected the same target frame (full image first, all six tiles): all marked paper, hand, and strokes present, with confidences 0.94, 0.98, and 0.96. This is a workflow smoke check, not an accuracy estimate; no human-labeled ground truth was used.

## Current behavior and limits

Quality, sheet-like pixels, and dark marks near those pixels are heuristic signals used to rank frames for human review. They do not create confirmed labels, segmentation masks, or keypoints. The `strokes_density` and `ink_density` fields currently expose the same pixel proxy.

Hand detection, hand landmarks, and semantic scene classification are not integrated. Their fields are explicitly marked unavailable. Triage does not drop frames.

## Corrections and improvements

| Priority | Finding | Recommended improvement |
|---|---|---|
| Medium | `paper.presence` is the fraction of bright, low-chroma pixels, not a sheet segmentation. Pale walls, blank surfaces, and notebook patterns can score as paper. | Keep the heuristic label explicit; validate against varied negative scenes, then add spatial coherence and boundary evidence or use the Draw sheet model. |
| Medium | The stroke/ink proxy counts dark neutral pixels near the paper-like mask. It can include printed text, shadows, hand edges, or nearby texture. | Treat these as one weak signal until annotated examples establish precision and recall; integrate a dedicated ink model for actual ink masks. |
| Done | CSV serialization lacked automated coverage. | `tests/test_cli_triage.py` now exercises the CLI and validates the CSV contract, unavailable values, stable order, and row limit. |
| Improved | Near-duplicate detection used a sequential scan of all representative hashes. | `_HammingIndex` narrows candidates through `threshold+1` disjoint bit blocks, then applies exact Hamming distance. The seeded equivalence test and benchmark matched old decisions. The benchmark is synthetic; real-corpus timing remains unknown. |
| Done | `--limit` could be mistaken for a compute limit. | CLI help now says it caps CSV rows after full ranking. |
| Low | Machine-readable layer status uses `heuristic` as the status for available signals. | Consider separating stable availability (`available`/`unavailable`) from method (`heuristic`/model identifier) before downstream consumers depend on this schema. |

## Three-reviewer workflow architecture

Keep image processing local and deterministic. `drawyt review-pack` emits a versioned package containing stable frame IDs, source provenance, image SHA-256, existing layer signals, a full-resolution image, and linked overlapping detail tiles. A Codex-side skill—not the Python CLI—submits the same target frame and its tiles independently to three reviewers, then stores each structured opinion with reviewer identity, per-layer `present|absent|uncertain`, confidence, evidence, and uncertainty.

Aggregation preserves all three opinions. Unanimity may prioritize a frame for faster human review; disagreement, uncertainty, or abstention goes to a human. Votes must not automatically create training labels, segmentation masks, or frame deletions. Three reviewers can share visual errors, so the process needs evaluation against a representative human-labeled set before its agreement rate is trusted. The Python CLI has no agent-spawning runtime; the project-local `$drawyt-multiscale-review` skill uses the supported native Codex reviewer surface instead. If the configured review surface indicates image transfer or requires additional authorization, the skill must stop and surface that boundary.

The review package defaults to 512-pixel tile edges with 64-pixel overlap, at most 64 tiles per frame and 500 tiles total. It fails before writing output when a limit would be exceeded, instead of silently dropping image regions.

## Review result

The independent code review found no critical or high-severity issues. The branch now has automated CSV and multiscale-package regressions. Current visual signals remain heuristics; the next meaningful quality gate is a human-labeled sample with hard negatives and diverse paper scenes. Three-reviewer agreement also needs comparison against that sample before it is treated as useful evidence.
