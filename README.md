# draw-yt

**Turn YouTube into a training dataset — then throw almost all of it away.**

`draw-yt` collects videos of people writing and drawing on real paper, extracts their frames, and prunes them down to a **coreset**: the smallest fraction of frames that teaches the most. It was built for [Draw](https://github.com/santleme/draw), a system that turns a webcam pointed at a desk into a graphics tablet — no tablet required — and whose main open problem is not the model, but **generalization**: a sheet detector that only works on one desk, one camera, and one light is not a success.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-orange)

---

## Why this exists

There is no public dataset of people writing and drawing on paper, on a desk, filmed from above, with the sheet annotated. We checked (Kaggle, HuggingFace, the literature): the field splits into *annotated sheets nobody is writing on* (SmartDoc, MIDV) and *real writing with no sheet annotation* (Kinetics-700 `writing`). Meanwhile, a sheet detector trained on a single setup collapses on other people's setups — measured on 567 YouTube clips from Kinetics, an earlier Draw detector fully rejected **64.9%** of them and was solid on only **3.2%**.

YouTube is where the missing data lives: thousands of desks, papers, notebooks, hands, cameras, and lights. But raw video is almost all redundancy — at 30 fps, a 10-minute video is 18,000 near-identical frames. Training on that is slow, expensive, and mostly teaches the model the same frame thousands of times.

So `draw-yt` does the opposite of hoarding: it **collects broadly and keeps ruthlessly little**.

## What it does

```
 search ──▶ license ──▶ download ──▶ extract ──▶ prune ──▶ judge ──▶ package
 queries     check       yt-dlp       ffmpeg      4 levels   labels    manifest
 (EN + PT)   per video                            (§ below)           + coreset
```

| Stage | What happens |
|---|---|
| **Discover** | Flat-playlist searches (EN + PT queries), optionally filtered to Creative Commons. Output: a candidates CSV. |
| **Verify** | License checked **per video** from YouTube's own `license` field — never assumed from the search filter. Framing classified from title, description, and thumbnail: single sheet, open notebook, desk without paper, screen, whiteboard… |
| **Download** | `yt-dlp`, capped at 720p, no audio (dead weight for vision), SHA-256 and byte size recorded per file. |
| **Extract** | Frames at 1 fps plus scene-change frames via `ffmpeg`. Every kept frame also keeps its `t+1s` pair — temporal pairs are what flicker evaluation and temporal training consume. |
| **Prune** | Four levels, cheapest first — see below. This is the heart of the project. |
| **Judge** | Pluggable domain judges score what survived: sheet presence, hand presence, ink density. Judges *rank and classify*; they never veto alone (see [Rules](#rules-that-are-not-negotiable)). |
| **Package** | A balanced coreset with a full manifest: every frame traceable to its video, timestamp, license, and the reason it was selected. |

### The pruning cascade

| Level | Name | Removes | Keeps / promotes |
|---|---|---|---|
| 0 | **Pixel** | Exact and near duplicates (perceptual hash), consecutive frames that barely changed, blurred frames (Laplacian variance), blown-out or black frames | One sharp representative of each moment |
| 1 | **Domain** | Endless repeats of "no sheet, no hand" beyond the hard-negative quota | Frames classified by the judges: sheet, notebook, hand, ink — and a deliberate quota of **hard negatives** (desks without paper, paper-like objects) |
| 2 | **Coreset** | Visual redundancy across the whole corpus: hundreds of near-identical white-desk frames collapse into a few | Cluster centers *and* border cases, selected with **k-center greedy** so the kept set covers the visual space with the fewest points |
| 3 | **Information** | Frames the current model already gets right with high confidence and has many neighbours like them | Frames where judges **disagree**, where the model is **uncertain** (active learning), and frames from rare setups |

**Target budget:** 2,000–5,000 unique frames (+ their temporal pairs) from 30+ distinct setups — roughly 13–34% of the frames extracted at 1 fps, and under 1% of the video at native frame rate.

## Quickstart

```bash
git clone https://github.com/santleme/draw-yt.git
cd draw-yt
python -m venv .venv && source .venv/bin/activate
pip install -e .

# external tools: ffmpeg on PATH, yt-dlp installed with the package
drawyt --help
```

### A first run (offline stages)

Pruning and coreset selection work on any folder of frames — no YouTube required:

```bash
# one-shot: download a video and prune its frames in a single command
drawyt fetch https://www.youtube.com/watch?v=VIDEO_ID --out runs/VIDEO_ID --judge --budget 500

# or stage by stage: extract frames from a local video
drawyt extract path/to/video.mp4 --out frames/VIDEO_ID --fps 1

# level 0 pruning + manifest
drawyt prune frames/VIDEO_ID --out work/VIDEO_ID

# rank the frames for human review (scores are suggestions, not labels)
drawyt triage frames/VIDEO_ID --out work/VIDEO_ID/review.csv --limit 200

# prepare full-resolution frames and overlapping detail tiles for review
drawyt review-pack frames/VIDEO_ID --out work/VIDEO_ID/review-pack --limit 20

# coreset selection across everything that survived
drawyt coreset work/ --budget 3000 --out coreset/
```

`review-pack` keeps each original frame as the review unit and creates detail tiles only for images larger than 512 px. Tiles overlap by 64 source pixels; the versioned `manifest.jsonl` records source/tile SHA-256 values and tile coordinates. A tile cap fails clearly instead of silently skipping image areas. To request three independent Codex opinions over the full frames and their tiles, invoke `$drawyt-multiscale-review` with the generated package. The Python CLI prepares local assets; the Codex skill orchestrates reviewers.

`drawyt triage` writes one CSV row per frame with independent layer fields:
quality (`quality_*`), sheet (`sheet_*`), hand (`hand_*`), hand landmarks
(`hand_landmarks_*`), strokes/ink (`strokes_*` / `ink_*`), and scene
(`scene_*`). Current quality, sheet, and stroke values are heuristic review
signals. Hand presence, hand landmarks, and semantic scene classification are
marked `unavailable` until validated detectors are integrated. Unavailable
values stay empty; they are not treated as negative detections. No layer
automatically removes a frame.

### The YouTube stages

```bash
# 1. discover candidates (Creative Commons filter on)
drawyt search --queries examples/queries.txt --cc-only --out candidates.csv

# 2. verify license + framing, one video at a time
drawyt check candidates.csv --out verified.csv

# 3. download the verified set (720p max, no audio, hashes recorded)
drawyt download verified.csv --out videos/
```

> **Measured download behaviour (2026-11-09, [full receipt](docs/LIVE_TEST_2026-11-09.md)):** from a datacenter IP, YouTube bot-checks the default web client — but the android player client (with Node.js as JS runtime) downloads fine from the same machine. `draw-yt` therefore tries named routes once each, in quality order (default → android-node → tv/ios/web-embedded), never loops on a dead route, and records which route delivered each file. In the measured session the android route delivered 360p (SABR experiment); on a residential connection the default route delivers up to the 720p cap.

## Concepts, in plain words

- **Dataset Pruning** — removing redundant or low-quality examples from a large dataset without losing performance. Levels 0–3.
- **Coreset Selection** — a *coreset* is a small, weighted subset that approximates the full dataset: a model trained on it behaves almost as if trained on everything. Level 2, via k-center coverage.
- **Active Learning** — instead of labelling everything, let the model's doubt choose what gets labelled: disagreement and uncertainty are the most informative examples. Level 3.
- **Data-Centric AI** — the umbrella: improve the data instead of swapping the model. Draw's measured bottleneck is *translation* — turning abundant raw material into labelled, judgeable examples — not model capacity.

## Rules that are not negotiable

These come from hard lessons in the Draw project and are enforced by the pipeline's design, not by good intentions:

1. **Split by session, never by frame.** One video = one session = one setup. No video (and no single-setup channel) ever appears in both training and evaluation data.
2. **License is verified per video, never presumed.** A search filter is a hint; the video's own license field is the evidence.
3. **YouTube pixels are not redistributed.** CC BY material may travel with attribution (the manifest carries credit per video); standard-license material is kept as an index — IDs, timestamps, measurements — for internal use only.
4. **Who generates does not judge.** Models that produce pseudo-labels never decide alone what enters the final set: there is a classical veto, a random human-review quota (10–20%), and full review of disagreement cases.
5. **Sealed benchmarks never become training data.** Anything already consumed by an evaluation ruler is excluded by list, with the intersection *proven* to be zero by script.
6. **Measure before promising.** Every stage writes a receipt (in/out counts). Download routes, pruning rates, and budgets in this README are design targets until a run measures them; measured numbers replace targets.

## Output layout

```
<workdir>/
  candidates.csv        # discovery output
  verified.csv          # license + framing, per video
  manifest.jsonl        # one line per downloaded video (hash, license, split)
  frames/<video_id>/    # extracted frames and their t+1s pairs
  work/<video_id>/      # pruning manifest per video
  coreset/
    frames/             # the selected frames
    manifest_frames.csv # one line per frame: source, judges, cluster,
                        # selection reason, label kind (reviewed | pseudo)
    receipts/           # per-stage, per-run receipts
```

Every frame in the coreset can be traced back: `frame → video → channel → license → query that found it`.

## Project status

Active development. What is implemented and tested today:

- [x] Frame-level pruning (Level 0): perceptual-hash dedup, temporal near-duplicate removal, blur/exposure gates
- [x] Frame features: sharpness, luminance, colour-histogram embeddings (NumPy + Pillow only)
- [x] Coreset selection (Level 2): k-center greedy coverage with per-cluster quotas
- [x] YouTube discovery, per-video license verification, and download wrappers around `yt-dlp`
- [x] Live end-to-end proof: real videos downloaded and pruned on a bot-checked network ([receipt](docs/LIVE_TEST_2026-11-09.md)); `drawyt fetch` one-shot command
- [x] Frame extraction via `ffmpeg`, manifests and receipts for every stage
- [x] Domain judges (Level 1) wired behind the `Judge` interface — baseline bright-region judge scores kept frames today (`--judge`); Draw's neural sheet/ink models and hand landmarker plug into the same interface
- [x] Layered frame triage for human review: quality, sheet-like pixels, and stroke-like marks; hand, hand landmarks, and semantic scene layers are explicitly marked unavailable
- [x] Multiscale review package: full frames, overlapping detail tiles, source coordinates and hashes, plus a Codex-side three-reviewer workflow
- [ ] Neural judges (Draw's sheet/ink ONNX + hand landmarker) behind that interface
- [ ] Information pruning (Level 3): judge disagreement + uncertainty ranking
- [ ] First full run: pilot on 5 videos, then the 33-video CC BY batch already identified

The full design — architecture, data schema, class quotas, quality gates, measured anchors, and the phased plan — lives in the [spec](docs/SPEC.pt-BR.md) (Portuguese, as originally written for the Draw project).

## Development

```bash
pip install -e .
python -m unittest discover -s tests -v   # standard library only, no extra deps
```

Core dependencies are deliberately minimal: **NumPy + Pillow**. `yt-dlp` and `ffmpeg` are external tools invoked as subprocesses. Domain judges are optional plugins behind a small interface (`drawyt.judges.Judge`), so the pipeline runs end-to-end without any ML model, and Draw's ONNX models can be plugged in where they belong — as judges, not as gatekeepers.

## Relationship to Draw

`draw-yt` is the data front-end for [Draw](https://github.com/santleme/draw). Draw's production models — a neural sheet detector and a pixel-level ink segmenter — were trained on one person's desk. This project's job is to give them the rest of the world: diverse real setups for training, hard negatives for the presence problem, and a sealed **other-setup holdout** so "does it work on someone else's desk?" becomes a number instead of a worry.

## License

Code: [MIT](LICENSE).

Videos and frames are **not** covered by this license. Each video keeps the license declared by its author on YouTube; the manifest records it per video, and CC BY attribution travels with the data.
