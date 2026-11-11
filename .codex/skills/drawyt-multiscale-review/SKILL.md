---
name: drawyt-multiscale-review
description: Review a local draw-yt review package using three independent Codex vision reviewers, combining full-frame context with detail tiles.
---

# Draw-yt multiscale review

Use when the user asks to review a package produced by `drawyt review-pack` or explicitly invokes `$drawyt-multiscale-review`.

## Contract

- The **source frame** is the only review unit and receives one opinion from each reviewer.
- Reviewers inspect the full image first. Tiles are zoomed detail views of that same image, not separate samples or votes.
- Keep all three opinions, evidence, uncertainty, and disagreements. Agreement only prioritizes human review; it never creates a final training label, mask, or deletion decision.
- Use only the package assets supplied for the review. Do not add a separate API/upload integration; if the configured Codex review surface indicates that image data will be transferred or needs extra authorization, stop and surface that boundary.

## Procedure

1. Read `manifest.jsonl`; verify its schema, `frame_id`s, and referenced full/tile paths exist. Reject malformed entries or paths escaping the package directory.
2. For each frame, prepare its full image path and listed tile paths. Require each reviewer to inspect the full image first, then use the tiles for details. Use tile pixel coordinates to map any observation back to the source frame. Treat unavailable heuristic layers as unknown, not as absent.
3. Launch three native vision reviewers in parallel. Give each the same frame IDs, full images, tiles, current layer signals, and taxonomy. Do not reveal one reviewer’s output to another.
   - Reviewer A: focus on whether the surface is paper, notebook, or a hard negative.
   - Reviewer B: focus on visible hand, drawing/writing marks, and whether marks belong to the paper.
   - Reviewer C: adversarial check for confounds such as printed text, bark/texture, shadows, screen, wall, or misleading crops.
   All three must still assess the same frame-level schema and may answer `uncertain`.
4. Collect one structured response per reviewer and frame:

   ```json
   {"frame_id":"...","reviewer_id":"A","paper":"present|absent|uncertain","hand":"present|absent|uncertain","strokes":"present|absent|uncertain","confidence":0.0,"evidence":[{"tile_id":"...","xyxy":[0,0,0,0],"note":"..."}],"uncertainty":"..."}
   ```

   Evidence can point to the full image (`tile_id: null`) or a tile. Coordinates are source-image pixels. Do not infer precision beyond what is visible.
5. Aggregate per source frame and layer:
   - Three matching non-uncertain answers: `agreement_candidate`.
   - Any disagreement, uncertainty, missing opinion, or malformed response: `human_review_required`.
   Preserve the original answers; never replace them with the aggregate.
6. Save `agent_opinions.jsonl` and `review_summary.jsonl` under a new `reviews/` directory inside the package. Include `package_id`, frame ID, reviewer ID, and the original opinion. Do not overwrite prior review results; create a uniquely named run directory.
7. Report counts of agreement candidates and frames needing human review, plus any incomplete or unreadable assets. Do not claim accuracy from reviewer agreement alone.

## Failure handling

- If a full frame or tile is missing, skip its opinion and mark the frame for human review.
- If package size is too large for one review session, split the work into smaller user-approved batches; never silently omit frames or tiles.
- If the native agent surface is unavailable, stop after validating the package and explain that three opinions were not produced. Do not replace native reviewers with remote API calls.
