"""drawyt command line: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .coreset import assign_clusters, kcenter_greedy, medoid_index
from .discover import search_query
from .download import download_video, file_sha256
from .extract import extract_frames
from .features import frame_embedding
from .ids import parse_video_id
from .judges import BrightRegionJudge
from .license_check import check_license
from .manifest import FrameRecord, VideoRecord, write_frames_csv, write_receipt, write_videos_jsonl
from .prune import IMAGE_SUFFIXES, prune_frames
from .review_pack import build_review_pack
from .triage import triage_frames


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cmd_search(args) -> int:
    queries = [q.strip() for q in Path(args.queries).read_text(encoding="utf-8").splitlines() if q.strip()]
    by_id: dict[str, dict] = {}
    for query in queries:
        for hit in search_query(query, limit=args.limit, cc_only=args.cc_only):
            if hit["video_id"] in by_id:
                by_id[hit["video_id"]]["queries"] += f";{query}"
            else:
                by_id[hit["video_id"]] = hit
    rows = sorted(by_id.values(), key=lambda r: -r["view_count"])
    _write_csv(rows, Path(args.out))
    print(f"search: {len(rows)} unique candidates -> {args.out}")
    return 0


def cmd_check(args) -> int:
    rows = _read_csv(Path(args.input))
    out_rows = []
    for row in rows:
        info = check_license(row["video_id"])
        merged = {**row, **{k: v for k, v in info.items() if v not in ("", 0)}}
        merged["cc_by_confirmed"] = "sim" if info["cc_by_confirmed"] else "nao"
        merged["license_field"] = info["license"]
        out_rows.append(merged)
        print(f"check: {row['video_id']} cc_by={merged['cc_by_confirmed']}")
    _write_csv(out_rows, Path(args.out))
    confirmed = sum(1 for r in out_rows if r["cc_by_confirmed"] == "sim")
    print(f"check: {confirmed}/{len(out_rows)} CC BY confirmed -> {args.out}")
    return 0


def cmd_download(args) -> int:
    rows = _read_csv(Path(args.input))
    out_dir = Path(args.out)
    records: list[VideoRecord] = []
    failures = 0
    for row in rows:
        if args.cc_only and row.get("cc_by_confirmed") != "sim":
            continue
        try:
            path, route_used = download_video(row["video_id"], out_dir)
        except Exception as exc:  # one attempt per route; report, do not loop
            print(f"download: FAILED {row['video_id']}: {exc}", file=sys.stderr)
            failures += 1
            continue
        records.append(
            VideoRecord(
                video_id=row["video_id"],
                url=row.get("url", ""),
                title=row.get("title", ""),
                channel=row.get("channel", ""),
                duration_s=int(row.get("duration_s") or 0),
                license=row.get("license_field", ""),
                cc_by_confirmed=row.get("cc_by_confirmed") == "sim",
                sha256=file_sha256(path),
                bytes=path.stat().st_size,
                download_route=route_used,
            )
        )
        print(f"download: ok {row['video_id']} ({path.stat().st_size} bytes)")
    write_videos_jsonl(records, out_dir / "manifest.jsonl")
    write_receipt(
        {"stage": "download", "ok": len(records), "failed": failures},
        out_dir / "receipt_download.json",
    )
    print(f"download: {len(records)} ok, {failures} failed -> {out_dir}")
    return 0 if records or not failures else 1


def cmd_extract(args) -> int:
    frames = extract_frames(Path(args.video), Path(args.out), fps=args.fps)
    print(f"extract: {len(frames)} frames -> {args.out}")
    return 0


def cmd_prune(args) -> int:
    frames_dir = Path(args.frames)
    video_id = args.video_id or frames_dir.name
    judge = BrightRegionJudge() if args.judge else None
    records, receipt = prune_frames(frames_dir, video_id, judge=judge)
    out = Path(args.out)
    write_frames_csv(records, out / "frames_manifest.csv")
    write_receipt({"stage": "prune_level0", **receipt}, out / "receipt_prune.json")
    print(f"prune: {receipt['kept']}/{receipt['input_frames']} kept, dropped={receipt['dropped']} -> {out}")
    return 0


def cmd_triage(args) -> int:
    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        print(f"triage: frames directory not found: {frames_dir}", file=sys.stderr)
        return 1
    rows = triage_frames(frames_dir, video_id=args.video_id)
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        print(f"triage: no image frames found in {frames_dir}", file=sys.stderr)
        return 1
    _write_csv(rows, Path(args.out))
    print(f"triage: {len(rows)} review candidates -> {args.out} (scores are suggestions; fill review_decision)")
    return 0


def cmd_review_pack(args) -> int:
    try:
        receipt = build_review_pack(
            Path(args.frames),
            Path(args.out),
            video_id=args.video_id,
            limit=args.limit,
            tile_size=args.tile_size,
            overlap=args.overlap,
            max_tiles_per_frame=args.max_tiles_per_frame,
            max_total_tiles=args.max_total_tiles,
        )
    except (OSError, ValueError) as exc:
        print(f"review-pack: {exc}", file=sys.stderr)
        return 1
    print(
        f"review-pack: {receipt['frames']} frames, {receipt['tiles']} detail tiles "
        f"-> {receipt['manifest']} (local package; tiles are not extra votes)"
    )
    return 0


def _run_coreset(paths: list[Path], video_id: str, budget: int, out: Path) -> dict:
    embeddings = []
    for path in paths:
        with Image.open(path) as image:
            embeddings.append(frame_embedding(image))
    matrix = np.stack(embeddings)
    seed = medoid_index(matrix)
    chosen = kcenter_greedy(matrix, budget, seed_index=seed)
    clusters = assign_clusters(matrix, chosen)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    records: list[FrameRecord] = []
    for idx in chosen:
        path = paths[idx]
        dest = out / "frames" / path.name
        shutil.copy2(path, dest)
        records.append(
            FrameRecord(
                frame_id=f"{video_id}_{idx:06d}",
                video_id=video_id,
                path=str(dest),
                cluster_id=int(clusters[idx]),
                selection_reason="kcenter_cover",
            )
        )
    write_frames_csv(records, out / "manifest_frames.csv")
    receipt = {"stage": "coreset_kcenter", "input_frames": len(paths), "budget": budget, "selected": len(records)}
    write_receipt(receipt, out / "receipt_coreset.json")
    return receipt


def cmd_coreset(args) -> int:
    frames_dir = Path(args.frames)
    video_id = args.video_id or frames_dir.name
    if args.from_manifest:
        rows = _read_csv(Path(args.from_manifest))
        paths = [Path(r["path"]) for r in rows if r.get("kept_level0") in ("True", "true", "1")]
        if not paths:
            print("coreset: prune manifest has no kept frames", file=sys.stderr)
            return 1
    else:
        paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            print("coreset: no frames found", file=sys.stderr)
            return 1
    receipt = _run_coreset(paths, video_id, args.budget, Path(args.out))
    print(f"coreset: {receipt['selected']}/{receipt['input_frames']} selected -> {args.out}")
    return 0


def cmd_fetch(args) -> int:
    """One-shot personal pipeline: URL/ID -> video -> frames -> pruned manifest."""
    video_ref = parse_video_id(args.url)
    root = Path(args.out)
    path, route_used = download_video(video_ref, root / "videos")
    frames = extract_frames(path, root / "frames" / video_ref, fps=args.fps)
    judge = BrightRegionJudge() if args.judge else None
    records, receipt = prune_frames(root / "frames" / video_ref, video_ref, judge=judge)
    write_frames_csv(records, root / "work" / video_ref / "frames_manifest.csv")
    combined = {
        "stage": "fetch",
        "video_id": video_ref,
        "route": route_used,
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "frames_extracted": len(frames),
        "prune": receipt,
    }
    if args.budget:
        kept_paths = [Path(r.path) for r in records if r.kept_level0]
        combined["coreset"] = _run_coreset(kept_paths, video_ref, args.budget, root / "coreset")
    write_receipt(combined, root / "receipt_fetch.json")
    print(
        f"fetch: {video_ref} via {route_used}: {combined['bytes']} bytes, "
        f"{len(frames)} frames, {receipt['kept']} kept after pruning -> {root}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drawyt", description="YouTube -> coreset pipeline for Draw.")
    parser.add_argument("--version", action="version", version=f"drawyt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="discover candidates via yt-dlp")
    p.add_argument("--queries", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--cc-only", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("check", help="verify license per video")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("download", help="download verified videos (720p, no audio)")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--cc-only", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("fetch", help="one-shot: download a video and prune its frames")
    p.add_argument("url", help="YouTube URL or bare video ID")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=0, help="if >0, also build a coreset of this size from the kept frames")
    p.add_argument("--judge", action="store_true", help="score kept frames with the baseline sheet/ink judge")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("extract", help="extract frames from a local video (ffmpeg)")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=1.0)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("prune", help="level-0 pruning over a frames folder")
    p.add_argument("frames")
    p.add_argument("--out", required=True)
    p.add_argument("--video-id")
    p.add_argument("--judge", action="store_true", help="score kept frames with the baseline sheet/ink judge")
    p.set_defaults(fn=cmd_prune)

    p = sub.add_parser("triage", help="rank frames for human review; never drops frames")
    p.add_argument("frames", help="directory containing extracted image frames")
    p.add_argument("--out", required=True, help="review CSV; record decisions in review_decision")
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int, default=0, help="maximum CSV rows after full ranking (0 means all)")
    p.set_defaults(fn=cmd_triage)

    p = sub.add_parser("review-pack", help="prepare local full-frame and detail-tile review assets")
    p.add_argument("frames", help="directory containing extracted image frames")
    p.add_argument("--out", required=True, help="new or empty review-package directory")
    p.add_argument("--video-id")
    p.add_argument("--limit", type=int, default=50, help="maximum top-ranked frames (0 means all)")
    p.add_argument("--tile-size", type=int, default=512, help="maximum tile edge in source pixels")
    p.add_argument("--overlap", type=int, default=64, help="tile overlap in source pixels")
    p.add_argument("--max-tiles-per-frame", type=int, default=64, help="fail instead of omitting tiles above this count")
    p.add_argument("--max-total-tiles", type=int, default=500, help="fail instead of creating an oversized review package")
    p.set_defaults(fn=cmd_review_pack)

    p = sub.add_parser("coreset", help="k-center coreset selection over a frames folder")
    p.add_argument("frames")
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--video-id")
    p.add_argument("--from-manifest", help="prune frames_manifest.csv: select only from frames kept at level 0")
    p.set_defaults(fn=cmd_coreset)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
