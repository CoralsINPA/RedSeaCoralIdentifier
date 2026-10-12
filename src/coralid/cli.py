"""Command line: `python -m coralid <command> --config <file>`.

Every command is a thin wrapper over a library function. All read the run's config (for example
`configs/iou60_sta55_pts32.yaml`) and the data root (`--data-root`, `paths.data_root` or
`$CORALID_DATA_ROOT`). None of them overwrites a file: a result that exists is loaded or left
alone, and the long jobs resume where they stopped.

- `segment`: reads the photos and the CoralSCOP checkpoint (a GPU); creates `json/<stem>.json`
  for the photos that have none, and a log in `segmentation/`.
- `build-manifest`: reads the mask JSONs and photos; creates `manifest/manifest.parquet` (and its
  `.meta.json`), with progress in `manifest/parts/`.
- `embed`: reads the manifest, JSONs and photos (a GPU); creates
  `features/<model>__masked_square.npz`.
- `zero-shot`: reads the manifest, JSONs and photos (a GPU); creates the baseline's `predictions/`
  and `features/` files.
- `evaluate`: reads the baseline, the manifest and the expert set; creates a folder in `reports/`.
- `export-review`: reads the baseline and the manifest; creates `reviews/exports/` (to hand out)
  and `reviews/keys/` (private).
- `train-probe`: reads a features file and a labels table; creates a folder in `reports/`.

Exit code: 0 when done, 2 when the command could not finish (the message says why; for the
resumable jobs, run the same command again).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from coralid import __version__
from coralid.baseline import BaselineError, BaselineIncomplete, run_baseline
from coralid.config import Config, ConfigError, load_config
from coralid.embed import EmbedError, EmbedIncomplete, run_embeddings
from coralid.jobs import (
    JobError,
    ManifestIncomplete,
    build_manifest,
    evaluate_baseline,
    export_blind_review,
    train_probe,
)
from coralid.probe import ProbeError
from coralid.segmentation import (
    SegmentationError,
    list_images,
    plan_segmentation,
    run_segmentation,
)

_INCOMPLETE = (BaselineIncomplete, EmbedIncomplete, ManifestIncomplete)
_STOPPED = (
    BaselineError,
    EmbedError,
    SegmentationError,
    JobError,
    ProbeError,
    ConfigError,
    FileNotFoundError,
    FileExistsError,
)


def _command(
    commands: argparse._SubParsersAction, name: str, help_text: str, description: str
) -> argparse.ArgumentParser:
    """Add a subcommand with the options every command has."""
    parser = commands.add_parser(name, help=help_text, description=description)
    parser.add_argument(
        "--config", required=True, help="a run config such as configs/iou60_sta55_pts32.yaml"
    )
    parser.add_argument(
        "--data-root", help="the data folder (default: paths.data_root or $CORALID_DATA_ROOT)"
    )
    parser.add_argument("--quiet", action="store_true", help="no progress bar")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of `python -m coralid`."""
    parser = argparse.ArgumentParser(prog="coralid", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"coralid {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    seg = _command(
        commands,
        "segment",
        "create the missing mask JSONs with CoralSCOP (never touches an existing JSON)",
        "Segment every photo that has no JSON yet and write json/<stem>.json for it. A photo "
        "whose JSON exists is skipped and the file is never rewritten, so this is safe on the "
        "active run (to redo a photo, move its JSON aside first). Errors go to "
        "segmentation/segment.failures.jsonl (no JSON is written, the next run retries); a photo "
        "with no masks gets an empty JSON with the note 'no masks'. "
        "Needs the vendored CoralSCOP (git submodule update --init), its checkpoint "
        "(sam.checkpoint) and a GPU.",
    )
    seg.add_argument("--limit", type=int, help="segment at most this many photos in this run")
    seg.add_argument("--images", nargs="+", metavar="NAME", help="only these photos")
    seg.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    seg.add_argument(
        "--save-features",
        action="store_true",
        help="also keep CoralSCOP's cate_preds and fc_features in segmentation/extras/",
    )
    seg.add_argument(
        "--dry-run", action="store_true", help="only say how many photos need a JSON; load nothing"
    )

    man = _command(
        commands,
        "build-manifest",
        "build the run's manifest of masks, filter decisions and sharpness",
        "Build manifest/manifest.parquet: one row per mask with the area, aspect, containment, "
        "visibility and sharpness decisions and their reasons. Resumable (manifest/parts/), "
        "written only when every photo is done, never over an existing manifest (it is loaded "
        "if its filter and crop settings match the config, else this stops).",
    )
    man.add_argument(
        "--geometric",
        action="store_true",
        help="no photos: area, aspect and containment only, to manifest/manifest_geometric.parquet",
    )

    emb = _command(
        commands,
        "embed",
        "DINO embeddings of the masked square crop of every kept mask",
        "Embed every keep == True row of the manifest with DINOv2 ViT-S/14 or DinoVdeau-large and "
        "write features/<model>__masked_square.npz (uids and raw float32 features) with a "
        ".meta.json. Resumable (features/parts/), written only when every photo is done, never "
        "over an existing file. CLIP-family features come from zero-shot.",
    )
    emb.add_argument("--model", help="dinov2-s14 or dinovdeau-l (default: embed.model)")
    emb.add_argument("--batch-size", type=int, help="override the model's own batch size")

    _command(
        commands,
        "zero-shot",
        "classify every kept crop of the run's manifest and save predictions and embeddings",
        "Classify every kept crop of the run's manifest with the configured model, prompts "
        "and label set. Writes predictions/<model>__<prompts>__<labels>__all.parquet and "
        "features/<model>__<crop mode>.npz (each with a .meta.json) under the run folder. "
        "Resumable: finished images are kept in predictions/parts/ and a failed image is "
        "retried on the next run. Never overwrites: if the results exist it loads them.",
    )

    ev = _command(
        commands,
        "evaluate",
        "score the saved baseline against the expert set and the survey counts",
        "Evaluate predictions/<model>__...__all.parquet against the expert set (all rows, "
        "landscape rows, portrait rows) and the ground-truth genus counts, and write the report "
        "as a new folder under reports/.",
    )
    ev.add_argument(
        "--expert-csv", help="the expert set (default: <run>/assaf_expert_test_set.csv)"
    )

    rev = _command(
        commands,
        "export-review",
        "write a blind review sheet of one genus (hand-out files and a hidden key)",
        "Sample crops the baseline calls the genus plus random others, with no model output "
        "shown, and write the sheets and PDFs to reviews/exports/ (hand these out) and the answer "
        "key to reviews/keys/ (keep it private).",
    )
    rev.add_argument("--genus", default="Millepora", help="the genus to review (default Millepora)")
    rev.add_argument("--audit-csv", help="an old audit to reuse (default for Millepora: V2's)")
    rev.add_argument("--n-pred", type=int, default=150, help="crops the baseline calls the genus")
    rev.add_argument("--n-other", type=int, default=150, help="crops it called something else")
    rev.add_argument("--n-assaf", type=int, default=40, help="crops on the expert's sheet")
    rev.add_argument("--seed", type=int, default=0)
    rev.add_argument("--no-pdf", action="store_true", help="write the CSV sheets only")

    pr = _command(
        commands,
        "train-probe",
        "train a linear probe on stored embeddings and report held-out accuracy",
        "Logistic regression on features/<probe.features>.npz, trained on a labels table you "
        "give (image_name, coral_id and a genus column), cross-validated by photo. Writes a new "
        "folder under reports/ with held-out accuracy, per-genus precision and recall and the "
        "confusion matrix. It never makes labels and never reports training accuracy.",
    )
    pr.add_argument("--labels", help="the labels table, CSV or parquet (default: probe.labels)")
    pr.add_argument("--label-column", help="the genus column of the labels (default: genus)")
    pr.add_argument("--features", help="features file stem in <run>/features/")
    pr.add_argument("--n-splits", type=int, help="cross-validation folds (default 5)")
    pr.add_argument("--C", type=float, help="inverse regularization strength (default 1.0)")
    pr.add_argument("--class-weight", choices=["balanced", "none"], help="default balanced")
    pr.add_argument("--no-standardize", action="store_true", help="skip feature scaling")
    pr.add_argument("--no-zero-shot", action="store_true", help="skip the zero-shot comparison")
    pr.add_argument("--name", help="report name (default: the features stem)")
    return parser


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.data_root:
        cfg.paths.data_root = args.data_root
    return cfg


# --------------------------------------------------------------------------- the commands


def _segment(args: argparse.Namespace) -> int:
    cfg = _config(args)
    if args.save_features:
        cfg.sam.save_features = True
    if args.dry_run:
        images = list_images(cfg.images_dir)
        plan = plan_segmentation(images, cfg.json_dir)
        print(
            f"{len(images)} photos: {len(plan.done)} have a JSON (never rewritten), "
            f"{len(plan.todo)} to segment"
        )
        return 0
    result = run_segmentation(
        cfg,
        image_names=args.images,
        limit=args.limit,
        device=args.device,
        progress=not args.quiet,
    )
    print(
        f"{result.n_images} photos: {result.n_already_done} already had a JSON (untouched), "
        f"{len(result.written)} written ({result.n_empty} with no masks), "
        f"{result.n_remaining} still without one"
    )
    print(f"  mask JSONs: {cfg.json_dir}")
    if result.failures:
        shown = "; ".join(f"{f['image']}: {f['error']}" for f in result.failures[:5])
        print(f"INCOMPLETE: {len(result.failures)} photo(s) failed ({shown})", file=sys.stderr)
        return 2
    return 0


def _build_manifest(args: argparse.Namespace) -> int:
    result = build_manifest(_config(args), geometric=args.geometric, progress=not args.quiet)
    table = result.manifest
    verb = "Loaded the saved manifest" if result.status == "already_done" else "Wrote"
    kept = f", {int(table['keep'].sum())} kept" if "keep" in table.columns else ""
    print(f"{verb}: {len(table)} masks on {table['image_name'].nunique()} photos{kept}")
    column = "reject_reason" if "keep" in table.columns else "stage1_reason"
    reasons = table[column].fillna("kept").value_counts()
    print("  " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
    print(f"  manifest: {result.path}")
    return 0


def _embed(args: argparse.Namespace) -> int:
    cfg = _config(args)
    if args.batch_size:
        cfg.embed.batch_size = args.batch_size
    result = run_embeddings(cfg, args.model, progress=not args.quiet)
    verb = "Loaded the saved features" if result.status == "already_done" else "Wrote"
    print(f"{verb}: {len(result.uids)} crops, {result.features.shape[1]} dimensions")
    print(f"  features: {result.paths.features}")
    return 0


def _zero_shot(args: argparse.Namespace) -> int:
    result = run_baseline(_config(args), progress=not args.quiet)
    verb = "Loaded the saved baseline" if result.status == "already_done" else "Wrote"
    print(f"{verb}: {len(result.predictions)} crops")
    print(f"  predictions: {result.paths.predictions}")
    print(f"  features:    {result.paths.features}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    report, folder = evaluate_baseline(_config(args), args.expert_csv)
    print("Expert set, the saved baseline, in the label set (Wilson 95% interval):")
    for group, evaluation in report.expert_new.items():
        acc = evaluation.accuracy_in_set
        low, high = acc["ci95"] or (float("nan"), float("nan"))
        share = acc["accuracy"] if acc["accuracy"] is not None else float("nan")
        print(
            f"  {group:9} correct {acc['correct']}/{acc['total']} = {share:.3f} "
            f"({low:.3f}-{high:.3f}) | {evaluation.status_counts}"
        )
    print(f"  report: {folder}")
    return 0


def _export_review(args: argparse.Namespace) -> int:
    cfg = _config(args)
    review, files = export_blind_review(
        cfg,
        args.genus,
        args.audit_csv,
        n_pred=args.n_pred,
        n_other=args.n_other,
        n_assaf=args.n_assaf,
        seed=args.seed,
        pdf=not args.no_pdf,
    )
    print(f"Wrote a blind review of {args.genus}: {review.meta['n_on_sheet']} crops")
    for what, path in files.items():
        print(f"  {what:12} {path}")
    print(f"Hand out {cfg.reviews_dir / 'exports'}. KEEP {cfg.reviews_dir / 'keys'} PRIVATE.")
    return 0


def _train_probe(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = cfg.probe
    p.label_column = args.label_column or p.label_column
    p.features = args.features or p.features
    p.n_splits = args.n_splits or p.n_splits
    p.C = args.C if args.C is not None else p.C
    if args.class_weight:
        p.class_weight = None if args.class_weight == "none" else args.class_weight
    p.standardize = p.standardize and not args.no_standardize
    report, data, folder = train_probe(
        cfg, args.labels, name=args.name, compare_zero_shot=not args.no_zero_shot
    )
    acc = report.accuracy
    low, high = acc["ci95"] or (float("nan"), float("nan"))
    print(
        f"Held-out accuracy {acc['correct']}/{acc['total']} = {acc['accuracy']:.3f} "
        f"({low:.3f}-{high:.3f}) on {report.n_labeled} labeled crops from {report.n_images} "
        f"photos; always answering the commonest genus gives {report.majority_baseline:.3f}"
    )
    if report.zero_shot and report.zero_shot["total"]:
        shot = report.zero_shot
        done = f"{shot['correct']}/{shot['total']} = {shot['accuracy']:.3f}"
        print(f"  zero-shot on the same crops: {done}")
    for note in report.warnings:
        print(f"  note: {note}")
    print(f"  labels left out: {data.dropped}")
    print(f"  report: {folder}")
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "segment": _segment,
    "build-manifest": _build_manifest,
    "embed": _embed,
    "zero-shot": _zero_shot,
    "evaluate": _evaluate,
    "export-review": _export_review,
    "train-probe": _train_probe,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return the process exit code (0 done, 2 incomplete or refused)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        return COMMANDS[args.command](args)
    except _INCOMPLETE as err:
        print(f"INCOMPLETE: {err}", file=sys.stderr)
        return 2
    except _STOPPED as err:
        print(f"STOPPED: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
