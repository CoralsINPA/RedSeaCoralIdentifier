"""Command line: `python -m coralid <command> --config <file>`.

Stage B provides one command, `zero-shot`: the 20-genus baseline job of `coralid.baseline`
(resumable, never overwrites). The other commands of the plan (segment, build-manifest, embed,
evaluate, export-review, train-probe) arrive in Stage C.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from coralid import __version__
from coralid.baseline import BaselineError, BaselineIncomplete, run_baseline
from coralid.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of `python -m coralid`."""
    parser = argparse.ArgumentParser(prog="coralid", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"coralid {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    zero = commands.add_parser(
        "zero-shot",
        help="classify every kept crop of the run's manifest and save predictions and embeddings",
        description=(
            "Classify every kept crop of the run's manifest with the configured model, prompts "
            "and label set. Writes predictions/<model>__<prompts>__<labels>__all.parquet and "
            "features/<model>__<crop mode>.npz (each with a .meta.json) under the run folder. "
            "Resumable: finished images are kept in predictions/parts/ and a failed image is "
            "retried on the next run. Never overwrites: if the results exist it loads them."
        ),
    )
    zero.add_argument(
        "--config", required=True, help="a run config such as configs/iou60_sta55_pts32.yaml"
    )
    zero.add_argument(
        "--data-root", help="the data folder (default: paths.data_root or $CORALID_DATA_ROOT)"
    )
    zero.add_argument("--quiet", action="store_true", help="no progress bar")
    return parser


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.data_root:
        cfg.paths.data_root = args.data_root
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return the process exit code (0 done, 2 incomplete or refused)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "zero-shot":
        try:
            result = run_baseline(_config(args), progress=not args.quiet)
        except BaselineIncomplete as err:
            print(f"INCOMPLETE: {err}", file=sys.stderr)
            return 2
        except BaselineError as err:
            print(f"STOPPED: {err}", file=sys.stderr)
            return 2
        verb = "Loaded the saved baseline" if result.status == "already_done" else "Wrote"
        print(f"{verb}: {len(result.predictions)} crops")
        print(f"  predictions: {result.paths.predictions}")
        print(f"  features:    {result.paths.features}")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")  # argparse rejects the others


if __name__ == "__main__":
    raise SystemExit(main())
