# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line export of recorded trajectories to training data."""

import argparse
from pathlib import Path

from rpent.utils.logging import get_logger

logger = get_logger("flywheel")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rpent-flywheel",
        description="Export VLA training frames or cleaned VLM planner turns.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--vla",
        action="store_true",
        help="Export VLA training data (default).",
    )
    mode.add_argument(
        "--vlm",
        action="store_true",
        help="Export API planner turns from environment-confirmed successes to samples.jsonl.",
    )
    parser.add_argument(
        "--include-unsuccessful",
        action="store_true",
        help="Also export VLM turns from episodes with failed or unknown environment outcomes.",
    )
    parser.add_argument(
        "--image-history", type=int, help="VLM observation groups to keep (default: 2)."
    )
    parser.add_argument(
        "--reasoning",
        choices=("drop", "keep"),
        help="VLM thinking policy (default: drop).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="One trajectory directory or a directory containing trajectory runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for the training dataset and report.json.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob for child directories to search recursively (default: *). "
        "Ignored when input-root contains task.json. Quote shell wildcards.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Positive window length used only for report counts (default: 10). "
        "Exported frames and tails are unchanged.",
    )
    parser.add_argument(
        "--repo-id",
        default="rpent/vla-sft",
        help="Local LeRobot dataset identifier (default: rpent/vla-sft). "
        "Does not upload data.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=0,
        help="Image-writing threads shared by all cameras; "
        "0 writes synchronously (default: 0).",
    )
    return parser


def main() -> int:
    """Export trajectories and display the resulting counts and paths."""
    parser = _parser()
    args = parser.parse_args()
    if not args.vlm and (args.image_history is not None or args.reasoning is not None):
        parser.error("--image-history and --reasoning require --vlm")
    if args.include_unsuccessful and not args.vlm:
        parser.error("--include-unsuccessful requires --vlm")
    try:
        if args.vlm:
            from rpent.flywheel.vlm_clean import export_vlm_samples

            report = export_vlm_samples(
                args.input_root,
                args.output_dir,
                pattern=args.pattern,
                image_history=2 if args.image_history is None else args.image_history,
                reasoning=args.reasoning or "drop",
                include_unsuccessful=args.include_unsuccessful,
            )
            print(
                f"Episodes: {report['episode_count']} read, {report['selected_episode_count']} selected, {report['exported_episode_count']} exported"
            )
            print(f"Turns read: {report['turn_count']}")
            print(f"Samples: {report['sample_count']}")
            print(f"Unique images: {report['image_count']}")
            print(f"Excluded turns: {len(report['excluded'])}")
            print(f"Report: {args.output_dir / 'report.json'}")
            return 0
        from rpent.flywheel.export import export_dataset

        report = export_dataset(
            args.input_root,
            args.output_dir,
            pattern=args.pattern,
            horizon=args.horizon,
            repo_id=args.repo_id,
            image_writer_threads=args.image_writer_threads,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot":
            raise
        logger.error(
            'LeRobot is required. Install it with: uv pip install -e ".[flywheel]"'
        )
        return 1
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    output = Path(report["output_dir"])
    print(f"Trajectories: {report['trajectory_count']}")
    print(f"Exported trajectories: {report['exported_trajectory_count']}")
    print(f"Segments: {report['episode_counts']['train']}")
    print(f"Frames: {report['frame_counts']['train']}")
    print(
        f"Complete windows (horizon={report['horizon']}): "
        f"{report['window_counts']['train']}"
    )
    if report["frame_counts"]["train"]:
        print(f"Dataset: {output / report['dataset_paths']['train']}")
    else:
        logger.warning("No training frames exported; see report.json for exclusions.")
    print(f"Report: {output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
