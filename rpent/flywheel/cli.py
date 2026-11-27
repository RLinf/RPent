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
        description="Export VLA training frames or API planner turns for VLM training.",
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
        help="Also export VLM turns from episodes with failed or unknown environment outcomes (VLM only).",
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
        default=None,
        help="Positive window length used only for report counts (VLA only, default: 10). "
        "Exported frames and tails are unchanged.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Local LeRobot dataset identifier (VLA only, default: rpent/vla-sft). "
        "Does not upload data.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=None,
        help="Image-writing threads shared by all cameras; "
        "0 writes synchronously (VLA only, default: 0).",
    )
    return parser


def main() -> int:
    """Export trajectories and display the resulting counts and paths."""
    parser = _parser()
    args = parser.parse_args()
    if args.include_unsuccessful and not args.vlm:
        parser.error("--include-unsuccessful requires --vlm")
    vla_only = [
        flag
        for flag, value in (
            ("--horizon", args.horizon),
            ("--repo-id", args.repo_id),
            ("--image-writer-threads", args.image_writer_threads),
        )
        if value is not None
    ]
    if args.vlm and vla_only:
        parser.error(f"{', '.join(vla_only)} cannot be used with --vlm")
    try:
        from rpent.flywheel.export import export_dataset, export_vlm_samples

        if args.vlm:
            report = export_vlm_samples(
                args.input_root,
                args.output_dir,
                pattern=args.pattern,
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
        vla_kwargs = {
            name: value
            for name, value in (
                ("horizon", args.horizon),
                ("repo_id", args.repo_id),
                ("image_writer_threads", args.image_writer_threads),
            )
            if value is not None
        }
        report = export_dataset(
            args.input_root,
            args.output_dir,
            pattern=args.pattern,
            **vla_kwargs,
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
