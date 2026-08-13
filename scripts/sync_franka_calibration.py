"""Sync Franka hand-eye YAML calibration files into the project bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "robots"
    / "franka"
    / "calibration"
    / "hand_eye_calibration.json"
)


def _parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"parameters": {}, "transformation": {}}
    section: str | None = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            data.setdefault(section, {})
            continue
        if section is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        data[section][key.strip()] = _parse_scalar(value.strip())
    return data


def _load_handeye_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(path.read_text(errors="replace"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _parse_simple_yaml(path)


def _bundle_entry(path: Path) -> dict[str, Any]:
    data = _load_handeye_yaml(path)
    return {
        "source_name": path.name,
        "parameters": data.get("parameters") or {},
        "transformation": data.get("transformation") or {},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy Franka hand-eye YAML fields into the project bundle.",
    )
    parser.add_argument(
        "--external",
        default=str(ROOT.parent / "fr3_external_apriltag_eye_on_base.yaml"),
        help="External camera eye-on-base YAML path.",
    )
    parser.add_argument(
        "--wrist",
        default=str(ROOT.parent / "fr3_wrist_apriltag_ee_eye_on_hand.yaml"),
        help="Wrist camera eye-on-hand YAML path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output project calibration bundle path.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    external = Path(args.external).expanduser().resolve()
    wrist = Path(args.wrist).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not external.exists():
        raise FileNotFoundError(f"external calibration YAML not found: {external}")
    if not wrist.exists():
        raise FileNotFoundError(f"wrist calibration YAML not found: {wrist}")

    bundle = {
        "version": 1,
        "external": _bundle_entry(external),
        "wrist": _bundle_entry(wrist),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
