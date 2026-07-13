#!/usr/bin/env python3
"""Generate a printable ChArUco fiducial board for Franka camera calibration.

The output files are:

- ``<name>.png``: high-resolution board image with DPI metadata.
- ``<name>.pdf``: printable single-page PDF at the same physical scale.
- ``<name>.json``: board spec consumed by future calibration scripts.

Print the PDF at 100% / actual size, not "fit to page". After printing, measure
one square and use the measured value if it differs from the requested size.

Dependency:
    uv pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT_DIR = _REPO_ROOT / "resources" / "franka" / "calibration_boards"


def _require_cv2_aruco():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: cv2. Install with:\n"
            "  uv pip install opencv-contrib-python-headless\n"
            "or:\n"
            "  pip install opencv-contrib-python-headless"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "Installed cv2 lacks the aruco module. Install opencv-contrib, not plain opencv:\n"
            "  uv pip install opencv-contrib-python-headless"
        )
    return cv2


def _aruco_dictionary(cv2, name: str):
    aruco = cv2.aruco
    key = f"DICT_{name.upper()}" if not name.upper().startswith("DICT_") else name.upper()
    if not hasattr(aruco, key):
        available = sorted(attr[5:] for attr in dir(aruco) if attr.startswith("DICT_"))
        raise SystemExit(f"Unknown ArUco dictionary {name!r}. Available examples: {available[:12]}")
    return aruco.getPredefinedDictionary(getattr(aruco, key)), key


def _make_charuco_board(
    cv2,
    *,
    squares_x: int,
    squares_y: int,
    square_px: int,
    marker_px: int,
    dictionary,
):
    aruco = cv2.aruco
    try:
        return aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_px),
            float(marker_px),
            dictionary,
        )
    except Exception:
        return aruco.CharucoBoard_create(
            int(squares_x),
            int(squares_y),
            float(square_px),
            float(marker_px),
            dictionary,
        )


def _draw_board(board, image_size: tuple[int, int], margin_px: int) -> np.ndarray:
    """Draw a ChArUco board across OpenCV API versions."""
    width, height = image_size
    if hasattr(board, "generateImage"):
        img = board.generateImage((width, height), marginSize=int(margin_px), borderBits=1)
    else:
        img = board.draw((width, height), marginSize=int(margin_px), borderBits=1)
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def _save_outputs(
    image: np.ndarray,
    *,
    out_dir: Path,
    name: str,
    dpi: int,
    spec: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(image, mode="L")
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    json_path = out_dir / f"{name}.json"
    pil.save(png_path, dpi=(dpi, dpi))
    pil.save(pdf_path, "PDF", resolution=float(dpi))
    json_path.write_text(json.dumps(spec, indent=2))
    return {"png": str(png_path), "pdf": str(pdf_path), "json": str(json_path)}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--name", default="franka_charuco_7x5_25mm")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--marker-mm", type=float, default=18.0)
    parser.add_argument("--margin-mm", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dictionary", default="4X4_50", help="OpenCV ArUco dictionary suffix, e.g. 4X4_50 or APRILTAG_36h11.")
    parser.add_argument("--check-deps", action="store_true", help="Only check for cv2.aruco and exit.")
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    cv2 = _require_cv2_aruco()
    if args.check_deps:
        print(f"cv2 {cv2.__version__} with aruco: OK")
        return 0

    if args.squares_x < 2 or args.squares_y < 2:
        raise SystemExit("--squares-x and --squares-y must be >= 2")
    if not (0 < args.marker_mm < args.square_mm):
        raise SystemExit("--marker-mm must be >0 and < --square-mm")

    px_per_mm = float(args.dpi) / 25.4
    square_px = int(round(args.square_mm * px_per_mm))
    marker_px = int(round(args.marker_mm * px_per_mm))
    margin_px = int(round(args.margin_mm * px_per_mm))
    board_width_px = args.squares_x * square_px
    board_height_px = args.squares_y * square_px
    image_size = (board_width_px + 2 * margin_px, board_height_px + 2 * margin_px)

    dictionary, dictionary_name = _aruco_dictionary(cv2, args.dictionary)
    board = _make_charuco_board(
        cv2,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_px=square_px,
        marker_px=marker_px,
        dictionary=dictionary,
    )
    image = _draw_board(board, image_size, margin_px)

    spec = {
        "type": "charuco",
        "dictionary": dictionary_name,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_m": args.square_mm / 1000.0,
        "marker_length_m": args.marker_mm / 1000.0,
        "margin_m": args.margin_mm / 1000.0,
        "dpi": args.dpi,
        "image_width_px": int(image_size[0]),
        "image_height_px": int(image_size[1]),
        "print_instructions": [
            "Print the PDF at 100% / actual size.",
            "Disable fit-to-page or scaling.",
            "Use matte paper and mount it flat to cardboard/foam board.",
            "Measure one printed square and update square_length_m if needed.",
        ],
    }
    paths = _save_outputs(image, out_dir=args.out_dir, name=args.name, dpi=args.dpi, spec=spec)

    physical_w_mm = image_size[0] / px_per_mm
    physical_h_mm = image_size[1] / px_per_mm
    print(json.dumps({"paths": paths, "physical_size_mm": [round(physical_w_mm, 1), round(physical_h_mm, 1)], "spec": spec}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
