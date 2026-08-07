"""Package logger for run output and ``run.log`` files."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from rpent.utils.config import get_repo_root

# All loggers we configure live under this namespace so third-party
# libraries (httpx, anthropic, urllib3, …) don't bleed into our output.
_PKG_LOGGER_NAME = "rpent"

_log_initialized = False
_output_dir: Path | None = None


class _ColourFormatter(logging.Formatter):
    """Colour only the level marker; never mutates the shared record."""

    _COLOURS = {
        logging.DEBUG: "\033[90m",     # grey
        logging.INFO: "",
        logging.WARNING: "\033[93m",   # yellow
        logging.ERROR: "\033[91m",     # red
        logging.CRITICAL: "\033[95m",  # magenta
    }
    _LEVEL_LETTERS = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "C",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        body = super().format(record)
        colour = self._COLOURS.get(record.levelno, "")
        letter = self._LEVEL_LETTERS.get(record.levelno, "?")
        marker = f"{colour}{letter}{self._RESET}" if colour else letter
        return f"{marker} {body}"


class _CompactLevelFormatter(logging.Formatter):
    """Formatter that renders the log level as a single letter (D/I/W/E/C)."""

    _LEVEL_LETTERS = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "C",
    }

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = self._LEVEL_LETTERS.get(record.levelno, original[:1])
        try:
            return super().format(record)
        finally:
            record.levelname = original


class _StripPkgPrefixFilter(logging.Filter):
    """Strip the ``rpent.`` prefix from the logger name for display."""

    _PREFIX = _PKG_LOGGER_NAME + "."

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == _PKG_LOGGER_NAME:
            record.name = "root"
        elif record.name.startswith(self._PREFIX):
            record.name = record.name[len(self._PREFIX):]
        return True


def init_output_dir(log_dir: str | Path | None = None, verbose: bool = False) -> Path:
    """Create *log_dir* (defaults to ``<repo>/logs/``), set up logging, and
    return the resolved path.

    When *verbose* is True, both stdout and the ``run.log`` file log at DEBUG;
    otherwise both log at INFO.
    """
    global _log_initialized, _output_dir

    if log_dir is None:
        log_dir = get_repo_root() / "logs"
    _output_dir = Path(log_dir)
    if not _log_initialized and _output_dir.exists() and any(_output_dir.iterdir()):
        print(
            f"Warning: RPent output directory is not empty: {_output_dir}; existing files may be overwritten!",
            file=sys.stderr,
            flush=True,
        )
    _output_dir.mkdir(parents=True, exist_ok=True)

    if _log_initialized:
        return _output_dir

    level = logging.DEBUG if verbose else logging.INFO

    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False
    pkg_logger.handlers.clear()

    strip_filter = _StripPkgPrefixFilter()

    # -- stdout handler ---------------------------------------------------
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(
        _ColourFormatter("[%(name)s] %(message)s")
    )
    stdout_handler.addFilter(strip_filter)
    pkg_logger.addHandler(stdout_handler)

    # -- file handler (timestamped) --------------------------------------
    file_handler = logging.FileHandler(
        str(_output_dir / "run.log"), encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        _CompactLevelFormatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler.addFilter(strip_filter)
    pkg_logger.addHandler(file_handler)

    _log_initialized = True
    return _output_dir


def get_output_dir() -> Path:
    """Return the output directory set by the last ``init_output_dir`` call."""
    assert _output_dir is not None, (
        "init_output_dir must be called before get_output_dir"
    )
    return _output_dir


def get_logger(name: str = "") -> logging.Logger:
    """Return a logger below the ``rpent`` namespace."""
    if name:
        return logging.getLogger(f"{_PKG_LOGGER_NAME}.{name}")
    return logging.getLogger(_PKG_LOGGER_NAME)
