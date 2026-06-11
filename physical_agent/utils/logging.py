"""Unified run logger — prints to stdout + records to a log file.

Usage::

    from physical_agent.utils.logging import init_run_logging, get_logger

    # Call once at startup, e.g. after make_log_dir():
    init_run_logging(log_dir)

    # Then anywhere:
    logger = get_logger("agent")
    logger.info("driver ready in %.1fs", elapsed)
    logger.error("driver exited before becoming ready")

The logger writes human-readable messages to stdout and timestamped,
machine-parseable records to ``<log_dir>/run.log``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# All loggers we configure live under this namespace so third-party
# libraries (httpx, anthropic, urllib3, …) don't bleed into our output.
_PKG_LOGGER_NAME = "physical_agent"

_log_initialized = False
_log_dir: Path | None = None


class _ColourFormatter(logging.Formatter):
    """Minimal colour formatter for stdout (no external deps)."""

    _COLOURS = {
        logging.DEBUG: "\033[90m",     # grey
        logging.INFO: "",
        logging.WARNING: "\033[93m",   # yellow
        logging.ERROR: "\033[91m",     # red
        logging.CRITICAL: "\033[95m",  # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelno, "")
        if not colour:
            return super().format(record)
        original = record.levelname
        record.levelname = f"{colour}{original}{self._RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


class _StripPkgPrefixFilter(logging.Filter):
    """Strip the ``physical_agent.`` prefix from the logger name for display."""

    _PREFIX = _PKG_LOGGER_NAME + "."

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == _PKG_LOGGER_NAME:
            record.name = "root"
        elif record.name.startswith(self._PREFIX):
            record.name = record.name[len(self._PREFIX):]
        return True


def init_run_logging(log_dir: str | Path | None = None) -> None:
    """Initialise the run logger.

    Handlers are attached to the ``physical_agent`` package logger (not
    the Python root logger), so only logs emitted via :func:`get_logger`
    are captured.  Third-party libraries (httpx, anthropic, urllib3, …)
    propagate to the unconfigured root logger and stay silent.

    Must be called once (typically at process start).  Subsequent calls
    are no-ops.

    Parameters
    ----------
    log_dir:
        Directory that receives ``run.log``.  When *None* only stdout
        logging is active.
    """
    global _log_initialized, _log_dir

    if _log_initialized:
        return

    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)
    pkg_logger.setLevel(logging.DEBUG)
    pkg_logger.propagate = False
    pkg_logger.handlers.clear()

    strip_filter = _StripPkgPrefixFilter()

    # -- stdout handler (INFO and above) ----------------------------------
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(
        _ColourFormatter("[%(name)s] %(message)s")
    )
    stdout_handler.addFilter(strip_filter)
    pkg_logger.addHandler(stdout_handler)

    # -- file handler (DEBUG and above, timestamped) ---------------------
    if log_dir is not None:
        _log_dir = Path(log_dir)
        _log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            str(_log_dir / "run.log"), encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.addFilter(strip_filter)
        pkg_logger.addHandler(file_handler)

    _log_initialized = True


def get_logger(name: str = "") -> logging.Logger:
    """Return a logger namespaced under the ``physical_agent`` package.

    The returned logger inherits handlers from the package logger
    configured by :func:`init_run_logging`.  Passing an empty string
    returns the package logger itself.

    Typical usage::

        logger = get_logger("agent")
        logger.info("driver ready in %.1fs", elapsed)

    or with a dotted name::

        logger = get_logger("anthropic")
    """
    if name:
        return logging.getLogger(f"{_PKG_LOGGER_NAME}.{name}")
    return logging.getLogger(_PKG_LOGGER_NAME)


def get_log_dir() -> Path | None:
    """Return the log directory set by the last ``init_run_logging`` call."""
    return _log_dir
