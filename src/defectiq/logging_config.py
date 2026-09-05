"""Consistent logging across training, API and dashboard."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level or "INFO"
    if isinstance(resolved, str):
        resolved = logging.getLevelName(resolved.upper())

    logging.basicConfig(level=resolved, format=_FORMAT, datefmt=_DATE_FORMAT, stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
