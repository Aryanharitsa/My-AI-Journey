from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_time=True,
        show_path=False,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured rich-backed logger. Idempotent."""
    _configure_root()
    return logging.getLogger(name if name else "vitruvius")
