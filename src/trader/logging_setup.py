"""Configuration du logging structure (structlog, sortie JSON).

Chaque decision de trade doit rester tracable a posteriori : on logue en JSON
avec un timestamp UTC ISO-8601 et le nom du module emetteur.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(
    level: str = "INFO", json_output: bool = True, log_file: str | Path | None = None
) -> None:
    """Configure structlog et la stdlib. Idempotent."""
    global _CONFIGURED

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Retourne un logger structure. Configure le logging au premier appel."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
