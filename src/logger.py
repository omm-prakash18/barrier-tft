"""
logger.py
─────────
Structured logging for the TFT + FinBERT pipeline.

Provides a single `get_logger(name)` factory that returns a logger writing
JSON-structured records to stderr (for container log aggregators) and a
human-readable format to stdout, simultaneously.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("step.complete", step=1, rows=42_000, elapsed_sec=3.1)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts":      self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        # Attach any extra kwargs passed via log.info("msg", extra={...})
        if hasattr(record, "extra"):
            base.update(record.extra)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base)


class _HumanFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        prefix = f"{color}[{record.levelname[0]}]{self.RESET}"
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        msg = record.getMessage()
        extra_str = ""
        if hasattr(record, "extra"):
            extra_str = "  " + "  ".join(
                f"{k}={v}" for k, v in record.extra.items()
            )
        return f"{ts} {prefix} {record.name}: {msg}{extra_str}"


class _StructuredLogger(logging.Logger):
    """
    Logger subclass that supports structured keyword args.

    Usage:
        log.info("event.name", step=1, elapsed=3.2)
    """

    def _log_extra(
        self,
        level: int,
        msg: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        extra = {k: v for k, v in kwargs.items() if k not in (
            "exc_info", "stack_info", "stacklevel"
        )}
        record_kwargs = {k: v for k, v in kwargs.items() if k in (
            "exc_info", "stack_info", "stacklevel"
        )}
        self.log(level, msg, *args, extra={"extra": extra}, **record_kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:      # type: ignore[override]
        self._log_extra(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:     # type: ignore[override]
        self._log_extra(logging.DEBUG, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:   # type: ignore[override]
        self._log_extra(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:     # type: ignore[override]
        self._log_extra(logging.ERROR, msg, *args, **kwargs)


_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    logging.setLoggerClass(_StructuredLogger)
    root = logging.getLogger()
    root.setLevel(level)

    # Human-readable → stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_HumanFormatter())
    root.addHandler(sh)


def get_logger(name: str, level: int = logging.INFO) -> _StructuredLogger:
    """Return a structured logger for *name*. Call once per module."""
    _configure_root(level)
    return logging.getLogger(name)  # type: ignore[return-value]
