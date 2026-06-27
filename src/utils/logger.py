"""Centralised logging built on Loguru.

Call :func:`setup_logging` once at process start (the pipeline and the Streamlit
app both do this). Everywhere else just ``from src.utils.logger import logger``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Configure console + rotating file sinks. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()  # drop the default stderr handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, enqueue=True, backtrace=False)
    logger.add(
        log_path / "eoir_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,           # thread/process-safe — vital for our worker threads
        backtrace=True,
        diagnose=False,
    )
    _CONFIGURED = True
    logger.info("Logging initialised (level={}, dir={})", level, str(log_path))


__all__ = ["logger", "setup_logging"]
