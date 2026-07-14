"""Centralized logging configuration for BrandParadigm."""

import logging
import sys

from brandparadigm.config.settings import get_settings

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with the project's format."""
    _configure_root()
    return logging.getLogger(name)
