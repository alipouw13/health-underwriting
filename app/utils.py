
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional


def setup_logging() -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger("underwriting_assistant")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def format_datetime(value: Optional[str]) -> str:
    """Format an ISO8601 datetime string for display."""
    if not value:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return value


def safe_get(obj: Any, *keys: str, default: Any | None = None) -> Any:
    """Safely traverse nested dict-like objects using keys.

    Example:
        safe_get(data, "a", "b", "c") -> data["a"]["b"]["c"] or default
    """
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur
