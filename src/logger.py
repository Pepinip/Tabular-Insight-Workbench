"""Logging configuration for Tabular Insight Workbench."""

from __future__ import annotations

import logging
import os
from datetime import datetime

os.makedirs("outputs/logs", exist_ok=True)

log_filename = f"outputs/logs/app_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)

logger = logging.getLogger("tabular_insight")