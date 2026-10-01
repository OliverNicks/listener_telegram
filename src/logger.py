import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Ensure environment variables from .env are loaded before reading LOG_LEVEL
load_dotenv()

# Remove the default loguru handler
logger.remove()

# Read logging level from .env (defaults to INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 1. Attractive, easy-to-read colour output to the developer console
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level> "
        "<dim>{extra}</dim>"
    ),
    colorize=True,
)

# 2. Structured JSON log written to file in project root for production and analytics
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    str(LOGS_DIR / "listener_{time:YYYY-MM-DD}.log"),
    level="INFO",
    rotation="20 MB",      # Create a new file every 20 MB
    retention="14 days",   # Keep logs for the last 14 days
    compression="zip",     # Archive old logs
    serialize=True,        # Write each log as a separate JSON string
    encoding="utf-8",
)

__all__ = ["logger"]
