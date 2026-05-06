import logging
import os
import json
from datetime import datetime
from rich.logging import RichHandler


class RichJsonFormatter(logging.Formatter):
    """Rich JSON formatter that captures extra fields."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent", "system"),
            "message": record.getMessage(),
        }
        
        # Capture any extra fields passed via 'extra'
        if hasattr(record, "payload"):
            log_entry["payload"] = record.payload
        
        return json.dumps(log_entry, ensure_ascii=False, indent=2)


def configure_logging(level: str = "INFO") -> None:
    """Configure local logging to console (Rich) and file (Pretty JSON)."""
    
    logger = logging.getLogger()
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 1. Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console_handler)
    
    # 2. Pretty JSON File Handler
    os.makedirs("logs", exist_ok=True)
    # Note: Using 'w' for fresh log per run or 'a' for history. Let's use 'a' for history.
    file_handler = logging.FileHandler("logs/execution_trace.json", encoding="utf-8", mode="a")
    file_handler.setFormatter(RichJsonFormatter())
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logging.info("Logging system updated: Rich JSON enabled at logs/execution_trace.json")
