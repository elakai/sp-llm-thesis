import logging
import sys
import io
from datetime import datetime, timezone, timedelta
from src.config.constants import LOGS_FOLDER

# Philippine Standard Time (UTC+8)
PHT = timezone(timedelta(hours=8))

class PHTFormatter(logging.Formatter):
    """Formatter that always uses Philippine Standard Time (UTC+8)."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=PHT)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def setup_logging():
    # 1. Ensure the log folder exists FIRST
    LOGS_FOLDER.mkdir(parents=True, exist_ok=True) 
    log_file = LOGS_FOLDER / "axissant.log"
    
    logger = logging.getLogger("AXIsstant")
    logger.setLevel(logging.INFO)
    formatter = PHTFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 2. FILE HANDLER
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 3. CONSOLE HANDLER (Windows-safe UTF-8 wrapping)
    if sys.platform == "win32":
        # Wrap stdout to handle emojis without leaking file descriptors
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setFormatter(formatter)

    # Clean up any existing handlers to prevent duplicates on hot-reload
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger

# Still initialized at import, but now folder-safe
logger = setup_logging()