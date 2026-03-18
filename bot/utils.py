import json
import logging
import time
from pathlib import Path

from filelock import FileLock

# Paths & Locks ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_LOCK = FileLock(str(DATA_DIR / "chat_history.json") + ".lock")

# Logging ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

def safe_load_json(file_path, max_attempts=5):
    """Tries to safely load a JSON file, even if other processes are writing to it."""
    if not file_path.exists():
        return {}

    for attempt in range(max_attempts):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {int(k) if k.isdigit() else k: v for k, v in data.items()}
                return data
        except (json.JSONDecodeError, OSError, BlockingIOError, PermissionError) as e:
            if attempt < max_attempts - 1:
                logger.warning(f"File {file_path.name} busy or error (Attempt {attempt+1}/{max_attempts}): {e}")
                time.sleep(0.05)
            else:
                logger.error(f"Critical: Could not load {file_path.name} after {max_attempts} attempts.")
        
        except Exception as e:
            logger.error(f"Unexpected error while loading {file_path.name}: {e}")
            break    
    return {}