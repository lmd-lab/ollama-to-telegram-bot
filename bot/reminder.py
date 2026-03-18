import os
import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from dotenv import load_dotenv
from filelock import Timeout

from utils import safe_load_json, HISTORY_LOCK

# Paths -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "reminder.log" 
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging ----------------------------------------------------------------------------
file_handler = RotatingFileHandler(
    LOG_FILE, 
    maxBytes=1*1024*1024, 
    backupCount=5,         
    encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Environment & Settings ---------------------------------------------------------------
HISTORY_FILE = DATA_DIR / "chat_history.json"
# Todo: add settings to used selected model

OLLAMA_URL = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL")
if not MODEL:
    raise RuntimeError("OLLAMA_MODEL is not set in .env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

RAW_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if RAW_CHAT_ID is None:
    raise ValueError("TELEGRAM_CHAT_ID environment variable is missing.")
try:
    CHAT_ID = int(RAW_CHAT_ID)
except (TypeError, ValueError):
    logger.error(f"Invalid TELEGRAM_CHAT_ID: '{RAW_CHAT_ID}' is not a valid integer.")
    raise RuntimeError("Invalid TELEGRAM_CHAT_ID: must be set to a valid integer in your .env file.")

# Append to history file ------------------------------------------------------------
def append_to_history(chat_id: str, role: str, content: str):
    """Saves a new entry to the chat history file."""
    chat_key = int(chat_id)

    try:
        with HISTORY_LOCK:
            chat_histories = safe_load_json(HISTORY_FILE)
            if chat_histories:
                logger.info("Chat histories loaded successfully.")
            else:
                logger.info("No chat histories found. Starting fresh.")

            chat_histories.setdefault(chat_key, []).append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content
            })

            temp_file = HISTORY_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(chat_histories, f, ensure_ascii=False, indent=4)
            temp_file.replace(HISTORY_FILE)
            logger.info(f"Answer archived for chat {chat_id}")

    except Timeout:
        logger.error(f"Could not acquire file lock for {HISTORY_FILE} – another process is blocking.")
    except OSError as e:
        logger.error(f"File system error while saving history for chat {chat_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in append_to_history for chat {chat_id}: {e}", exc_info=True)

# Functions --------------------------------------------------------------------------
def get_prompt() -> str:
    """
    Sets the prompt for the model. 
    It first checks if an environment variable is set, 
    and if not, it returns a default prompt.
    
    :return: The prompt to be sent to the model.
    :rtype: str
    """
    prompt = os.getenv("OLLAMA_PROMPT")
    if not prompt or not prompt.strip():
        raise EnvironmentError("OLLAMA_PROMPT is missing in .env file")
    return prompt

def ask_ollama(prompt: str) -> str:
    """
    Asks the model for a response based on the provided prompt.
    
    :param prompt: The prompt to send to the model.
    :type prompt: str
    :return: The response from the model or an error message.
    :rtype: str
    """
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(OLLAMA_URL, json={
               "model": MODEL,
               "prompt": prompt,
               "stream": False,
               "keep_alive": 0
            })
            response.raise_for_status()
            data = response.json()

            answer = data.get("response", "").strip()
            if not answer:
                return "Error: Model returned an empty response."

            return answer
    except httpx.TimeoutException:
        return "Error: Ollama request timed out (120s)."
    except httpx.RequestError as e:
        return f"Error: Request failed - {e}"
    except ValueError:
        return "Error: Received invalid JSON or missing 'response' field."

def send_telegram_message(text: str):
    """
    Sends a message to a Telegram chat using the Telegram Bot API
    privded by the TELEGRAM_TOKEN and CHAT_ID environment variables.
    
    :param text: The text to send to the Telegram chat.
    :type text: str
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": text
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram Error: {data}")
    
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        raise

def main():
    try:
        prompt = get_prompt()
        answer = ask_ollama(prompt)

        if answer.startswith("Error:"):
            send_telegram_message("Reminder failed. Please check the logs for details.")
            logger.error(f"Ollama issue detected: {answer}")
            return

        marked_answer = f"[Automatic Reminder message]: {answer}"
        append_to_history(CHAT_ID, "assistant", marked_answer)
        send_telegram_message(answer)

    except Exception as e:
        logger.error(f"General error in main: {e}")

if __name__ == "__main__":
    main()