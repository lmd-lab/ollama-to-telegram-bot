import os
import logging
import json
import httpx 
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Config -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

HISTORY_FILE = BASE_DIR / "chat_history.json"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "generate_script.log"

OLLAMA_URL = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Logging ----------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

# Append to history file ------------------------------------------------------------
def append_to_history(chat_id: str, role: str, content: str):
    """Saves a new entry to the chat history file."""
    histories = {}
    chat_key = int(chat_id)

    try: 
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                histories = {int(k): v for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Could not load existing history: {e}")
    
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content
    }

    histories.setdefault(chat_key, []).append(new_entry)

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False, indent=4)
        logger.info(f"Answer archived for chat {chat_id}")
    except Exception as e:
        logger.error(f"Error saving history: {e}")

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
        return "Error: Ollama request timed out (60s)."
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
            logger.error(f"Ollama issue detected: {answer}")
            return

        append_to_history(CHAT_ID, "assistant", answer)
        send_telegram_message(answer)

    except Exception as e:
        logger.error(f"General error in main: {e}")

if __name__ == "__main__":
    main()