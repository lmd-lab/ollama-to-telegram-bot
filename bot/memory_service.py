import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from dotenv import load_dotenv


from utils import safe_load_json

# Paths -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "memory_service_script.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging ---------------------------------------------------------------------------
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
PROFILE_FILE = DATA_DIR / "user_profile.json"

OLLAMA_URL = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

PROFILE_MESSAGES_LIMIT = 30
MIN_MESSAGES_FOR_PROFILE = 10

RAW_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if RAW_CHAT_ID is None:
    raise ValueError("TELEGRAM_CHAT_ID environment variable is missing.")
try:
    CHAT_ID = int(RAW_CHAT_ID)
except (TypeError, ValueError):
    logger.error(f"Invalid TELEGRAM_CHAT_ID: '{RAW_CHAT_ID}' is not a valid integer.")
    raise RuntimeError("Invalid TELEGRAM_CHAT_ID: must be set to a valid integer in your .env file.")

# Load data --------------------------------------------------------------------------------
def load_history(chat_id: int) -> list:
    """Loads the chat history for a given chat_id."""
    chat_histories = safe_load_json(HISTORY_FILE)
    if not chat_histories:
        logger.info("No chat histories found.")
        return []

    history = chat_histories.get(chat_id, [])
    if not history:
        logger.warning(f"No history found for chat_id {chat_id}.")
        return []

    logger.info(f"Loaded {len(history)} messages for chat_id {chat_id}.")
    return history

def load_user_profile(chat_id: int) -> str:
    """Loads the existing user profile for a given chat_id."""
    profiles = safe_load_json(PROFILE_FILE)
    if not profiles:
        logger.info("No user profiles found. Starting fresh.")
        return ""

    entries = profiles.get(chat_id, [])
    if not entries:
        logger.info(f"No existing profile for chat_id {chat_id}.")
        return ""

    latest = entries[-1].get("profile", "")
    logger.info(f"Loaded latest profile for chat_id {chat_id} (version {len(entries)}, updated {entries[-1].get('updated_at', 'unknown')}).")
    return latest

def save_user_profile(chat_id: int, new_profile: str):
    """Appends a new profile version for a given chat_id, keeping all previous versions."""
    profiles = safe_load_json(PROFILE_FILE)
 
    entries = profiles.get(chat_id, [])
    entries.append({
        "updated_at": datetime.now().isoformat(),
        "profile": new_profile
    })
    profiles[chat_id] = entries
 
    try:
        temp_file = PROFILE_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=4)
        temp_file.replace(PROFILE_FILE)
        logger.info(f"Profile version {len(entries)} for chat_id {chat_id} saved successfully.")
    except OSError as e:
        logger.error(f"File system error while saving profile for chat_id {chat_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while saving profile for chat_id {chat_id}: {e}", exc_info=True)

            
# Ask ollama and update memory ---------------------------------------------------------------
def build_prompt(old_profile: str, recent_messages: list) -> str:
    """Builds the prompt for the memory update."""
    formatted_chat = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages])
    return f"""Here is the current profile of the user:
{old_profile if old_profile else "No profile available yet."}

Here are the recent messages from the chat:
{formatted_chat}

Build a short profile of the user based STRICTLY on the current profile and the messages above.
If there is not enough information to make a confident statement, leave it out.
Do NOT invent or assume anything. Only include what is explicitly mentioned.
Respond ONLY with the bullet points, no extra text. If there is nothing to report, respond with "No profile available yet."
"""

def ask_ollama(prompt: str) -> str | None:
    """Sends the prompt to Ollama and returns the updated profile text."""
    logger.info(f"Sending memory update request to Ollama [Model: {MODEL}].")
    try:
        with httpx.Client(timeout=120) as client:
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,
                "options": {"num_predict": 300}
            }
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()

            new_profile = response.json().get("response", "").strip()
            if not new_profile:
                logger.warning("Ollama returned an empty response.")
                return None

            logger.info("Successfully received updated profile from Ollama.")
            return new_profile

    except httpx.TimeoutException:
        logger.error("Ollama request timed out after 120 seconds.")
        return None
    except httpx.ConnectError as e:
        logger.error(f"Could not connect to Ollama at {OLLAMA_URL}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama returned HTTP error {e.response.status_code}: {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request to Ollama failed: {e}")
        return None
    except ValueError:
        logger.error("Received invalid JSON from Ollama or missing 'response' field.")
        return None

# Main -------------------------------------------------------------------------------
def update_memory(chat_id: int):
    """Loads history and profile, asks Ollama to update the profile, and saves it."""
    logger.info(f"Starting memory update for chat_id {chat_id}.")

    history = load_history(chat_id)
    if not history:
        logger.warning("No history available. Aborting memory update.")
        return

    old_profile = load_user_profile(chat_id)


    user_messages = [m for m in history if m["role"] == "user"] 
    recent_messages = user_messages[-PROFILE_MESSAGES_LIMIT:]

    if len(recent_messages) < MIN_MESSAGES_FOR_PROFILE:
        logger.info(f"Not enough user messages ({len(recent_messages)}/{MIN_MESSAGES_FOR_PROFILE}) to build a profile. Skipping.")
        return

    logger.info(f"Using {len(recent_messages)} recent user messages for profile update.")

    prompt = build_prompt(old_profile, recent_messages)
    new_profile = ask_ollama(prompt)

    if not new_profile:
        logger.error("Memory update aborted: no valid response from Ollama.")
        return

    save_user_profile(chat_id, new_profile)
    logger.info(f"Memory update for chat_id {chat_id} completed successfully.")


if __name__ == "__main__":
    update_memory(CHAT_ID)