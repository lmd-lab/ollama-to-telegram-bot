import os
import logging
from pathlib import Path
import httpx
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Config & Paths ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Logging ----------------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "bot.log"
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

# Files -----------------------------------------------------------------------------
HISTORY_FILE = BASE_DIR / "chat_history.json"
SETTINGS_FILE = BASE_DIR / "chat_settings.json"

OLLAMA_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b") 

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is not set in the environment. Please set it in your .env file or environment variables.")

RAW_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if RAW_CHAT_ID is None:
    raise ValueError("TELEGRAM_CHAT_ID environment variable is missing.")
try:
    CHAT_ID = int(RAW_CHAT_ID)
except (TypeError, ValueError):
    logger.error(f"Invalid TELEGRAM_CHAT_ID: '{RAW_CHAT_ID}' is not a valid integer.")
    raise RuntimeError("Invalid TELEGRAM_CHAT_ID: must be set to a valid integer in your .env file.")

AVAILABLE_MODELS = {
    "llama": "llama3.2:1b",
    "qwen": "qwen2.5:1.5b",
}

try:
    MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
    if MAX_HISTORY < 1:
        raise ValueError
except (TypeError, ValueError):
    logger.warning("Invalid MAX_HISTORY environment variable. Using default value 20.")
    MAX_HISTORY = 20


# Conversation context & settings ----------------------------------------------------
chat_histories = {} # {chat_id (int): [{"timestamp": ..., "role": "user"/"assistant", "content": ...}, ...]}
chat_settings = {} # {chat_id (int): {"model": str, "offset": int}}

def load_histories():
    """Loads chat histories from the JSON file on startup."""
    global chat_histories
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                chat_histories = {int(k): v for k, v in data.items()}
            logger.info("History loaded from file.")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

def save_histories():
    """Saves the current chat histories to a JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_histories, f, ensure_ascii=False, indent=4)
        logger.info("History saved to file.")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

def load_settings():
    """Loads the current chat settings (models, offsets) from a JSON file."""
    global chat_settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Key wieder in int umwandeln
                chat_settings = {int(k): v for k, v in data.items()}
            logger.info("Settings loaded from file.")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
    else:
        logger.info("No settings file found. Starting with empty settings.")

def save_settings():
    """Saves the current chat settings (models, offsets) to a JSON file."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_settings, f, ensure_ascii=False, indent=4)
        logger.info("Settings saved to file.")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# Helper functions ----------------------------------------------------------------------
async def notify_me(text):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"Critical: notify_me failed. Original message: {text}. Error: {e}")

def get_model(chat_id: int) -> str:
    return chat_settings.get(chat_id, {}).get("model", DEFAULT_MODEL)

def get_offset(chat_id: int) -> int:
    return chat_settings.get(chat_id, {}).get("offset", 0)

def set_model(chat_id: int, model_key: str):
    if model_key in AVAILABLE_MODELS:
        settings = chat_settings.setdefault(chat_id, {"model": DEFAULT_MODEL, "offset": 0})
        settings["model"] = AVAILABLE_MODELS[model_key]
        save_settings()

def set_offset(chat_id: int, offset: int):
    settings = chat_settings.setdefault(chat_id, {"model": DEFAULT_MODEL, "offset": 0})
    settings["offset"] = offset
    save_settings()

def get_history(chat_id: int) -> list[dict]:
    return chat_histories.setdefault(chat_id, [])

def append_message(chat_id: int, role: str, content: str):
    history = get_history(chat_id)
    history.append({
        "timestamp": datetime.now().isoformat(), 
        "role": role, 
        "content": content})
    save_histories()

# Authorization check ----------------------------------------------------------------------
def is_authorized(update: Update) -> bool:
    return update.effective_chat.id == CHAT_ID

# Ollama interaction ----------------------------------------------------------------------
async def query_ollama(chat_id: int, prompt: str) -> str:
    """
    Handles history management, API request, and error handling.
    """
    model = get_model(chat_id)
    full_history = get_history(chat_id)
    offset = get_offset(chat_id)

    # Logic: Only messages AFTER offset, then limited by MAX_HISTORY
    visible_history = full_history[offset:]
    recent_history = visible_history[-MAX_HISTORY:] if len(visible_history) > MAX_HISTORY else visible_history

    system_instruction = {"role": "system", "content": "Keep responses short and concise."}

    messages = [system_instruction] + recent_history + [{"role": "user", "content": prompt}]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "1m",
        "options": {
            #"num_predict": 500,  # Optional: Limit response length
            "temperature": 0.7,  # Optional: Adjust creativity
            },
    }

    logger.info(f"Sending request to Ollama [Model: {model}, Chat: {chat_id}]")
    start_time = datetime.now()

    try:
        # Request with a 2-minute timeout for slower local models
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Ollama responded in {duration:.2f}s")

            answer = response.json()["message"]["content"]
            append_message(chat_id, "user", prompt)
            append_message(chat_id, "assistant", answer)
            return answer

    except httpx.ConnectError as e:
        logger.error(f"Connection Error: Ollama unreachable at {OLLAMA_URL}. Details: {e}")
        await notify_me("The bot has a hiccup!")
        return "Ollama is unreachable. Is the service running?"

    except httpx.TimeoutException:
        logger.warning(f"Timeout: Model '{model}' did not respond within 120s.")
        await notify_me("The bot has a hiccup!")
        return f"Timeout: '{model}' is taking too long."

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
        await notify_me("The bot has a hiccup!")
        return f"Ollama returned an error (HTTP {e.response.status_code})."

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Unexpected Ollama Error after {duration:.2f}s: {e}", exc_info=True)
        await notify_me("The bot has a hiccup!")
        return f"An unexpected error occurred: {str(e)}"

# Telegram-Handlers ----------------------------------------------------------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return 

    current_model = get_model(update.effective_chat.id)
    help_text = (
        f"Hi! I am your Ollama bot (Model: *{current_model}*).\n"
        "Just send me a message, or switch the model using /model.\n" 
        "Use /stats to view message statistics.\n" 
        "Use /clear to reset our conversation history (keeps messages in archive).\n"
        f"Or /clear and any number between 1 and {MAX_HISTORY} to reduce remembered messages to that number (keeps messages in archive).\n\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return

    keyboard = [
        [
            InlineKeyboardButton("Llama 3.2 (1B)", callback_data="model_llama"),
            InlineKeyboardButton("Qwen 2.5 (1.5B)", callback_data="model_qwen"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose a model:", reply_markup=reply_markup)

async def unload_model(model_name: str):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "keep_alive": 0
            })
            logger.info(f"Model {model_name} was unloaded.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"Model {model_name} not loaded.")
        else:
            logger.error(f"Error unloading {model_name}: {e}")

async def model_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    chat_id = update.effective_chat.id
    
    query = update.callback_query
    chat_id = query.message.chat.id
    await query.answer()

    chat_id = query.message.chat_id
    old_model = get_model(chat_id)    
    model_key = query.data.split("_")[1]  
    new_model = AVAILABLE_MODELS[model_key]

    if old_model != new_model:
        await query.edit_message_text(f"Unloading {old_model} and switching to {new_model}...")
        await unload_model(old_model)

        set_model(chat_id, model_key)
        await query.edit_message_text(f"Model successfully switched to: *{new_model}*", parse_mode="Markdown")
    else:
        await query.edit_message_text(f"*{new_model}* is already active.", parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        logger.warning(f"Unauthorized access attempt by user {update.effective_user.id}")
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    reply = await query_ollama(chat_id, user_text)
    await update.message.reply_text(reply)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    chat_id = update.effective_chat.id
    history = get_history(chat_id)

    if not context.args:     
        new_offset = len(history)
        set_offset(chat_id, new_offset)
        logger.info(f"Chat {chat_id} history cleared (new offset: {new_offset})")
        await update.message.reply_text("Memory cleared for AI!")
        return

    try:        
        number = int(context.args[0])
        if 1 <= number <= MAX_HISTORY:
            new_offset = max(len(history) - number, 0)
            set_offset(chat_id, new_offset)
            logger.info(f"Chat {chat_id} history cleared to last {number} messages (new offset: {new_offset})")
            await update.message.reply_text(f"Memory cleared to last {number} messages for AI!")
        else:
            logger.warning(f"Invalid /clear argument: {context.args[0]} (must be between 1 and {MAX_HISTORY})")
            await update.message.reply_text(f"Please provide a number between 1 and {MAX_HISTORY}.")

    except ValueError:
        logger.warning(f"Invalid /clear argument: {context.args[0]} (not a number)")
        await update.message.reply_text("Invalid argument. Use /clear or /clear <number>.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return

    chat_id = update.effective_chat.id
    history = get_history(chat_id)
    offset = min(get_offset(chat_id), len(history))
    
    total = len(history)
    visible_history = history[offset:]
    visible = len(visible_history)
    ai_sees = min(visible, MAX_HISTORY) 
    first_msg_time = history[0]["timestamp"].split("T")[0] if history else "No messages yet"
    first_msg_time = history[0]["timestamp"].split("T")[0] if history else "No messages yet"
    
    await update.message.reply_text(
        f"Chat Statistics\n\n"
        f"Current Model: `{get_model(chat_id)}`\n"
        f"First message: {first_msg_time}\n"
        f"Total archived messages: `{total}`\n"
        f"Offset: `{offset}`\n"
        f"Messages since last /clear: `{visible}`\n"
        f"Messages visible to AI: `{ai_sees}` / {MAX_HISTORY}\n",
        parse_mode="Markdown"
    )

# main program ----------------------------------------------------------------------
def main():
    load_histories()
    load_settings() 

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Handler registrieren
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CallbackQueryHandler(model_select, pattern=r"^model_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Chat-Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()