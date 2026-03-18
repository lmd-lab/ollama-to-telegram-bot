# Ollama Telegram Bot & Reminder

A Python-based Telegram assistant that combines a 24/7 Chat Bot and a Scheduled Reminder. It uses Ollama to generate intelligent, context-aware responses.
.
Features
   - Persistent Chat: Interactive bot with history management and model switching (/model).
   - Admin & Info Commands: 
      - /help: Get a list of available commands and usage instructions.
      - /stats: View current session statistics (e.g., history size, active model).
   - Smart Reminders: Scheduled prompts sent via systemd timers.
   - Integrated Memory_ integrates user profile into chat_bot context window
   - Robust Logging: Automatic log rotation in a dedicated /logs directory.

Typical use cases:
   - motivational messages
   - reminders
   - random thoughts

## Project Structure
```text
.
├── bot/
│   ├── chat_bot.py        # Main interactive bot
│   ├── reminder.py        # Scheduled reminder script
│   ├── memory_service.py  # Periodic user profile updates
│   └── utils.py           # Shared utilities (safe JSON loading, file locks)
├── data/                  # JSON data files (ignored by git)
├── logs/                  # Log files (ignored by git)
├── systemd/               # Service & Timer templates
├── .env                   # Local configuration  (API keys, etc.)
├── .gitignore             # Keeps your secrets safe
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # Project documentation

```
## Prerequisites

- A running [Ollama](https://ollama.com) instance
- A Telegram bot token (create one via [@BotFather](https://t.me/BotFather))
- Your Telegram chat ID (get it via [@userinfobot](https://t.me/userinfobot))

## Configuration

Create a `.env` file with the following variables:

```env
OLLAMA_GENERATE_URL=http://localhost:11434/api/generate
OLLAMA_CHAT_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.2:1b
OLLAMA_PROMPT=Send a short, motivating message.
MAX_HISTORY=20
TELEGRAM_TOKEN=your_bot_token_here # Get token from https://t.me/BotFather
TELEGRAM_CHAT_ID=your_chat_id_here # Get your ID from https://t.me/userinfobot
```

## Setup

Clone and Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Automation with systemd

For scheduled execution on Linux systems, you can use the provided systemd service and timer files.

1. The Chat Bot (Always On)

```bash
sudo cp systemd/ollama_chat.service.example /etc/systemd/system/ollama_chat.service
# Edit paths in /etc/systemd/system/ollama_chat.service
sudo systemctl enable --now ollama_chat.service
```

2. The Reminder (Scheduled)

```bash
sudo cp systemd/ollama_reminder.service.example /etc/systemd/system/ollama_reminder.service
sudo cp systemd/ollama_reminder.timer.example /etc/systemd/system/ollama_reminder.timer
# Edit paths in both files
sudo systemctl enable --now ollama_reminder.timer
```

3. The Memory (Scheduled)

```bash
sudo cp systemd/ollama_memory.service.example /etc/systemd/system/ollama_memory.service
sudo cp systemd/ollama_memory.timer.example /etc/systemd/system/ollama_memory.timer
# Edit paths in both files
sudo systemctl enable --now ollama_memory.timer
```


### Operations & Maintenance

**Check status:**
To see if your timer is active and when the next reminder will trigger:
```bash
systemctl status ollama_reminder.timer
systemctl list-timers --all | grep ollama
```

**Manual Test:**
Trigger the reminder immediately without waiting for the timer:
```bash
sudo systemctl start ollama_reminder.service
```

**View live logs:**
Since we use a dedicated logs/ directory, you can watch the files directly:
```bash
tail -f logs/bot.log
tail -f logs/reminder.log
tail -f logs/memory_service.log
```

Or use the systemd journal:
```bash
journalctl -u ollama_chat.service -f
```

### Customizing the Schedule

The default timer runs every 2 hours between 08:00 and 00:00. To change this, edit the `OnCalendar` line in the timer file:
```ini
# Daily at 08:00
OnCalendar=*-*-* 08:00:00

# Every hour
OnCalendar=hourly

# Multiple times per day
OnCalendar=*-*-* 09:00,12:00,18:00:00
```

Test your calendar expression:
```bash
systemd-analyze calendar "*-*-* 08,10,12,14,16,18,20,22,00:00:00"
```

For more information on systemd timer syntax, see `man systemd.time`.

## Future Ideas / To-Dos

    [x] Shared utility module for safe JSON loading
    [x] File locking to prevent race conditions between bot and reminder
    [x] User memory: periodic profile updates via memory_service.py
    [x] Memory: integrate user profile into chat_bot context window

    [ ] Database Migration: Replace the JSON-based history storage with SQLite to prevent O(n) search operations and memory bottlenecks as the history grows.
    [ ] Data Retention Policy: Implement an automated cleanup logic to prune or archive chat logs older than 30 days.
    [ ] Context Window Management: 
      [ ] Implement Token-based history slicing for more precise context control.
      [ ] Conversation Summarization: Automatically generate a summary of older messages when the context limit is reached, preserving long-term "memory" while keeping the processing load low.

    [ ] Unit Tests: Add pytest for core functions (e.g., `ask_ollama`).
    [ ] Dockerfile: Containerize the bot for easy deployment.

    [ ] Multi-User Support: Allow multiple Chat IDs in a whitelist.
    [ ] Webhook Support: Replace polling with Telegram webhooks for better scalability (advanced).
    [ ] Secret Management: Integration with Bitwarden CLI.


Note: Dual API usage is a design decision: /api/generate is used for standalone tasks where a full conversation context is unnecessary, reducing overhead; api/chat is used for the main conversation flow to leverage structured message history and role-based prompting.

