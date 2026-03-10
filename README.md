# Ollama Telegram Bot & Reminder

A Python-based Telegram assistant that combines a 24/7 Chat Bot and a Scheduled Reminder. It uses Ollama to generate intelligent, context-aware responses.
.
Features
   - Persistent Chat: Interactive bot with history management and model switching (/model).
   - Admin & Info Commands: 
      - /help: Get a list of available commands and usage instructions.
      - /stats: View current session statistics (e.g., history size, active model).
   - Smart Reminders: Scheduled prompts sent via systemd timers.
   - Robust Logging: Automatic log rotation in a dedicated /logs directory.

Typical use cases:
   - motivational messages
   - reminders
   - random thoughts

## Project Structure

.
├── bot/
│   ├── chat_bot.py      # Main interactive bot
│   └── reminder.py      # Scheduled reminder script
├── logs/                # Local log files (ignored by git)
├── .env                 # Local configuration
├── .gitignore           # Keeps your secrets safe
└── systemd/             # Service & Timer templates

## Prerequisites

- A running [Ollama](https://ollama.com) instance
- A Telegram bot token (create one via [@BotFather](https://t.me/BotFather))
- Your Telegram chat ID (get it via [@userinfobot](https://t.me/userinfobot))

## Configuration

Create a `.env` file with the following variables:

```env
OLLAMA_URL=http://localhost:11434/api/generate
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
python main.py
```

# Automation with systemd

For scheduled execution on Linux systems, you can use the provided systemd service and timer files.

1. The Chat Bot (Always On)

```bash
sudo cp systemd/ollama_telegram_bot.service.example /etc/systemd/system/ollama_telegram_bot.service
# Edit paths in /etc/systemd/system/ollama_telegram_bot.service
sudo systemctl enable --now ollama_telegram_bot.service
```

2. The Reminder (Scheduled)

```bash
sudo cp systemd/ollama_reminder.service.example /etc/systemd/system/ollama_reminder.service
sudo cp systemd/ollama_reminder.timer.example /etc/systemd/system/ollama_reminder.timer
# Edit paths in both files
sudo systemctl enable --now ollama_reminder.timer
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
sudo systemctl start ollama_telegram_bot.service
```

**View live logs:**
Since we use a dedicated logs/ directory, you can watch the files directly:
```bash
tail -f logs/bot.log
tail -f logs/generate_script.log
```

Or use the systemd journal:
```bash
journalctl -u ollama_telegram_bot.service -f
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

## Future Ideas / To-Do

    [ ] Switch to /api/chat instead of using both endpoints (/api/generate and /api/chat)
    - [ ] Webhook Support: Replace polling with Telegram webhooks for better scalability (advanced).
- [ ] Unit Tests: Add pytest for core functions (e.g., `ask_ollama`).
- [ ] Dockerfile: Containerize the bot for easy deployment.
    [ ] Multi-User Support: Allow multiple Chat IDs in a whitelist.
    [ ] Secret Management: Integration with Bitwarden CLI.