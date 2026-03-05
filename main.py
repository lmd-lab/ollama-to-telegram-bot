import os
import requests
import datetime
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("OLLAMA_MODEL")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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

PROMPT = get_prompt()

print(PROMPT)

def ask_ollama(prompt: str) -> str:
    """
    Asks the model for a response based on the provided prompt.
    
    :param prompt: The prompt to send to the model.
    :type prompt: str
    :return: The response from the model or an error message.
    :rtype: str
    """
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60) # maybe too short for some models

        print(response.json())
        print(response.json()["response"])

        response.raise_for_status()
        data = response.json()

        answer = data.get("response", "").strip()
        if not answer or not answer.strip():
            return "Error: Model returned an empty response."

        return answer

    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out (60s)."
    except requests.exceptions.RequestException as e:
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
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram Error: {data}")


def main():
    answer = ask_ollama(PROMPT)

    if answer.startswith("Error:"):
        print(f"Ollama issue detected: {answer}")

    try:
        send_telegram_message(answer)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


if __name__ == "__main__":
    main()
