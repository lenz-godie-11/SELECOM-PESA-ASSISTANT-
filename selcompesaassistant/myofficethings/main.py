from telegram.ext import ApplicationBuilder
# from Utils.config import TELEGRAM_BOT_TOKEN
from bot.handlers import get_handlers
import os
from dotenv import load_dotenv

load_dotenv()

# TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def main():
    app = ApplicationBuilder().token("8439230157:AAENO3WwcZ6H5ee8Fx2OiMOX1o1AmRC98Gs").build()
    
    for handler in get_handlers():
        app.add_handler(handler)
    
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()