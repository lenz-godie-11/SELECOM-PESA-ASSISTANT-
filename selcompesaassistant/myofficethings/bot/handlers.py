import io
# from telegram.ext import Update
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters
from services import ai_services, speech_service
from services.ai_services import get_structured_ai_response, clear_conversation_memory

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Karibu! Tuma ujumbe wa sauti au maandishi!")

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear the conversation memory for the user."""
    user_id = update.effective_user.id
    clear_conversation_memory(user_id)
    await update.message.reply_text("Conversation memory cleared! Starting fresh.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    voice_file = await update.message.voice.get_file()
    ogg_io = io.BytesIO()
    await voice_file.download_to_memory(out=ogg_io)
    
    wav_io = speech_service.convert_ogg_to_wav(ogg_io)
    text = speech_service.transcribe_audio(wav_io)
    
    response = await get_structured_ai_response(user_id, text)
    await update.message.reply_text(f"Ulisema: {text}\n\nJibu: {format_response(response)}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    response = await get_structured_ai_response(user_id, update.message.text)
    await update.message.reply_text(format_response(response))

def format_response(response):
    """Format the structured response into a human-readable message."""
    parts = []
    
    # Add greeting if present
    if response.greeting:
        parts.append(response.greeting)
    
    # Add main response
    if response.main_response:
        parts.append(response.main_response)
    
    # Add additional info if present
    if response.additional_info:
        parts.append(f"\n💡 {response.additional_info}")
    
    # Add contact info if present
    if response.contact_info:
        parts.append(f"\n📞 {response.contact_info}")
    
    # Add next steps if present
    if response.next_steps:
        parts.append(f"\n➡️ {response.next_steps}")
    
    return "\n".join(parts)

def get_handlers():
    return [
        CommandHandler('start', start),
        CommandHandler('clear', clear_memory),
        MessageHandler(filters.VOICE, handle_voice),
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    ]