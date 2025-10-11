import os
import io
from io import BytesIO
import random
import torch
import timm
import numpy as np
import cv2
import requests
from PIL import Image
from torchvision import transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import logging
import tempfile

# --- Logging Setup ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment and Models ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# (This section is simplified as the bot will now call the API)

# --- Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome! Send me an image, video, or video URL to analyze for deepfakes.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You can send an image file, a video file, or a URL to a video (e.g., from YouTube, Twitter).")

# ✅ --- THIS FUNCTION CONTAINS YOUR CUSTOM LOGIC --- ✅
def format_response(data: dict, analysis_type: str):
    """
    Formats the JSON response from the API into a readable string,
    applying the custom prediction and score-swapping logic.
    """
    # 1. Get raw confidence scores as percentages
    real_confidence = data['confidence']['real'] * 100
    manipulated_confidence = data['confidence']['manipulated'] * 100

    # 2. Determine the prediction based on the > 20% threshold
    prediction = "Real"
    if manipulated_confidence > 20:
        prediction = "Manipulated"
        
    # 3. Apply your swap logic for display purposes
    if manipulated_confidence < 50 and prediction == "Manipulated":
        # Pythonic way to swap two variables
        real_confidence, manipulated_confidence = manipulated_confidence, real_confidence

    # 4. Return the formatted string with the final values
    return (
        f"🤖 **{analysis_type} Analysis Result**:\n"
        f"Status: *{prediction}*\n"
        f"Confidence:\n"
        f"  - Real: {real_confidence:.2f}%\n"
        f"  - Manipulated: {manipulated_confidence:.2f}%"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received image from user.")
    processing_message = await update.message.reply_text("🔄 Analyzing image...")
    
    photo = await update.message.photo[-1].get_file()
    photo_bytes = await photo.download_as_bytearray()
    
    try:
        files = {'file': ('image.jpg', photo_bytes, 'image/jpeg')}
        response = requests.post("http://localhost:8000/predict-image/", files=files)
        
        if response.status_code == 200:
            await processing_message.edit_text(format_response(response.json(), "Image"), parse_mode='Markdown')
        else:
            await processing_message.edit_text(f"Error from server: {response.text}")
    except requests.exceptions.ConnectionError:
        await processing_message.edit_text("❌ Error: Could not connect to the local analysis server.")
    except Exception as e:
        await processing_message.edit_text(f"An error occurred: {e}")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received video from user.")
    processing_message = await update.message.reply_text("🔄 Analyzing video... (this may take a moment)")

    video_file = await update.message.video.get_file()
    video_bytes = await video_file.download_as_bytearray()

    try:
        files = {'file': (video_file.file_path, video_bytes, 'video/mp4')}
        response = requests.post("http://localhost:8000/predict-video-from-upload/", files=files, timeout=120)

        if response.status_code == 200:
            await processing_message.edit_text(format_response(response.json(), "Video"), parse_mode='Markdown')
        else:
            await processing_message.edit_text(f"Error from server: {response.text}")
    except requests.exceptions.ConnectionError:
        await processing_message.edit_text("❌ Error: Could not connect to the local analysis server.")
    except Exception as e:
        await processing_message.edit_text(f"An error occurred: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_text = update.message.text
    if message_text.startswith('http'):
        logger.info(f"Received URL from user: {message_text}")
        processing_message = await update.message.reply_text(f"🔄 Analyzing video from URL...")
        
        try:
            # Note: The FastAPI backend expects JSON with a key 'url'
            response = requests.post("http://localhost:8000/predict-video-from-url/", json={"url": message_text}, timeout=180)

            if response.status_code == 200:
                await processing_message.edit_text(format_response(response.json(), "Video"), parse_mode='Markdown')
            else:
                await processing_message.edit_text(f"Error from server: {response.text}")
        except requests.exceptions.ConnectionError:
            await processing_message.edit_text("❌ Error: Could not connect to the local analysis server.")
        except Exception as e:
            await processing_message.edit_text(f"An error occurred: {e}")
    else:
        await update.message.reply_text("Please send an image, video file, or a video URL.")

def main():
    if not TELEGRAM_TOKEN:
        logger.critical("FATAL: TELEGRAM_TOKEN not found in .env file.")
        return
        
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("🤖 Starting bot with full media support...")
    application.run_polling()

if __name__ == "__main__":
    main()