Deepfake Image and Video Analyzer
This project is a comprehensive deepfake detection system featuring a FastAPI backend, a Chrome Extension, and a Telegram Bot. It uses the MViTv2 (Multiscale Vision Transformer) model to analyze images and video frames for potential manipulations.

🚀 Features
Multi-Format Support: Analyze local images, uploaded video files, and videos from URLs (YouTube, Twitter, etc.).

FastAPI Backend: A high-performance API that serves as the central analysis engine for both the extension and the bot.

Chrome Extension: A side-panel interface for quick analysis of media while browsing.

Telegram Bot: A mobile-friendly bot that provides deepfake analysis, reverse image search (via Google Lens/SerpAPI), and content insights using Google Vision API.

Face Detection: Automatically isolates faces in media using OpenCV Haar Cascades to ensure the analysis focuses on relevant subjects.

🛠️ Project Structure
/Backend: Contains the FastAPI server (main.py), the Telegram bot scripts (deepfake_bot.py, telegram_bot.py), and the AI model requirements.

/extension: Contains the Chrome Extension files, including the side-panel UI and background service workers.

requirements.txt: Lists necessary Python dependencies for the environment.

⚙️ Setup and Installation
1. Backend Setup
Navigate to the /Backend directory.

Install dependencies:

Bash

pip install -r requirements.txt
Ensure the pre-trained model file best_vit_model.pth is placed in the /Backend folder.

Create a .env file with your credentials:

Code snippet

TELEGRAM_TOKEN=your_bot_token
GOOGLE_VISION_API_KEY=your_key
SERP_API_KEY=your_key
Run the API server:

Bash

uvicorn main:app --reload
2. Chrome Extension Setup
Open Chrome and navigate to chrome://extensions/.

Enable Developer mode.

Click Load unpacked and select the /extension folder.

The analyzer can be opened via the extension icon or side panel.

3. Telegram Bot
Run the bot script:

Bash

python deepfake_bot.py
Search for your bot on Telegram and start the conversation to begin analyzing media.

📊 Analysis Logic
The system uses a custom threshold for detection. If the "Manipulated" confidence score exceeds 20%, the media is flagged as manipulated. For display purposes, a "swap logic" is applied if the manipulated confidence is between 20% and 50% to ensure the results are presented clearly to the user.

📝 Dependencies
Key libraries used in this project include:

Torch & Timm: For the MViTv2 vision transformer model.

OpenCV: For face detection and video frame processing.

FastAPI: For the web server.

yt-dlp: For downloading and processing video URLs.

python-telegram-bot: For the bot interface.
