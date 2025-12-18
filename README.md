🧠 Deepfake Image & Video Analyzer

AI-Powered Media Forensics System using Vision Transformers

📌 Overview

The Deepfake Image & Video Analyzer is an end-to-end AI-driven media authenticity detection system designed to identify manipulated or synthetically generated content in images and videos.

The project combines a Vision Transformer–based deepfake detection model (MViTv2) with a production-grade backend and multiple real-world interfaces (Chrome Extension and Telegram Bot), demonstrating both machine learning depth and system engineering capability.

This system is built for scalability, real-time inference, and cross-platform accessibility, making it suitable for cybersecurity, misinformation detection, and digital forensics use cases.

🎯 Key AI/ML Objectives

Detect facial manipulations and synthetic media

Perform frame-level video inference

Apply face-centric analysis to reduce noise

Serve ML inference via low-latency APIs

Integrate AI analysis into real user workflows

🚀 Features
🔍 Multi-Modal Media Analysis

Image files (.jpg, .png)

Uploaded video files (.mp4, .avi)

Video URLs (YouTube, Twitter/X, etc.)

🧠 Deep Learning Core

Model: Multiscale Vision Transformer v2 (MViTv2)

Framework: PyTorch + TIMM

Inference Strategy:

Face-level analysis

Frame sampling for videos

Confidence-based manipulation scoring

🎯 Face-Focused Detection

Uses OpenCV Haar Cascades

Automatically detects and isolates faces

Ensures the model focuses only on relevant facial regions

⚡ FastAPI ML Backend

Central inference engine

Handles:

Media preprocessing

Model inference

Confidence scoring

API responses for multiple clients

🌐 Chrome Extension

Side-panel based UI

Analyze images and videos directly while browsing

Designed for real-time misinformation checks

🤖 Telegram Bot

Mobile-friendly AI interface

Supports:

Media upload & URL analysis

Reverse image search (Google Lens via SerpAPI)

Metadata & content insights (Google Vision API)

🧩 System Architecture
User (Web / Chrome / Telegram)
        |
        v
FastAPI Inference Server
        |
        v
Face Detection (OpenCV)
        |
        v
MViTv2 Deepfake Model (PyTorch)
        |
        v
Confidence Scoring & Explainable Output

🛠️ Project Structure
Deepfake-Analyzer/
│
├── Backend/
│   ├── main.py              # FastAPI inference server
│   ├── deepfake_bot.py      # Telegram bot logic
│   ├── telegram_bot.py
│   ├── best_vit_model.pth   # Pretrained MViTv2 model
│   └── utils/               # Preprocessing & helpers
│
├── extension/
│   ├── manifest.json
│   ├── background.js
│   ├── sidepanel.html
│   └── sidepanel.js
│
├── requirements.txt
└── README.md

⚙️ Setup & Installation
1️⃣ Backend (ML Inference Server)
cd Backend
pip install -r requirements.txt


Place the pretrained model:

Backend/best_vit_model.pth


Create .env:

TELEGRAM_TOKEN=your_bot_token
GOOGLE_VISION_API_KEY=your_key
SERP_API_KEY=your_key


Run the server:

uvicorn main:app --reload

2️⃣ Chrome Extension

Open chrome://extensions/

Enable Developer Mode

Click Load Unpacked

Select the /extension folder

The analyzer opens via the extension icon or side panel.

3️⃣ Telegram Bot
python deepfake_bot.py


Search for your bot on Telegram and start analyzing media instantly.

📊 Detection Logic (AI Explanation)

The model outputs a Manipulation Confidence Score

Thresholding Strategy:

> 20% → Flagged as Potentially Manipulated

20%–50% → UI swap logic applied for clarity

This avoids false certainty while maintaining interpretable results

This design prioritizes responsible AI output over binary classification.

🧪 ML Stack & Dependencies
Core ML & Vision

PyTorch

TIMM

OpenCV

Backend & Integration

FastAPI

Uvicorn

yt-dlp (video extraction)

External AI Services

Google Vision API

SerpAPI (Reverse Image Search)

Bot & Extension

python-telegram-bot

Chrome Extensions API

📈 Use Cases

Deepfake detection & misinformation analysis

Social media content verification

Cybersecurity & digital forensics

Journalism & fact-checking tools

AI safety & responsible ML research

🧠 Skills Demonstrated

Vision Transformers (ViTs)

Deepfake detection pipelines

ML model deployment (FastAPI)

Face detection & video processing

API-driven ML systems

Cross-platform AI integration

Responsible AI confidence handling
