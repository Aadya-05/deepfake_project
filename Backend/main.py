from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import io
import random
import os
import tempfile
import yt_dlp

# --- Load a SINGLE Model for Both Image and Video ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We will use the MViTv2 model for both images and video frames
model = timm.create_model("mvitv2_base_cls", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("best_vit_model.pth", map_location=device))
model.to(device)
model.eval()
print("Single deepfake model (MViTv2) loaded for all tasks.")


# --- Helper Functions and App Setup ---
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Image transforms for the MViTv2 model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_video_frames(video_path: str):
    """Helper function to analyze video frames using the SINGLE image model."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1: raise ValueError("Video file is empty.")
        
        n_frames_to_analyze = 30
        frame_indices = np.linspace(0, frame_count - 1, min(n_frames_to_analyze, frame_count), dtype=int)
        predictions = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: continue

            faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
            if not len(faces): continue

            x, y, w, h = faces[0]
            face_frame = frame[y:y+h, x:x+w]
            
            face_image = Image.fromarray(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor) # Use the single 'model'
                probs = torch.softmax(output, dim=1)[0]
                # Index 1 is 'Real' for this model, so we want the 'Manipulated' prob at index 0
                predictions.append(probs[0].item())

        cap.release()
    except Exception as e:
        print(f"Error during video frame analysis: {e}")
        return None
    if not predictions: return None

    avg_fake_prob = np.mean(predictions)
    return {
        "prediction": "Manipulated" if avg_fake_prob > 0.5 else "Real",
        "confidence": {"real": 1.0 - avg_fake_prob, "manipulated": avg_fake_prob}
    }

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Deepfake Detection API (Simplified - Single Model) is running"}

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_cascade.detectMultiScale(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), 1.1, 4)
        face_pil = image
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_pil = Image.fromarray(cv2.cvtColor(img_cv[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
        
        input_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor) # Use the single 'model'
            probs = torch.softmax(output, dim=1)[0]
        return {
            "prediction": "Real" if torch.argmax(probs).item() == 1 else "Manipulated",
            "confidence": {"real": float(probs[1].item()), "manipulated": float(probs[0].item())},
            "face_detected": len(faces) > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-video-from-upload/")
async def predict_video_from_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name
    result = analyze_video_frames(video_path)
    os.remove(video_path)
    if result is None:
        raise HTTPException(status_code=400, detail="Could not process video. No faces detected or file is invalid.")
    return result

@app.post("/predict-video-from-url/")
async def predict_video_from_url(url: str = Body(..., embed=True)):
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s')}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download video from URL: {e}")
    result = analyze_video_frames(video_path)
    if os.path.exists(video_path):
        os.remove(video_path)
    if result is None:
        raise HTTPException(status_code=400, detail="Could not process video from URL. No faces detected or link is invalid.")
    return result