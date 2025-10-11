from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import timm
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np
import io
import random
import os

# --- Load Content/Label Detection Model Locally ---
try:
    detection_model = fasterrcnn_resnet50_fpn(weights=None)
    detection_model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
    detection_model.eval()
    print("Successfully loaded local object detection model.")
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
except Exception as e:
    print(f"Could not load detection model: {e}")
    detection_model = None

def get_content_labels(image: Image.Image, threshold=0.7):
    if not detection_model: return []
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)
        with torch.no_grad():
            prediction = detection_model([img_tensor])[0]
        labels = []
        for i in range(len(prediction['labels'])):
            score = prediction['scores'][i].item()
            if score > threshold:
                label_name = COCO_INSTANCE_CATEGORY_NAMES[prediction['labels'][i].item()]
                if label_name not in [l['description'] for l in labels]:
                     labels.append({'description': label_name, 'confidence': score})
        return labels[:5]
    except Exception as e:
        print(f"Error in content analysis: {str(e)}")
        return []

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("mvitv2_base_cls", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("best_vit_model.pth", map_location=device))
model.to(device)
model.eval()

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
MAX_FILE_SIZE = 50 * 1024 * 1024

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API (Free Version) is running"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # *** THE FIX IS HERE ***
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # Changed from COLOR_RGB_BGR
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_pil = image
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_cv = img_cv[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_cv, cv2.COLOR_BGR2RGB))
        input_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
        labels = get_content_labels(image)
        result = {
            "prediction": "Real" if predicted_class == 1 else "Manipulated",
            "confidence": {
                "real": float(probs[1].item()),
                "manipulated": float(probs[0].item())
            },
            "face_detected": len(faces) > 0,
            "labels": labels
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))