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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

try:
    detection_model = fasterrcnn_resnet50_fpn(weights=None)
    detection_model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
    detection_model.eval()
    print("Successfully loaded local object detection model for bot.")
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
    print(f"Could not load detection model for bot: {e}")
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

class FreeReverseImageSearch:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}
    async def search(self, image: Image.Image):
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            upload_url = "http://images.google.com/searchbyimage/upload"
            multipart = {'encoded_image': ('image.jpg', img_bytes), 'image_content': ''}
            response = requests.post(upload_url, files=multipart, allow_redirects=False, timeout=10)
            search_url = response.headers.get("Location")
            if not search_url: return {'error': 'Could not get search URL from Google.'}
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            page_matches = []
            for a_tag in soup.find_all('a', href=True):
                if a_tag.find('h3') and '/url?q=' in a_tag['href']:
                    url = a_tag['href'].split('/url?q=')[1].split('&')[0]
                    domain = url.split('//')[-1].split('/')[0]
                    if url not in [p['url'] for p in page_matches]:
                        page_matches.append({'url': url, 'domain': domain})
            return {'page_matches': page_matches[:5]}
        except Exception as e:
            print(f"Error in reverse image search: {str(e)}")
            return {'page_matches': []}

class DeepfakeDetector:
    def __init__(self, model_path="best_vit_model.pth"):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("mvitv2_base_cls", pretrained=False, num_classes=2)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {e}")
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def detect_faces(self, img_cv):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
    def predict(self, image):
        # *** THE FIX IS HERE ***
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # Changed from COLOR_RGB_BGR

        faces = self.detect_faces(img_cv)
        face_pil = image
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_cv = img_cv[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_cv, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
        return {"prediction": "Real" if predicted_class == 1 else "Manipulated", "confidence": {"real": float(probs[1].item()),"manipulated": float(probs[0].item())}, "face_detected": len(faces) > 0, "faces_count": len(faces)}

try:
    deepfake_detector = DeepfakeDetector()
    reverse_search = FreeReverseImageSearch()
except Exception as e:
    print(f"❌ Error initializing services: {str(e)}")
    exit()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome! Send an image to check if it's a deepfake.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send any image to analyze it.")
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    processing_message = await update.message.reply_text("🔄 Analyzing image... Please wait.")
    try:
        photo = await update.message.photo[-1].get_file()
        photo_bytes = await photo.download_as_bytearray()
        image = Image.open(BytesIO(photo_bytes)).convert("RGB")
        await processing_message.edit_text("🔍 Running deepfake check...")
        deepfake_result = deepfake_detector.predict(image)
        await processing_message.edit_text("🏷️ Identifying image content...")
        labels = get_content_labels(image)
        await processing_message.edit_text("🌐 Searching for image online...")
        search_result = await reverse_search.search(image)
        response = f"🤖 **Deepfake Analysis**:\n" \
                   f"Status: *{deepfake_result['prediction']}*\n" \
                   f"Confidence:\n" \
                   f"  - Real: {deepfake_result['confidence']['real']*100:.2f}%\n" \
                   f"  - Manipulated: {deepfake_result['confidence']['manipulated']*100:.2f}%\n"
        if deepfake_result['face_detected']: response += f"\n👤 **Face Analysis**:\nFaces Detected: {deepfake_result['faces_count']}\n"
        if labels:
            response += f"\n🖼️ **Image Content**:\n"
            for label in labels: response += f"- {label['description']} ({label['confidence']*100:.1f}%)\n"
        if search_result and search_result.get('page_matches'):
            response += f"\n🔗 **Possible Online Sources**:\n"
            for match in search_result['page_matches']: response += f"- [{match['domain']}]({match['url']})\n"
        else:
            response += "\n\n_No matching images found online._"
        await processing_message.edit_text(response, parse_mode='Markdown')
    except Exception as e:
        print(f"Error in handle_image: {e}")
        await processing_message.edit_text(f"❌ An error occurred during analysis.")

def main():
    if not TELEGRAM_TOKEN:
        print("❌ Error: TELEGRAM_TOKEN not found in .env file.")
        return
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    print("🤖 Starting Free Deepfake Detection Bot...")
    application.run_polling()

if __name__ == "__main__":
    main()