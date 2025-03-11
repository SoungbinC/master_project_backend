from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO

app = FastAPI()

# Define label mappings
label2id = {"glaucoma": 0, "cataract": 1, "normal": 2, "diabetic_retinopathy": 3}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping


# Define input model
class ScanInput(BaseModel):
    scan_id: int
    scan_url: str


# Load model at startup
print("🚀 Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Sequential(nn.Linear(512, len(label2id)))  # Match output classes
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


async def process_image_from_url(image_url):
    """Download and preprocess the image."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        return image
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None


# 🔹 Allow GET and POST requests for `/predict/`
@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(request: Request, data: ScanInput = None):
    """Prediction endpoint supporting both GET & POST"""
    try:
        # Log request method
        print(f"📥 Received {request.method} request to /predict/")

        # GET method: Extract parameters from URL
        if request.method == "GET":
            scan_id = request.query_params.get("scan_id")
            scan_url = request.query_params.get("scan_url")
            if not scan_id or not scan_url:
                raise HTTPException(
                    status_code=400, detail="Missing scan_id or scan_url"
                )
            scan_id = int(scan_id)

        # POST method: Use JSON body
        elif request.method == "POST" and data:
            scan_id = data.scan_id
            scan_url = data.scan_url
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Process image
        image_data = await process_image_from_url(scan_url)
        if image_data is None:
            raise HTTPException(status_code=400, detail="Invalid image URL")

        # Predict
        with torch.no_grad():
            logits = model(image_data)
            pred_class_id = torch.argmax(logits, axis=1).cpu().item()
            pred_class_name = id2label.get(pred_class_id, "Unknown")

        print(f"✅ Prediction: {pred_class_name}")
        return {"scan_id": scan_id, "classification": pred_class_name}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}


# Root endpoint for testing
@app.get("/")
async def home():
    return {"message": "Eye Disease Classification Model API is Running!"}
