from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import torch
import torchvision
import torch.nn as nn
import torchmetrics
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO

app = FastAPI()

# 🔹 Label mappings
label2id = {"glaucoma": 0, "cataract": 1, "normal": 2, "diabetic_retinopathy": 3}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping


# 🔹 Input model for request validation
class ScanInput(BaseModel):
    scan_id: int
    scan_url: str


# ✅ Define model architecture to match `Trainer`
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)

        # 🔹 Freeze all layers except last 15
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        # 🔹 Keep `block` layer as in Trainer
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # 4 classes
        )

        # 🔹 Do not remove `fc`, just replace it
        self.base.fc = nn.Sequential()

    def forward(self, x):
        x = self.base(x)  # 🔹 Keep the structure matching `Trainer`
        x = self.block(x)
        return x


# 🚀 Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

try:
    print("🚀 Loading model...")
    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(
        checkpoint, strict=False
    )  # 🔹 Use strict=False to handle minor mismatches
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 Image preprocessing
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


# 🔹 API Root - Health Check
@app.api_route("/", methods=["GET", "HEAD"])
async def home():
    return {"message": "Eye Disease Classification Model API is Running!"}
    # 🔹 Health check endpoint


# ✅ Prediction Endpoint - Supports Both `GET` and `POST`
@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(request: Request, data: ScanInput = None):
    """Handles both GET and POST requests for prediction"""
    try:
        print(f"📥 Received {request.method} request to /predict/")

        # Handle GET request with URL parameters
        if request.method == "GET":
            scan_id = request.query_params.get("scan_id")
            scan_url = request.query_params.get("scan_url")
            if not scan_id or not scan_url:
                raise HTTPException(
                    status_code=400, detail="Missing scan_id or scan_url"
                )
            scan_id = int(scan_id)

        # Handle POST request with JSON body
        elif request.method == "POST" and data:
            scan_id = data.scan_id
            scan_url = data.scan_url
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Process image
        image_data = await process_image_from_url(scan_url)
        if image_data is None:
            raise HTTPException(status_code=400, detail="Invalid image URL")

        # Model inference
        with torch.no_grad():
            logits = model(image_data)
            pred_class_id = torch.argmax(logits, axis=1).cpu().item()
            pred_class_name = id2label.get(pred_class_id, "Unknown")

        print(f"✅ Prediction: {pred_class_name}")
        return {"scan_id": scan_id, "classification": pred_class_name}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}
