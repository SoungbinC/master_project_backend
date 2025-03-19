from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO

# 🚀 Initialize FastAPI
app = FastAPI()

# 🔹 Label Mappings (4 Classes from `model.pth`)
label2id = {
    "glaucoma": 0,
    "cataract": 1,
    "normal": 2,
    "diabetic_retinopathy": 3,
}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping


# ✅ Input model for request validation
class ScanInput(BaseModel):
    scan_id: int
    scan_url: str


# ✅ Define `ResNet18` Model for `model.pth`
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)

        # 🔹 Freeze early layers (matches training config)
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        # 🔹 Classification Head (Matches `model.pth`)
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # ✅ Corrected: 4 classes
        )
        self.base.fc = nn.Identity()  # Remove ResNet's FC layer

    def forward(self, x):
        x = self.base(x)  # Extract features
        x = self.block(x)  # Pass through custom classifier
        return x


# 🚀 Load `model.pth` at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

try:
    print("🚀 Loading model.pth...")
    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # Allow minor mismatches
    model.eval()
    print("✅ model.pth loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model.pth: {e}")

# 🔹 Image preprocessing (same as training)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


@app.get("/")
async def root():
    return {"message": "🚀 Eye Disease Classification API is Running!"}


@app.api_route("/predict", methods=["GET", "POST"])
@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(request: Request, data: ScanInput = None):
    """Handles both GET and POST requests for prediction"""
    try:
        if request.method == "GET":
            scan_id = request.query_params.get("scan_id")
            scan_url = request.query_params.get("scan_url")
            if not scan_id or not scan_url:
                raise HTTPException(
                    status_code=400, detail="Missing scan_id or scan_url"
                )
            scan_id = int(scan_id)

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
            probabilities = (
                torch.softmax(logits, dim=1).cpu().numpy().flatten()
            )  # ✅ Softmax
            pred_class_id = np.argmax(probabilities)
            pred_class_name = id2label.get(pred_class_id, "Unknown")
            class_probs = {
                id2label[i]: round(float(probabilities[i]), 4)
                for i in range(len(probabilities))
            }

        return {
            "scan_id": scan_id,
            "classification": pred_class_name,
            "probabilities": class_probs,
        }

    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}
