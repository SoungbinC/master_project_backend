from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO

app = FastAPI()

# 🔹 Updated Label Mappings with Full Names
label2id = {
    "Normal": 0,
    "Diabetic Retinopathy": 1,
    "Glaucoma": 2,
    "Cataract": 3,
    "AMD (Age-related Macular Degeneration)": 4,
    "Hypertension": 5,
    "Myopia": 6,
    "Other Abnormalities": 7,
}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping


# 🔹 Input model for request validation
class ScanInput(BaseModel):
    scan_id: int
    scan_url: str


# ✅ Define `ResNet50` Model (Matches `best_model.pth`)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 🔹 Freeze early layers (as done in training)
        for param in list(self.base.parameters())[:-20]:
            param.requires_grad = False

        # 🔹 Modify the final FC layer
        num_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 8),  # 8 classes
            nn.Sigmoid(),  # Sigmoid for multi-label classification
        )

    def forward(self, x):
        return self.base(x)


# 🚀 Load `best_model.pth` at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

try:
    print("🚀 Loading best_model.pth...")
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(
        checkpoint, strict=False
    )  # 🔹 Use strict=False to handle minor mismatches
    model.eval()
    print("✅ best_model.pth loaded successfully!")
except Exception as e:
    print(f"❌ Error loading best_model.pth: {e}")

# 🔹 Image preprocessing (matches training pipeline)
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
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
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
