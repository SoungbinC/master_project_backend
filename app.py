from fastapi import FastAPI
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import transforms

app = FastAPI()

# Define label mappings
label2id = {"glaucoma": 0, "cataract": 1, "normal": 2, "diabetic_retinopathy": 3}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping


# Define input format
class ScanInput(BaseModel):
    scan_id: int
    scan_url: str


# Define model architecture (must match trained model)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)

        # Freeze some layers
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(label2id)),  # Use len(label2id) to match class count
        )
        self.base.fc = nn.Sequential()

    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

try:
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# Define preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
    ]
)


def process_image_from_url(image_url):
    """Download image and preprocess it for the model."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error if download fails

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        return image
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None


@app.get("/")
def home():
    """Home Route"""
    return {"message": "Eye Disease Classification Model API is Running!"}


@app.post("/predict/")
def predict_post(data: ScanInput):
    """POST request for prediction using JSON input"""
    try:
        image_data = process_image_from_url(data.scan_url)
        if image_data is None:
            return {"error": "Invalid URL or failed to download image."}

        with torch.no_grad():
            logits = model(image_data)
            pred_class_id = torch.argmax(logits, axis=1).cpu().item()
            pred_class_name = id2label.get(pred_class_id, "Unknown")

        return {"scan_id": data.scan_id, "classification": pred_class_name}

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return {"error": "Internal Server Error"}
