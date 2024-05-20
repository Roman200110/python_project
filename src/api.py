from fastapi import FastAPI, File, UploadFile
from src.model import AnimalClassifier
from PIL import Image
import torch
import io
import torchvision.transforms as transforms  # Add this line

app = FastAPI()

model = AnimalClassifier(num_classes=4)
model.load_state_dict(torch.load("path/to/model_weights.pth"))
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
    class_names = ['cat', 'dog', 'bird', 'fish']
    return {"prediction": class_names[predicted.item()]}
