import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

app = FastAPI(title = "Image Classifier API")

#CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load Model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# ROUTES
@app.get("/")
def home():
    return {"message" : "Image Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
        return {
            "filename" : file.filename,
            "prediction" : classes[pred]
        }
    except Exception as e:
        return {"error" : str(e)}