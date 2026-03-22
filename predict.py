import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from model import CNN

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Load model
#model = CNN(num_classes=6)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load image
test_img = input("Enter path to test image: ")
test_img = "data/pred/" + test_img + ".jpg"
img = Image.open(test_img)
img = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img)
    pred = output.argmax(dim=1).item()

print("Prediction:", classes[pred])