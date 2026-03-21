import torch
from PIL import Image
from torchvision import transforms
from model import CNN

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Load model
model = CNN(num_classes=6)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((150, 150)),
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