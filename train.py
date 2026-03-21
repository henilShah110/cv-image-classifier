import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print("Starting script...")

# Dataset
train_data = datasets.ImageFolder("data/train", transform=transform)
test_data = datasets.ImageFolder("data/test", transform=transform)
print("Dataset loaded")
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)
print("Dataloader ready")

# Model
#model = CNN(num_classes=len(train_data.classes)) #Step 1: Using my own model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT) #Step 2: Using pretrained ResNet18

model = model.to(device)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = torch.nn.Linear(model.fc.in_features, len(train_data.classes))
print("Model initialized")

# Loss & Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            pred_labels = preds.argmax(dim=1)

            correct += (pred_labels == y).sum().item() #add the number of correct predictions
            total += y.size(0)

    return correct / total


print("Starting training...")
# Training loop
for epoch in range(2):
    model.train()
    total_loss = 0
    for i, (X, y) in enumerate(train_loader):
        #print(f"Loading batch {i}")
        X, y = X.to(device), y.to(device)

        #print(f"Forward pass for batch {i}")
        preds = model(X)
        loss = loss_fn(preds, y)

        #print(f"Backward pass for batch {i}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_acc = get_accuracy(model, train_loader)
    test_acc = get_accuracy(model, test_loader)

    print(f"Epoch {epoch+1} DONE, Total Loss: {total_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")