import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# Transforms
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])
print("Starting script...")

# Dataset
train_data = datasets.ImageFolder("data/train", transform=transform)
print("Dataset loaded")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
print("Dataloader ready")

# Model
model = CNN(num_classes=len(train_data.classes))
print("Model initialized")

# Loss & Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
# Training loop
for epoch in range(5):
    total_loss = 0

    for i, (X, y) in enumerate(train_loader):
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} DONE, Total Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")